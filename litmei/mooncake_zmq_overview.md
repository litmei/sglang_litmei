# SGLang Mooncake 与 ZMQ 通信及绑核机制概述

> 本文档梳理 SGLang 中 **Mooncake**（RDMA KV 传输引擎）和 **ZMQ**（进程间/节点间通信框架）相关的核心代码，重点关注通信拓扑、套接字绑定方式、以及 CPU/NUMA 绑核策略。

> **背景**: 在同构 CPU 平台部署 SGLang 时，管理面数据传输（请求分发、KV cache PD 间传输）与推理过程共享同一网络平面，存在带宽竞争问题。本文档在末尾专辟 [第 6 节](#6-同构平台适配分析通信线程降低与绑核建议) 分析如何**降低通信线程资源占用**以及**对通信线程进行绑核**以隔离管理面与推理面的网络流量。

---

## 1. Mooncake Transfer Engine

### 1.1 概述

Mooncake 是一个基于 RDMA（Remote Direct Memory Access）的 KV cache 跨节点传输引擎，在 SGLang 中用于 **PD 分离部署（Prefill/Decode disaggregation）** 场景下，将 prefill 节点的 KV cache 快速传输到 decode 节点。

**核心文件：**

| 文件 | 用途 |
|---|---|
| `python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py` | Mooncake Transfer Engine 核心封装：初始化、内存注册、同步/批量传输 |
| `python/sglang/srt/disaggregation/mooncake/conn.py` | PD 分离架构下 Mooncake 的 KV 管理器，包含发送/接收/传输线程池 |
| `python/sglang/srt/disaggregation/mooncake/utils.py` | Mooncake 工具函数（自定义内存池检查等） |
| `python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py` | HiCache 的 Mooncake 存储后端 |

### 1.2 通信架构

```
┌──────────────────┐        Mooncake RDMA         ┌──────────────────┐
│  Prefill Server  │ ◄──────────────────────────► │  Decode Server   │
│  (MooncakeKVManager, PREFILL mode)              │  (MooncakeKVManager, DECODE mode)
│                  │                              │                  │
│  TransferEngine  │  批量 RDMA write (GPU↔GPU)    │  TransferEngine  │
│  + 传输线程池     │                              │  + 解码线程       │
└──────────────────┘                              └──────────────────┘
        │  ▲                                              │
        │  │ ZMQ IPC (控制面)                              │
        ▼  │                                              ▼
┌──────────────────┐                              ┌──────────────────┐
│  TokenizerManager│                              │  TokenizerManager│
└──────────────────┘                              └──────────────────┘
```

- **数据面（Data Plane）**: Mooncake RDMA — 直接在 GPU 显存之间搬运 KV cache，绕过 CPU。
- **控制面（Control Plane）**: ZMQ IPC — 用于交换连接信息（endpoint、session_id、目标指针等）。

### 1.3 Mooncake TransferEngine 初始化

```python
# mooncake_transfer_engine.py
class MooncakeTransferEngine:
    def __init__(self, hostname, gpu_id=None, ib_device=None):
        from mooncake.engine import TransferEngine
        self.engine = TransferEngine()
        self.hostname = hostname
        self.gpu_id = gpu_id or 0

        # IB 设备选择：支持单值、逗号分隔、JSON 映射、JSON 文件
        if os.environ.get("MC_FORCE_TCP") == "1":
            self.ib_device = ""  # 强制 TCP 时绕过 IB 设备
        else:
            self.ib_device = get_ib_devices_for_gpu(ib_device, self.gpu_id)

        self.initialize(hostname=self.hostname, device_name=self.ib_device)
        self.session_id = f"{self.hostname}:{self.engine.get_rpc_port()}"
```

#### 初始化条件（`maybe_init_shared_mooncake_transfer_engine`）

| 条件 | 触发场景 |
|---|---|
| `disaggregation_mode != "null"` 且 `disaggregation_transfer_backend == "mooncake"` | PD 分离部署 |
| `enable_hierarchical_cache` 且 `hicache_storage_backend == "mooncake"` 且 `SGLANG_HICACHE_MOONCAKE_REUSE_TE` | HiCache Mooncake 存储 |
| `encoder_only` 或 `language_only` 且 `encoder_transfer_backend == "mooncake"` | 编码器分离部署 |
| `enable_elastic_expert_backup` 且 `elastic_ep_backend` 不为 None | 弹性专家备份 |

### 1.4 传输协议与 IB 设备绑定

Mooncake 支持多种底层传输协议，通过 `MOONCAKE_PROTOCOL` 环境变量选择（默认 `"rdma"`）：

| 协议 | 描述 |
|---|---|
| `rdma` | RDMA over InfiniBand / RoCE（默认） |
| `efa` | AWS EFA |
| `tcp` | TCP 回退（`MC_FORCE_TCP=1`） |
| `ascend` | Ascend NPU 场景 |

IB 设备绑定通过 `--disaggregation-ib-device` / `--mooncake-ib-device` 参数指定，支持三种格式：

1. **单值/逗号列表**: `"mlx5_0,mlx5_1"` — 所有 GPU 共享
2. **JSON 映射**: `'{"0": "mlx5_0,mlx5_1", "1": "mlx5_2"}'` — 按 GPU ID 映射
3. **JSON 文件路径**: `"/path/to/config.json"` — 从文件读取映射

在 Ascend NPU 场景下，Mooncake 通过 `hostname:port:npu_{phy_id}` 格式注册端点。

### 1.5 传输线程池与队列

在 `MooncakeKVManager` 的 prefill 模式中，数据传输采用**多线程并发模型**：

```
MooncakeKVManager (PREFILL mode)
│
├─ SGLANG_DISAGGREGATION_THREAD_POOL_SIZE (默认: min(max(4, 0.5*cpu_count//8), 12))
│    决定总线程数
├─ SGLANG_DISAGGREGATION_QUEUE_SIZE (默认: 4)
│    决定传输队列数
│
├─ 每个队列对应一个 ThreadPoolExecutor
│   └─ 每个 Executor 的线程数 = pool_size // queue_size
│
└─ 每个队列一个 daemon 传输工作线程 (transfer_worker)
     ├─ 从 FastQueue 取任务
     ├─ 通过 Mooncake TransferEngine 执行 RDMA 传输
     └─ 支持 staging buffer 机制（异构 TP 场景）
```

**关键环境变量：**

| 变量 | 默认值 | 说明 |
|---|---|---|
| `SGLANG_DISAGGREGATION_THREAD_POOL_SIZE` | `min(max(4, cpu_count//16), 12)` | 传输线程池大小 |
| `SGLANG_DISAGGREGATION_QUEUE_SIZE` | `4` | 传输队列数量（必须 ≤ 线程池大小） |
| `SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT` | `300` | 秒，bootstrap 超时 |
| `SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL` | `5.0` | 秒，心跳间隔 |
| `SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE` | `2` | 最大心跳失败次数 |
| `SGLANG_ENABLE_FAILED_SESSION_PROBE` | `False` | 是否启用失败会话探测 |
| `SGLANG_MOONCAKE_CUSTOM_MEM_POOL` | `None` | 自定义内存池类型（NVLINK/BAREX/INTRA_NODE_NVLINK） |
| `SGLANG_MOONCAKE_SEND_AUX_TCP` | `False` | 辅助数据通过 TCP 而非 RDMA 发送 |
| `MC_FORCE_TCP` | 不设置 | 强制 Mooncake 使用 TCP 而非 RDMA |
| `MOONCAKE_PROTOCOL` | `"rdma"` | 传输协议（rdma/efa/tcp） |
| `MOONCAKE_MASTER` | `None` | Mooncake Store master 地址 |
| `MOONCAKE_CLIENT` | `None` | Mooncake Store client 地址 |
| `MOONCAKE_LOCAL_HOSTNAME` | `"localhost"` | 本地 hostname |

### 1.6 内存注册

Mooncake 需要将 GPU 上 KV cache  buffer 注册到传输引擎：

```python
# 批量注册（KV 数据 + 辅助数据 + state 数据）
engine.batch_register(ptrs, lengths)   # RDMA 内存注册
engine.batch_deregister(ptrs)          # 注销
```

- KV 数据 buffer（`kv_data_ptrs` / `kv_data_lens`）
- 辅助数据 buffer（`aux_data_ptrs` / `aux_data_lens`）
- State 数据 buffer（`state_data_ptrs` / `state_data_lens`，按 state_type 分组）

---

## 2. ZMQ 通信框架

### 2.1 概述

ZMQ（ZeroMQ）是 SGLang 内部各进程之间主要的 **IPC 通信框架**。它覆盖了从 HTTP 请求入队、tokenize、调度、推理到 detokenize 的完整 pipeline。

**核心文件：**

| 文件 | 用途 |
|---|---|
| `python/sglang/srt/utils/network.py` | ZMQ 套接字创建/配置/绑定/连接：`get_zmq_socket`、`get_zmq_socket_on_host`、`config_socket` |
| `python/sglang/srt/managers/tokenizer_manager.py` | TokenizerManager — ZMQ PUSH 发送请求给 Scheduler，PULL 接收 detokenizer 结果 |
| `python/sglang/srt/managers/scheduler.py` | Scheduler — ZMQ PULL 接收 TokenizerManager 请求 |
| `python/sglang/srt/managers/detokenizer_manager.py` | DetokenizerManager — ZMQ PULL 接收 Scheduler 结果 |
| `python/sglang/srt/managers/data_parallel_controller.py` | DP Controller — ZMQ 分发请求到各 DP worker |
| `python/sglang/srt/managers/multi_tokenizer_mixin.py` | MultiTokenizerRouter — ZMQ PUSH 路由到 TokenizerWorker |
| `python/sglang/srt/managers/communicator.py` | `FanOutCommunicator` — 基于 ZMQ 的扇出请求/收集响应原语 |
| `python/sglang/srt/managers/load_snapshot.py` | LoadSnapshot — ZMQ PUSH/PULL 跨节点负载收集 |
| `python/sglang/srt/observability/forward_pass_metrics.py` | FPM — ZMQ PUB/SUB 前向指标广播 |
| `python/sglang/srt/managers/io_struct.py` | ZMQ 序列化/反序列化：`sock_send`/`sock_recv`（支持 pickle 和 msgpack） |

### 2.2 进程间 ZMQ IPC 拓扑

```
                            HTTP Request
                                │
                                ▼
                     ┌─────────────────────┐
                     │  TokenizerManager    │  ← ZMQ PULL (from Detokenizer)
                     │  (tokenize + 路由)   │
                     └────────┬────────────┘
                              │ ZMQ PUSH
                              ▼
              ┌───────────────────────────────┐
              │  DataParallelController (可选) │  ← 负载均衡后分发
              │  ZMQ PUSH 到各 DP rank        │
              └────────┬──────────────────────┘
                       │ ZMQ PUSH (每 DP rank)
                       ▼
              ┌────────────────────┐
              │  Scheduler (推理)   │  ← ZMQ PULL (from Tokenizer/DPC)
              │  ZMQ PUSH (结果)   │
              └────────┬───────────┘
                       │ ZMQ PUSH
                       ▼
              ┌──────────────────────┐
              │  DetokenizerManager   │  ← ZMQ PULL (from Scheduler)
              │  ZMQ PUSH (回传)     │
              └────────┬─────────────┘
                       │ ZMQ PUSH
                       ▼
              ┌──────────────────────┐
              │  TokenizerManager     │  ← 收集最终输出
              └──────────────────────┘
```

#### 2.2.1 单节点模式（IPC）

单节点时使用 **Unix Domain Socket (ipc://)**，由 `PortArgs.init_new()` 生成临时文件路径：

```python
# server_args.py - 单节点
PortArgs(
    tokenizer_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
    scheduler_input_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
    detokenizer_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
    rpc_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
    metrics_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
)
```

#### 2.2.2 多节点 DP Attention 模式（TCP）

DP Attention 跨节点时使用 **TCP 传输**。端口从 `server_args.port + ZMQ_TCP_PORT_DELTA`（233）偏移：

```
port_base = server_args.port + 233
  ├── port_base + 0: tokenizer_ipc_name
  ├── port_base + 1: detokenizer_ipc_name
  ├── port_base + 2: rpc_ipc_name
  ├── port_base + 3: metrics_ipc_name
  ├── port_base + 4: scheduler_input_ipc_name (DP Controller → Scheduler)
  └── port_base + 5: load_collector_ipc_name (多节点 DP)
```

端点格式：`"tcp://{host}:{port}"`

### 2.3 ZMQ 套接字类型与配置

| 套接字类型 | 使用场景 |
|---|---|
| `zmq.PUSH` | 发送端：TokenizerManager → Scheduler, Scheduler → Detokenizer, DP Controller → Scheduler |
| `zmq.PULL` | 接收端：Scheduler 接收请求, Detokenizer 接收结果 |
| `zmq.PAIR` | 一对一对等通信 |
| `zmq.DEALER` / `zmq.REQ` / `zmq.REP` | 请求-响应模式（RPC） |
| `zmq.PUB` / `zmq.SUB` | 发布-订阅：forward_pass_metrics PUB 指标，外部 Dynamo planner SUB 订阅 |

#### 缓冲区配置（`config_socket`）

```python
def config_socket(socket, socket_type):
    # 根据总内存和可用内存动态决定 ZMQ 缓冲区大小
    if total_mem > 32 and available_mem > 16:  # >32GB 总内存且 >16GB 可用
        buf_size = 512 MB
    else:
        buf_size = -1 (系统默认)

    # PUSH: 设置 SNDHWM=0 (无限), SNDBUF=buf_size
    # PULL: 设置 RCVHWM=0 (无限), RCVBUF=buf_size
    # DEALER/REQ/REP/PAIR: 同时设置收发
```

#### 安全默认绑定（CVE-2026-3060）

```python
# get_zmq_socket_on_host 默认绑定到 127.0.0.1
if host is None:
    host = "127.0.0.1"  # 仅本地访问，防止未授权外部连接
```

### 2.4 关键 ZMQ 通信路径

#### 2.4.1 请求管线

**单 DP（无 DP Controller）:**
```
TokenizerManager → ZMQ PUSH → Scheduler (ZMQ PULL)
```

**多 DP（有 DP Controller）:**
```
TokenizerManager → ZMQ PUSH → DP Controller (ZMQ PULL)
    → 负载均衡 → ZMQ PUSH → Scheduler[0..N-1] (ZMQ PULL, 各 DP rank)
```

#### 2.4.2 结果回传

```
Scheduler → ZMQ PUSH → Detokenizer (ZMQ PULL) → [decode]
    → ZMQ PUSH → TokenizerManager (ZMQ PULL, recv_from_detokenizer)
```

#### 2.4.3 LoadSnapshot（DP 负载均衡）

```
Scheduler (每 DP rank)
    │ ZMQ PUSH (跨节点时)
    ▼
ZmqShmLoadSnapshotReader (node 0, ZMQ PULL)
    │ 写入
    ▼
/dev/shm mmap ← TokenizerManager / DPController 读取
```

触发条件：`enable_dp_attention && nnodes > 1` 或 `SGLANG_LOAD_SNAPSHOT_USE_ZMQ=1`

ZMQ PULL 套接字的所有权通过 `zmq_reader_owner()` 决定：
- DP Controller 在 load-aware 模式下拥有
- 否则由 TokenizerManager 或 MultiTokenizerRouter 拥有

#### 2.4.4 Forward Pass Metrics（ZMQ PUB/SUB）

```python
# 每轮前向传播后，Scheduler 通过 ZMQ PUB 广播指标
# 外部消费者（如 Dynamo planner）通过 ZMQ SUB 订阅
self.server_args.forward_pass_metrics_ipc_name  # ZMQ 端点地址
```

#### 2.4.5 编码器分离部署

```python
# encoder_transfer_backend 支持三种模式
ENCODER_TRANSFER_BACKEND_CHOICES = [
    "zmq_to_scheduler",   # 编码器输出经 ZMQ 发送到 Scheduler
    "zmq_to_tokenizer",   # 编码器输出经 ZMQ 发送到 Tokenizer
    "mooncake",           # 编码器输出通过 Mooncake RDMA 传输
]
```

### 2.5 FanOutCommunicator

`FanOutCommunicator` 是一个通用的 ZMQ 扇出请求 / 收集响应原语：

- **queueing 模式**: 请求串行化，并发调用者在 FIFO 队列中等待
- **watching 模式**: 并发调用者共享同一个进行中的请求，都收到相同结果

使用场景：DP Attention 中对所有 DP rank 广播请求并等待全部响应。

### 2.6 多 Tokenizer 模式

当 `tokenizer_worker_num > 1` 时，多个独立的 `TokenizerWorker` 进程通过 ZMQ PUSH 将 tokenized 请求发送给 `MultiTokenizerRouter`，由后者统一通过 ZMQ PUSH 路由到 DP Controller 或 Scheduler：

```
TokenizerWorker[0..N-1] → ZMQ PUSH
                                ▼
                    MultiTokenizerRouter (ZMQ PULL)
                                │ ZMQ PUSH
                                ▼
                    DP Controller / Scheduler
```

---

## 3. CPU / NUMA 绑核机制

### 3.1 概述

SGLang 支持多种 CPU 绑核策略，用于提升多 GPU / 多节点场景下的性能，减少跨 NUMA 节点内存访问延迟。

**相关环境变量 & 配置：**

| 变量/参数 | 默认值 | 说明 |
|---|---|---|
| `SGLANG_SET_CPU_AFFINITY` | `False` | 是否启用 GPU 进程 CPU 亲和性绑定 |
| `SGLANG_NUMA_BIND_V2` | 不设置 | 是否启用 NUMA v2 绑核（通过 numactl 包装子进程） |
| `--numa-node` | `None` | 手动指定 NUMA 节点映射 |

### 3.2 GPU 进程 CPU 亲和性绑定（`set_gpu_proc_affinity`）

**位置**: `python/sglang/srt/utils/common.py`

**触发点**: `python/sglang/srt/managers/scheduler.py` 中的 `configure_scheduler_process()`

```python
if envs.SGLANG_SET_CPU_AFFINITY.get():
    set_gpu_proc_affinity(
        server_args.pp_size, server_args.tp_size, server_args.nnodes, gpu_id
    )
```

**绑核算法：**

```python
def set_gpu_proc_affinity(pp_size, tp_size, nnodes, gpu_id):
    # 1. 计算每节点的 TP 大小
    nnodes_per_tp_group = max(nnodes // pp_size, 1)
    tp_size_per_node = tp_size // nnodes_per_tp_group

    # 2. 分配物理核心
    total_pcores = psutil.cpu_count(logical=False)  # 物理核心总数
    num_cores_bind = total_pcores // tp_size_per_node  # 每个 TP 进程的核心数

    # 3. 计算起始核心
    start_cpu_id = (gpu_id * num_cores_bind) % total_pcores
    end_cpu_id = start_cpu_id + num_cores_bind

    # 4. 考虑超线程 (HT)
    if HT enabled:
        bind_cpu_ids = lower_cores + upper_cores  # 物理+逻辑核心
    else:
        bind_cpu_ids = [start..end)  # 仅物理核心

    # 5. 设置进程 CPU 亲和性
    p.cpu_affinity(bind_cpu_ids)
```

**示例**（4 GPU, TP=4, 单节点, 32 物理核心, HT 开启）:

```
GPU 0: CPU 0-7 + 32-39
GPU 1: CPU 8-15 + 40-47
GPU 2: CPU 16-23 + 48-55
GPU 3: CPU 24-31 + 56-63
```

### 3.3 NUMA 绑定（`numactl` v2 模式）

**位置**: `python/sglang/srt/utils/numa_utils.py`

**原理**: 通过 `configure_subprocess()` 上下文管理器，在 spawn 子进程前将 multiprocessing 的可执行文件包装为 `numactl {args} python` 脚本，使得子进程在启动时就处在正确的 NUMA 节点上。

```python
@contextmanager
def configure_subprocess(server_args, gpu_id):
    if envs.SGLANG_NUMA_BIND_V2.get():
        numa_node = get_numa_node_if_available(server_args, gpu_id)
        if numa_node is not None:
            numactl_args = _numactl_cpu_mem_args(numa_node, gpu_id)
            # 验证 numactl 可用性
            executable = _create_numactl_executable(numactl_args)
            # 包装: numactl --cpunodebind=N --membind=N python original_script.py
            with _mp_set_executable(executable=executable):
                yield
```

**NUMA 节点发现** (`get_numa_node_if_available`):

1. `--numa-node` 指定 → 直接使用
2. 否则调用 `_query_numa_node_for_gpu(gpu_id)` → 通过 NVML 查询 GPU 所在 NUMA 节点
3. 如果查询失败或权限不足 → 返回 None（跳过绑定）

**NUMA 绑定回退**:

旧的 NUMA 绑定逻辑（`SGLANG_NUMA_BIND_V2` 未设置时）在 `scheduler.py` 中通过 `get_numa_node_if_available` + `numa_bind_to_node` 实现。

### 3.4 CPU 平台的 OpenMP 线程绑定

**位置**: `python/sglang/srt/distributed/bootstrap.py`

在 CPU 平台（AMX 或 ARM64）上，通过 `sgl-kernel` 的 `init_cpu_threads_env` 将 OpenMP 线程绑定到指定 CPU 核心：

```python
def _init_cpu_threads_env(tp_size, tp_rank, local_omp_cpuid):
    if is_cpu_amx_available or is_cpu_arm64:
        torch.ops.sgl_kernel.init_cpu_threads_env(local_omp_cpuid)
        os.environ["LOCAL_SIZE"] = str(tp_size)
        torch.ops.sgl_kernel.initialize(tp_size, tp_rank)
```

### 3.5 绑核策略对比

| 策略 | 启用方式 | 作用范围 | 机制 |
|---|---|---|---|
| CPU Affinity | `SGLANG_SET_CPU_AFFINITY=1` | 当前进程 | `psutil.Process.cpu_affinity()` |
| NUMA v2 | `SGLANG_NUMA_BIND_V2=1` | spawn 子进程 | `numactl` 包装可执行文件 |
| NUMA 旧版 | 自动（检测到 NUMA） | 当前进程 | `numa_bind_to_node()` |
| OpenMP 绑定 | CPU 平台自动 | OpenMP 线程 | `sgl_kernel.init_cpu_threads_env` |

---

## 4. Mooncake 与 ZMQ 的交互

### 4.1 PD 分离中的混合通信

在 PD 分离部署中，Mooncake 和 ZMQ 协同工作：

```
Phase 1: Bootstrap (ZMQ)
────────────────────────
Prefill Scheduler ──ZMQ──► Decode Scheduler: 交换 endpoint、session_id、目标指针

Phase 2: KV Transfer (Mooncake RDMA)
─────────────────────────────────────
Prefill Scheduler ──RDMA──► Decode Scheduler: 批量传输 KV cache 数据

Phase 3: 心跳/控制 (ZMQ)
────────────────────────
Prefill Scheduler ◀──ZMQ──► Decode Scheduler: 健康检查、staging 通知
```

**关键数据流：**

1. `TransferInfo` 和 `KVArgsRegisterInfo` 通过 ZMQ 序列化（`from_zmq` 方法）在 prefill 和 decode 之间交换
2. `mooncake_session_id` 通过 ZMQ 控制面传递，用于 RDMA 数据面的 session 标识
3. `CHUNK_READY` 消息通过 ZMQ 从 prefill 发送到 decode，通知 staging chunk 传输完成

### 4.2 辅助数据传输

通过 `SGLANG_MOONCAKE_SEND_AUX_TCP` 环境变量控制辅助数据（aux data：序列化后的采样参数等）的传输方式：

- `False`（默认）: 通过 Mooncake RDMA 发送
- `True`: 通过独立的 **TCP 连接** 发送（绕过 RDMA，适用于辅助数据量小的场景）

### 4.3 编码器分离部署的传输后端

```
encoder_transfer_backend
    ├── "zmq_to_scheduler"  ─── 编码器输出 → ZMQ → Scheduler
    ├── "zmq_to_tokenizer"  ─── 编码器输出 → ZMQ → TokenizerManager
    └── "mooncake"          ─── 编码器输出 → Mooncake RDMA → 目标节点
```

---

## 5. 总结

| 维度 | Mooncake | ZMQ |
|---|---|---|
| **主要用途** | 跨节点 GPU 显存间 KV cache 传输 | 进程间/节点间控制消息、请求路由 |
| **传输层** | RDMA (InfiniBand/RoCE/EFA) | TCP / Unix Domain Socket |
| **数据面** | 批量 GPU↔GPU RDMA write | PUSH/PULL/PAIR/PUB-SUB 模式 |
| **绑核/绑定** | IB 设备绑定（按 GPU 映射） | 默认绑定 `127.0.0.1`（CVE 安全加固） |
| **并发模型** | 可配置线程池 + 传输队列 | 异步 I/O（zmq.asyncio） |
| **主要场景** | PD 分离、HiCache、弹性 EP | 内部管线通信、负载均衡、指标广播 |
| **关键配置** | `--disaggregation-ib-device`, `MOONCAKE_PROTOCOL` | `ZMQ_TCP_PORT_DELTA=233`, `PortArgs` |
| **绑核** | IB 设备与 GPU 的 1:1/N:1 映射 | 无直接绑核，但用于控制 NUMA 绑核的进程管理 |

---

## 6. 同构平台适配分析：通信线程降低与绑核建议

> **场景**: 在同构 CPU 平台上部署 SGLang，计算与管理全在 CPU 上执行，管理面数据传输（请求分发、KV cache PD 间传输）与推理共用同一网络平面，存在带宽竞争。

### 6.1 通信线程全景清单

以下列出 SGLang 中所有与管理面数据传输相关的线程/进程及其网络行为：

| # | 线程/进程 | 位置 | 网络行为 | 是否可绑核 | 是否可调优数量 |
|---|---|---|---|---|---|
| 1 | **Mooncake transfer_worker** (N 个) | `mooncake/conn.py:1263` | RDMA 写 KV cache（数据面） | ✅ 可以 | ✅ `SGLANG_DISAGGREGATION_THREAD_POOL_SIZE` |
| 2 | **Mooncake bootstrap_thread** (1 个) | `mooncake/conn.py:1476` | ZMQ PULL 接收 decode 注册信息（控制面） | ✅ 可以 | ❌ 固定 1 个 |
| 3 | **Mooncake decode_thread** (1 个) | `mooncake/conn.py:1602` | ZMQ PULL 接收 prefill 状态同步（控制面） | ✅ 可以 | ❌ 固定 1 个 |
| 4 | **Mooncake heartbeat_checker** (1 个) | `common/conn.py:882` | HTTP GET 健康检查（控制面） | ✅ 可以 | ❌ 固定 1 个 |
| 5 | **Mooncake failed_session_probe** (1 个，可选) | `mooncake/conn.py:1761` | Mooncake send_probe RDMA 探测 | ✅ 可以 | ❌ 固定 1 个 |
| 6 | **Mooncake ThreadPoolExecutor 工作线程** (M 个) | `mooncake/conn.py:192` | 在 transfer_worker 内部执行 RDMA 传输 | ✅ 通过 transfer_worker 间接绑定 | ✅ 同 #1 |
| 7 | **ZMQ I/O 线程** (各 Context 独立) | 分散在各 Manager | 所有 ZMQ 套接字的底层 I/O 轮询 | ⚠️ 间接（设置线程亲和性困难） | ✅ `zmq.Context(io_threads=N)` |
| 8 | **DP Controller** (1 个进程) | `data_parallel_controller.py` | ZMQ PULL 接收请求 + ZMQ PUSH 分发 | ✅ 进程级（`SGLANG_SET_CPU_AFFINITY`） | ❌ 固定 1 个 |
| 9 | **TokenizerManager 线程** (1 个进程) | `tokenizer_manager.py` | ZMQ PUSH 发送到 Scheduler + ZMQ PULL 接收结果 | ✅ 进程级 | ❌ 固定 1 个 |
| 10 | **DetokenizerManager 线程** (1 个进程) | `detokenizer_manager.py` | ZMQ PULL 接收结果 + ZMQ PUSH 回传 | ✅ 进程级 | ❌ 固定 1 个 |

### 6.2 降低通信线程资源占用的策略

#### 策略 1：缩减 Mooncake 传输线程池

```python
# 当前默认值（mooncake/conn.py:181-182）:
# cpu_count * 0.5 // 8, clamped to [4, 12]
transfer_thread_pool_size = min(max(4, int(0.5 * cpu_count) // 8), 12)

# 在 CPU 平台建议:
# 设置 SGLANG_DISAGGREGATION_THREAD_POOL_SIZE=2 或更小
# 同时降低 SGLANG_DISAGGREGATION_QUEUE_SIZE=1
```

- 当前算法在 64 核 CPU 上默认生成 4 个传输线程，在 128 核上生成 8 个
- **同构平台建议**: 将 `SGLANG_DISAGGREGATION_THREAD_POOL_SIZE` 设置为 `2`（最低建议值），同时将 `SGLANG_DISAGGREGATION_QUEUE_SIZE` 设置为 `1`，以最大限度减少与推理争抢 CPU 时间片
- 每队列的 `ThreadPoolExecutor` 线程数 = `pool_size // queue_size`

#### 策略 2：关闭非必要 Mooncake 功能

| 功能 | 环境变量 | 建议值 | 说明 |
|---|---|---|---|
| 失败会话探测 | `SGLANG_ENABLE_FAILED_SESSION_PROBE` | `False` | 关闭后可减少 1 个 RDMA 探测线程 |
| Staging buffer | `SGLANG_DISAGG_STAGING_BUFFER` | `False` | 关闭异构 TP staging 路径，减少内部线程协调开销 |
| 辅助数据 TCP 发送 | `SGLANG_MOONCAKE_SEND_AUX_TCP` | `True` | 将小体量辅助数据走独立 TCP，避免占用 RDMA 队列 |
| Trace | `--enable-trace` | 不设置 | 关闭后 transfer_worker 内无 trace 埋点开销 |

#### 策略 3：缩减 ZMQ I/O 线程数

ZMQ `Context(io_threads=N)` 在每个进程中创建 N 个 I/O 线程用于后端轮询。当前的 io_threads 使用情况：

```python
# 各进程的 ZMQ Context 配置
TokenizerManager:       zmq.asyncio.Context(2)     # 默认 2 个 I/O 线程
DetokenizerManager:     zmq.Context(2)
Scheduler:              zmq.Context(2)
DP Controller:          zmq.Context(1 + dp_size)   # 1 + DP 数量
Mooncake disag:         zmq.Context()              # 默认 1 个
MultiTokenizerRouter:   zmq.Context()              # 默认 1 个
```

**问题**: `DP Controller` 的 `Context(1 + dp_size)` 在 DP size 较大时会产生过多 I/O 线程。对于 CPU 同构平台，建议修改为固定小值。

**建议**: 在 CPU 同构部署中，将所有 ZMQ Context 的 io_threads 统一设置为 `1` 或 `2`。对于 DP Controller，当前代码 `zmq.Context(1 + server_args.dp_size)` 可硬编码为 `zmq.Context(2)` 以节省线程资源。

#### 策略 4：降低心跳频率或关闭心跳

```python
# 当前默认
SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL = 5.0    # 5 秒
SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE = 2   # 2 次失败后判定下线

# 同构平台建议（降低心跳网络开销）
SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL = 30.0   # 30 秒
SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE = 3   # 3 次失败后判定下线
```

### 6.3 绑核策略分析

#### 6.3.1 现有绑核机制回顾

当前 SGLang 支持两种绑核方式：

1. **`SGLANG_SET_CPU_AFFINITY=1`** — 进程级 CPU 亲和性绑定（`set_gpu_proc_affinity`）
   - 按 `gpu_id` 在物理核心范围内线性分配
   - 考虑超线程（HT on/off）
   - **对所有进程一致生效**，无法精细区分管理线程与计算线程

2. **NUMA 绑定**（`SGLANG_NUMA_BIND_V2=1` + `--numa-node`）
   - 通过 `numactl` 包装子进程
   - 将进程绑定到指定 NUMA 节点
   - 同样无法区分进程内不同线程

**关键问题**: 现有绑核机制都是 **进程级** 的，无法将 Mooncake 传输线程、ZMQ I/O 线程绑定到与推理线程不同的 CPU 核心上。

#### 6.3.2 建议的绑核方案

##### 方案 A：进程分离 + 独立绑核（推荐）

将管理面通信分离到独立进程，不同进程走不同绑核策略：

```
┌──────────────────────────┐     ┌──────────────────────────┐
│  推理进程 (Scheduler)     │     │  通信代理进程 (Proxy)     │
│                          │     │                          │
│  SGLANG_SET_CPU_AFFINITY │     │  手动绑核到独立 CPU 集合  │
│  → 绑定到 CPU 0-31       │     │  → 绑定到 CPU 32-47      │
│                          │     │                          │
│  ZMQ PULL recv           │◄───►│  ZMQ PUSH send (转发请求) │
│  Mooncake RDMA (接收)    │◄───►│  Mooncake RDMA (发送)    │
└──────────────────────────┘     └──────────────────────────┘
```

**优点**: 彻底隔离计算面与管理面的 CPU 争抢
**代价**: 需要额外开发通信代理进程，增加一次 ZMQ 转发延迟

##### 方案 B：线程级绑核（中等改造成本）

直接在 Mooncake 和 ZMQ 的关键线程上设置 CPU 亲和性：

```python
# 在 start_prefill_thread 的 bootstrap_thread 中:
def bootstrap_thread():
    # 绑定到指定 CPU 核心
    import psutil
    proc = psutil.Process(os.getpid())
    existing_affinity = proc.cpu_affinity()
    # 将 bootstrap 线程绑定到独立核心（例如 CPU 0）
    threading.current_thread().name = "mooncake-bootstrap"
    # 通过 taskset 或直接设置
    os.sched_setaffinity(0, {0})  # 仅示例
    ...
```

**需要修改的关键位置**:

| 线程 | 文件 | 行号 | 建议绑核策略 |
|---|---|---|---|
| `bootstrap_thread` | `mooncake/conn.py:1476` | 绑定到专用管理核心（如 CPU 0） |
| `decode_thread` | `mooncake/conn.py:1602` | 绑定到专用管理核心（如 CPU 0） |
| `transfer_worker` (N 个) | `mooncake/conn.py:1263` | 按 shard 分散到不同管理核心 |
| `heartbeat_checker` | `common/conn.py:885` | 绑定到专用管理核心 |
| `_failed_session_probe_loop` | `mooncake/conn.py:1761` | 绑定到专用管理核心 |

**Python 中设置线程绑核的注意事项：**

```python
# 方法 1：使用 os.sched_setaffinity（仅限 Linux）
import os
import threading

def bind_current_thread_to_cpu(cpu_ids: set):
    """将当前线程绑定到指定 CPU 核心列表。"""
    os.sched_setaffinity(0, cpu_ids)  # 0 表示当前线程

# 在 thread target 开头调用
threading.Thread(target=lambda: (
    bind_current_thread_to_cpu({0, 1}),
    actual_worker_function()
)).start()
```

**注意**: `os.sched_setaffinity(0, ...)` 在 Linux 上作用于**调用线程**（而非进程），这是 Python 中实现线程级绑核的标准方式。

##### 方案 C：利用 ZMQ 套接字隔离（低改造成本）

将不同网络特征的通信拆分到不同 ZMQ Context，创建独立的 I/O 线程池：

```python
# 当前：所有 ZMQ 套接字共享一个 Context
self._zmq_ctx = zmq.Context()

# 建议：按通信类型拆分
self._zmq_ctx_control = zmq.Context(io_threads=1)   # 控制面（心跳、状态同步）
self._zmq_ctx_data = zmq.Context(io_threads=1)       # 数据面（请求转发）

# 控制面套接字用 control context
self.server_socket = self._zmq_ctx_control.socket(zmq.PULL)

# 数据面套接字用 data context
self.data_socket = self._zmq_ctx_data.socket(zmq.PUSH)
```

**优点**: 控制面和数据面的 ZMQ I/O 线程物理隔离，互不影响
**代价**: 需修改 `CommonKVManager` 初始化代码

#### 6.3.3 网络平面隔离建议

对于同构平台**管理面与推理面共用一个网络平面**的问题，建议的组合方案：

```
1. 物理层面（推荐）:
   ┌─────────────────────────────────────────────────────┐
   │  同一节点上，使用不同网卡/不同端口:                   │
   │  - 管理面流量: eth0 (请求分发、KV transfer 控制面)    │
   │  - 推理面流量: eth1 (NCCL all-reduce 等)            │
   │  - Mooncake RDMA: ib0 (独立 IB 端口)                │
   └─────────────────────────────────────────────────────┘

2. 若只有一张网卡，则:
   ┌─────────────────────────────────────────────────────┐
   │  a) 降低 Mooncake 线程数（策略 1）                   │
   │  b) 将通信线程绑定到远离推理线程的 CPU 核心 (方案 B)  │
   │  c) 降低心跳频率（策略 4）                            │
   │  d) 关闭非必要功能（策略 2）                          │
   └─────────────────────────────────────────────────────┘
```

### 6.4 实施路线图

| 步骤 | 改动 | 工作量 | 收益 |
|---|---|---|---|
| 1 | 设置环境变量调优（策略 1+2+4） | 无需改代码 | ★★★ 中 |
| 2 | ZMQ Context io_threads 调优（策略 3） | 小（改几行参数） | ★★ 低-中 |
| 3 | Mooncake 传输线程绑核（方案 B） | 中（改 3-5 个线程） | ★★★★ 高 |
| 4 | ZMQ Context 按通信类型拆分（方案 C） | 中（重构 conn.py） | ★★★ 中 |
| 5 | 通信代理独立进程（方案 A） | 大（新进程架构） | ★★★★★ 最高 |
