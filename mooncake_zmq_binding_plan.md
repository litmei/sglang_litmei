# SGLang Mooncake/ZMQ 线程绑核与 NUMA 绑定实现方案

> 本文档描述将 SGLang 进程内 Mooncake 与 ZMQ 相关线程（含其底层 C++ 库线程）绑定到指定 CPU 核心 / NUMA 节点的完整方案，包括设计原理、需实现的内容、代码改动点、风险与落地步骤。
>
> 配套背景见 [mooncake_zmq_overview.md](./mooncake_zmq_overview.md)。

---

## 1. 目标与范围

### 1.1 目标

在同构 CPU 平台（无专用管理核/无独立网卡）上，将 Mooncake 传输线程、ZMQ I/O 线程及其底层 C++ 库创建的线程，**集中绑定到一组与推理线程不重叠的 CPU 核心**，并绑定 NUMA 内存策略，避免管理面通信流量与推理计算争抢 CPU 和内存带宽。

### 1.2 范围

| 在范围内 | 不在范围内 |
|---|---|
| SGLang Scheduler 进程内的 Python 线程（transfer_worker / bootstrap / decode / heartbeat / probe） | 由用户手动启动的外部进程（`mooncake_master`、`mooncake_store_service`、独立 `mooncake_client`） |
| SGLang 进程内构造的 C++ 对象拉起的工作线程（`TransferEngine` 的 RDMA polling 线程、`MooncakeDistributedStore` 的工作线程、ZMQ Context I/O 线程） | GPU 推理线程本身的绑核（已由 `SGLANG_SET_CPU_AFFINITY` 处理） |
| 新增 C++ 绑核 + 绑内存函数 | NCCL / cuBLAS 等推理库内部线程 |

### 1.3 非目标

- 不重写 Mooncake / ZMQ 通信库
- 不改变 PD 分离的进程拓扑
- 不负责外部 `mooncake_master` 进程的绑核（这部分需通过启动命令 `numactl` 包装，见第 9 节）

### 1.4 部署模式与绑核必要性的关系

> **结论先行**：即使采用"外接 mooncake"部署模式（用户独立拉起 `mooncake_master` / `mooncake_store_service` 等服务进程），本方案**仍然必须做**。外接模式只能省掉 HiCache 场景下的 `MooncakeStore` 服务端线程，**无法消除** PD 分离场景下的 `MooncakeTransferEngine` + `MooncakeKVManager` 线程，也无法消除 ZMQ I/O 线程。

#### 1.4.1 Mooncake 的两种接入模式

SGLang 接入 mooncake 有两种方式，但**只对 HiCache 场景的 `MooncakeStore` 生效**，与 PD 分离无关：

| 模式 | 触发条件 | 影响范围 |
|---|---|---|
| **内嵌模式** | `standalone_storage=True` 或 `MOONCAKE_GLOBAL_SEGMENT_SIZE > 0` | SGLang 进程承担 store 角色，`MooncakeDistributedStore()` C++ 对象在 SGLang 进程内构造并拉起 store 工作线程 |
| **外接模式** | 用户独立启动 `mooncake_master` / `mooncake_store_service` / `mooncake_client` | SGLang 进程仅构造轻量客户端对象连接外部进程；store 服务端线程在外部进程内 |

#### 1.4.2 各使用场景与组件依赖

| 场景 | 使用的 mooncake 组件 | 受"外接/内嵌"影响？ |
|---|---|---|
| **PD 分离**（prefill/decode disaggregation） | `MooncakeTransferEngine` + `MooncakeKVManager` | ❌ 不受影响 |
| **HiCache**（KV cache 分层存储） | `MooncakeStore` | ✅ 受影响 |
| **Encoder 分离**（mooncake 后端） | `MooncakeTransferEngine` | ❌ 不受影响 |
| **Elastic Expert Backup** | `MooncakeTransferEngine` | ❌ 不受影响 |

#### 1.4.3 关键事实

**1. PD 分离场景下 `MooncakeTransferEngine` 总在 SGLang 进程内构造**

[mooncake_transfer_engine.py:305-345](python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py#L305) 的 `maybe_init_shared_mooncake_transfer_engine` 明确触发条件，第一条即 PD 分离：

```python
use_mooncake_te = (
    (
        server_args.disaggregation_mode != "null"
        and server_args.disaggregation_transfer_backend == "mooncake"
    )   # ← PD 分离场景，必然在 SGLang 进程内构造 TransferEngine()
    or (server_args.enable_hierarchical_cache and ...)  # HiCache
    or (server_args.encoder_only and ...)               # Encoder 分离
    ...
)
```

只要走 PD 分离 + mooncake 后端，SGLang 进程内就**必然**构造 `TransferEngine()` C++ 对象（[mooncake_transfer_engine.py:122](python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py#L122)），它会拉起 **RDMA polling 线程**，该线程无法通过外接模式消除。

**2. `MooncakeKVManager` 的所有 Python 线程都在 SGLang 进程内**

[conn.py:156](python/sglang/srt/disaggregation/mooncake/conn.py#L156) 的 `MooncakeKVManager` 在 Scheduler 进程内拉起 `transfer_worker` / `bootstrap_thread` / `decode_thread` / `heartbeat_checker` / `ThreadPoolExecutor` 工作线程。这些线程做的是 KV cache 的 RDMA 传输调度，**与 mooncake store 是否外接完全无关**。

**3. ZMQ 是 SGLang 全局通信骨架，与 mooncake 部署模式无关**

无论是否使用 mooncake，SGLang 的请求管线都依赖 ZMQ IPC：

```
TokenizerManager → ZMQ PUSH → Scheduler (ZMQ PULL)
                              ↓ ZMQ PUSH
                          Detokenizer
```

SGLang 进程一启动，`zmq.Context()` 就会在 Scheduler 进程内被创建，libzmq 的 I/O 线程随之启动。这部分**无法通过外接 mooncake 消除**。

**4. 外接模式下 SGLang 进程内仍存在 `MooncakeDistributedStore` 客户端对象**

即便外接，SGLang 进程内仍会构造一个 `MooncakeDistributedStore()` 客户端对象（[mooncake_store.py:370](python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py#L370)）连接外部服务。该客户端对象是否拉起工作线程需用 `strace` 实测（方法见第 13.1 节）。

#### 1.4.4 外接模式对各线程的消除效果

| 线程类别 | 外接模式能否消除 | 说明 |
|---|---|---|
| `MooncakeTransferEngine` 的 RDMA polling 线程 | ❌ 不能 | PD 分离必需，在 SGLang 进程内构造 |
| `MooncakeKVManager` 的 transfer_worker / bootstrap / decode / heartbeat / probe | ❌ 不能 | PD 分离必需，做 KV 传输调度 |
| `ThreadPoolExecutor` 工作线程 | ❌ 不能 | 同上 |
| `MooncakeStore` 的 store 服务端工作线程（仅 HiCache 内嵌模式） | ✅ 能 | 外接后由外部进程承担 |
| `MooncakeDistributedStore` 客户端线程（仅 HiCache） | ⚠️ 部分 | 客户端对象仍在 SGLang 进程内，是否拉起线程待实测 |
| ZMQ I/O 线程 | ❌ 不能 | 与 mooncake 部署模式无关 |

#### 1.4.5 各部署场景下本方案的必要性

> **重要区分**：PD 分离的 KV 传输（`--disaggregation-transfer-backend mooncake`）与 HiCache 的 KV 存储（`--enable-hierarchical-cache --hicache-storage-backend mooncake`）是**两条独立配置**。前者走 `MooncakeTransferEngine` + `MooncakeKVManager`，必在 SGLang 进程内；后者走 `MooncakeStore`，可内嵌或外接。两者可组合使用：根据 [mooncake_store/README.md:409-414](python/sglang/srt/mem_cache/storage/mooncake_store/README.md#L409)，PD 分离场景下只有 **prefill worker** 可叠加 HiCache，decode worker 不涉及 store。

| 部署场景 | 外接 mooncake store？ | 本方案是否必要？ | 原因 |
|---|---|---|---|
| **纯 PD 分离**（无 HiCache） | N/A（不涉及 store） | ✅ 必要 | `MooncakeTransferEngine` + `MooncakeKVManager` 线程必在 SGLang 进程内（prefill 和 decode 都有） |
| **PD 分离 + prefill 启 HiCache**（外接 store） | 是 | ✅ 必要 | prefill 进程内有 `MooncakeTransferEngine` + `MooncakeKVManager` + `MooncakeStore` 客户端 + ZMQ；decode 进程内有前两者 + ZMQ。外接仅省 prefill 的 store 服务端线程 |
| **PD 分离 + prefill 启 HiCache**（内嵌 store） | 否 | ✅ 必要 | 同上 + prefill 进程内额外有 `MooncakeStore` 服务端线程 |
| **纯 HiCache**（无 PD 分离） | 是 | ✅ 必要 | 仍有 `MooncakeStore` 客户端 + ZMQ（若启用 `SGLANG_HICACHE_MOONCAKE_REUSE_TE` 还有 `MooncakeTransferEngine`） |
| **纯 HiCache**（无 PD 分离） | 否（内嵌） | ✅ 必要 | 同上 + `MooncakeStore` 服务端线程 |
| **纯单节点推理**（无 mooncake） | N/A | ⚠️ 仅 ZMQ 部分 | ZMQ I/O 线程仍在 SGLang 进程内，流量小可绑可不绑 |
| **Encoder 分离**（mooncake 后端） | N/A | ✅ 必要 | `MooncakeTransferEngine` 必在 SGLang 进程内 |

#### 1.4.6 结论

外接 mooncake 模式**不能省掉本方案**。本方案覆盖的 SGLang 进程内线程中，绝大多数（PD 分离的全部线程、ZMQ I/O 线程）与 mooncake 部署模式无关，必须在 SGLang 进程内通过本方案绑核。外接模式仅能省掉 HiCache 内嵌场景下的 `MooncakeStore` 服务端线程，属于边际收益。

因此，**无论采用内嵌还是外接 mooncake，本方案都需要实施**（除非纯单节点无 PD 分离且不关心 ZMQ 绑核）。

---

## 2. 现状分析

### 2.1 Mooncake 线程模型（在 SGLang 进程内）

以 PD 分离为例，`MooncakeKVManager` 在 Scheduler 进程内创建以下线程（位置以 `python/sglang/srt/disaggregation/mooncake/conn.py` 为准）：

| 线程 | 创建位置 | 数量 | 角色 |
|---|---|---|---|
| `transfer_worker` | [conn.py:206-215](python/sglang/srt/disaggregation/mooncake/conn.py#L206) | `SGLANG_DISAGGREGATION_QUEUE_SIZE`（默认 4） | 从 FastQueue 取任务，提交 RDMA 传输 |
| `ThreadPoolExecutor` 工作线程 | [conn.py:192-197](python/sglang/srt/disaggregation/mooncake/conn.py#L192) | `SGLANG_DISAGGREGATION_THREAD_POOL_SIZE // queue_size`（默认每 executor 1-3 线程） | 执行实际传输/分片逻辑 |
| `bootstrap_thread`（PREFILL） | [conn.py:1600](python/sglang/srt/disaggregation/mooncake/conn.py#L1600) | 1 | 接收 decode 端的预分配通知 |
| `decode_thread`（DECODE） | [conn.py:1671](python/sglang/srt/disaggregation/mooncake/conn.py#L1671) | 1 | 接收 prefill 端的状态/数据通知 |
| `heartbeat_checker` | [common/conn.py:929](python/sglang/srt/disaggregation/common/conn.py#L929) | 1 | 周期 HTTP 探活 prefill 节点 |
| `_failed_session_probe_loop` | [conn.py:228-231](python/sglang/srt/disaggregation/mooncake/conn.py#L228) | 0 或 1 | 失败会话探测（可选） |

底层 C++ 库还会在构造时创建：

| 线程 | 构造触发点 | 创建者 |
|---|---|---|
| RDMA completion polling 线程 | [mooncake_transfer_engine.py:122](python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py#L122) `TransferEngine()` | mooncake C++ |
| `MooncakeDistributedStore` 工作线程 | [mooncake_store.py:370](python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py#L370) `MooncakeDistributedStore()` | mooncake C++ |
| ZMQ Context I/O 线程 | [network.py](python/sglang/srt/utils/network.py) `zmq.Context()` | libzmq C++ |

### 2.2 ZMQ 线程模型

ZMQ 在 SGLang 进程内**不显式创建 Python 线程**，所有 I/O 由 libzmq 内部线程池完成。默认 1 个 I/O 线程，由 `zmq.Context(io_threads=N)` 控制。

### 2.3 现有绑核机制（不够用）

| 机制 | 触发方式 | 粒度 | 不足 |
|---|---|---|---|
| `SGLANG_SET_CPU_AFFINITY=1` | [common.py: set_gpu_proc_affinity](python/sglang/srt/utils/common.py) | 整个进程 | 无法区分通信线程与推理线程 |
| `SGLANG_NUMA_BIND_V2=1` | [numa_utils.py: configure_subprocess](python/sglang/srt/utils/numa_utils.py) | 子进程 | 通过 `numactl` 包装子进程，不作用于进程内线程 |
| CPU 平台 OMP 绑核 | [bootstrap.py: _init_cpu_threads_env](python/sglang/srt/distributed/bootstrap.py) | OpenMP 线程 | 仅 CPU 推理平台启用；绑定 OMP 线程而非通信线程 |
| Python `os.sched_setaffinity` | — | 线程 | 仅 CPU，无 NUMA 内存策略 |

**核心缺口**：现有机制无法在进程内对**特定子线程**同时设置 CPU affinity + NUMA 内存策略。

---

## 3. 设计原理

### 3.1 Linux 线程继承机制（方案成立的基础）

Linux 下 `pthread_create`（Python `threading.Thread` 底层调用）通过 `clone()` 创建新线程时，**自动继承**父线程的以下属性：

| 属性 | 继承机制 | 验证 |
|---|---|---|
| CPU affinity mask | `clone()` 拷贝父线程 `cpumask` | `man sched_setaffinity` |
| NUMA mempolicy | `clone()` 拷贝父线程 `mempolicy` | `man set_mempolicy`，`CLONE_VM` 不影响 mempolicy 继承 |
| NUMA cpuset | 同 affinity | — |

**关键结论**：只要在**父线程**（拉起通信线程的那个线程）里设置好 affinity + mempolicy，后续由它创建的所有 Python 线程和 C++ 线程都会继承。这是绕开"Python 拿不到 C++ 线程 tid"问题的根本途径。

### 3.2 为什么需要 C++ 侧实现

| 能力 | Python `os.sched_setaffinity` | C++ `numa_*` / `set_mempolicy` |
|---|---|---|
| 线程级 CPU 绑核 | ✅ | ✅ |
| 线程级 NUMA 内存绑定 | ❌ 标准库未暴露 | ✅ `set_mempolicy(2)` 是线程级 |
| 已分配内存迁移 | ❌ | ✅ `numa_migrate_pages` |
| 严格模式 | ❌ | ✅ `numa_set_strict` |

`set_mempolicy(2)` 是**线程级**的（不是进程级，常见误解）——只影响调用线程的后续内存分配，新线程通过 `clone()` 继承。因此 C++ 侧实现能完整覆盖 Python 漏掉的 NUMA 内存绑定。

### 3.3 为什么不复用 `init_cpu_threads_env`

[numa_utils.cpp: init_cpu_threads_env](sgl-kernel/csrc/cpu/numa_utils.cpp) 已有 C++ 实现，但它面向 OMP 池，会带来不需要的副作用：

| 副作用 | 对 mooncake 的影响 |
|---|---|
| `omp_set_num_threads(N)` 触发 OMP 池创建 | mooncake 不用 OMP，无意义且占资源 |
| `numa_migrate_pages` 迁移已有内存 | 阻塞数百 ms，启动慢 |
| `#pragma omp parallel` 启动多个线程 | mooncake 线程已有 Python 侧拉起 |

**结论**：新建一个**精简的纯线程级**函数，不碰 OMP、不迁移内存。

### 3.4 ZMQ 的原生 affinity 支持与混合绑定策略

> **关键发现**：ZMQ（libzmq >= 4.3, pyzmq >= 17.1）原生支持 I/O 线程的 CPU 亲和性设置，但**不支持 NUMA 内存绑定**。这决定了 ZMQ 部分必须采用混合策略。

#### 3.4.1 ZMQ 原生 CPU affinity API

```python
ctx = zmq.Context(io_threads=2)
# 必须在创建 socket 之前设置（可在 Context 创建之后）
ctx.set(zmq.THREAD_AFFINITY_CPU_ADD, 4)  # 将 CPU 4 加入 I/O 线程亲和集
ctx.set(zmq.THREAD_AFFINITY_CPU_ADD, 5)  # 将 CPU 5 加入
ctx.set(zmq.THREAD_NAME_PREFIX, b"zmq-io")  # 线程命名，便于调试
sock = ctx.socket(zmq.PUSH)  # 此后创建 socket，I/O 线程启动
```

libzmq 在 I/O 线程启动时调用 `pthread_setaffinity_np` 应用指定的 CPU 集合。若未设置 `THREAD_AFFINITY_CPU_ADD`，I/O 线程继承创建者的 affinity。

#### 3.4.2 ZMQ 不支持 NUMA

libzmq 没有暴露任何 `set_mempolicy` / `numa_set_membind` 接口。ZMQ I/O 线程的 NUMA 内存策略**只能**通过 Linux 线程继承机制获得——即创建 Context 的父线程先设好 mempolicy，I/O 线程通过 `clone()` 继承。

#### 3.4.3 混合绑定策略

ZMQ 部分采用**双管齐下**的混合策略：

| 目标 | 机制 | 时机 |
|---|---|---|
| NUMA 内存绑定 | `bind_current_thread()` 设置父线程 mempolicy → I/O 线程继承 | `zmq.Context()` **之前** |
| CPU 绑核 | `ctx.set(zmq.THREAD_AFFINITY_CPU_ADD, N)` 原生 API | `zmq.Context()` **之后**、`ctx.socket()` **之前** |

两者**不冲突**：`THREAD_AFFINITY_CPU_ADD` 只覆盖 CPU affinity mask，不影响已继承的 mempolicy。即使原生 API 覆盖了 CPU 继承值，NUMA mempolicy 仍保持继承。

#### 3.4.4 SGLang 中多个 zmq.Context 的处理

SGLang 在不同进程内创建**多个** `zmq.Context` 实例：

| 进程 | Context 创建位置 | io_threads |
|---|---|---|
| Scheduler | [ipc_channels.py:34](python/sglang/srt/managers/scheduler_components/ipc_channels.py#L34) `zmq.Context(2)` | 2 |
| DP Controller | [data_parallel_controller.py:148](python/sglang/srt/managers/data_parallel_controller.py#L148) `zmq.Context(1 + dp_size)` | 1+dp_size |
| Detokenizer | [detokenizer_manager.py:112](python/sglang/srt/managers/detokenizer_manager.py#L112) `zmq.Context(2)` | 2 |
| Tokenizer / MultiTokenizer | [multi_tokenizer_mixin.py:81,545](python/sglang/srt/managers/multi_tokenizer_mixin.py#L81) | 1 或 2 |
| LoadSnapshot | [load_snapshot.py:384,524](python/sglang/srt/managers/load_snapshot.py#L384) `zmq.Context.instance()` | 默认 1（进程级单例） |

本方案主要关注 **Scheduler 进程内**的 Context（因为 mooncake 线程也在 Scheduler 进程内）。但若需统一绑核，每个进程的 Context 都应处理。

**`zmq.Context.instance()` 单例的特殊性**：它是进程级单例，首次调用时创建，后续调用返回同一实例。若单例在绑定前已被创建，继承方案失效；原生 `THREAD_AFFINITY_CPU_ADD` 需在首次 `socket()` 前设置才有效。需确保绑定逻辑在 `instance()` 首次调用前或紧随其后执行。

#### 3.4.5 Mooncake 与 ZMQ 策略对比

| 组件 | CPU 绑核机制 | NUMA 绑定机制 | 推荐 |
|---|---|---|---|
| Mooncake Python 线程 | `bind_current_thread` 继承 | `bind_current_thread` 继承 | 继承方案（mooncake 无原生 API） |
| Mooncake C++ 线程（RDMA polling） | `bind_current_thread` 继承 | `bind_current_thread` 继承 | 继承方案（拿不到 tid） |
| ZMQ I/O 线程 | `THREAD_AFFINITY_CPU_ADD` 原生 **或** 继承 | **仅** `bind_current_thread` 继承 | **混合**：原生 CPU + 继承 NUMA |

---

### 3.5 绑定范围与时序约束

```
┌─────────────────────────────────────────────────────────────┐
│  Scheduler 进程启动                                          │
│                                                              │
│  1. 早期阶段：                                                │
│     [★ 绑定封装层调用 bind_current_thread() ★]              │
│        ↓ 设置主线程 CPU affinity + NUMA mempolicy            │
│                                                              │
│  2. 构造阶段（绑定必须在此之前完成）：                         │
│     - MooncakeTransferEngine()  → C++ RDMA polling 线程      │
│     - MooncakeDistributedStore() → C++ store 工作线程        │
│     - zmq.Context()             → libzmq I/O 线程           │
│     (以上 C++ 线程通过 clone() 继承主线程的 affinity/mempolicy)│
│                                                              │
│  3. 运行阶段：                                                │
│     - MooncakeKVManager 拉起 transfer_worker / bootstrap /  │
│       decode / heartbeat / probe 等 Python 线程              │
│     (Python 线程通过 clone() 继承)                            │
│                                                              │
│  4. 推理阶段：                                                │
│     - SGLANG_SET_CPU_AFFINITY 重新设置推理线程 affinity       │
│       （不会影响已启动的通信线程，因为 sched_setaffinity       │
│        只作用于调用线程本身，不重置已存在线程）                │
└─────────────────────────────────────────────────────────────┘
```

**关键约束**：绑定必须在第 2 步的 C++ 对象构造**之前**完成，否则这些 C++ 库拉起的线程会落在默认 affinity 上无法被覆盖。

---

## 4. 需实现的内容

### 4.1 C++ 侧：新增 `bind_current_thread` 函数

**文件**：`sgl-kernel/csrc/cpu/numa_utils.cpp`（追加）+ `sgl-kernel/csrc/cpu/numa_utils_pybind/binding.cpp`（追加绑定）

**函数签名**：

```cpp
// 绑定当前线程（不创建任何子线程，不迁移内存，不修改 OMP）
// cpu_ids: numactl 风格，如 "0,1,2" 或 "0-3"
// numa_node: -1 表示不绑内存；>=0 表示绑到指定 NUMA 节点
// strict: true=内存分配必须落在指定节点否则失败；false=尽力而为
// 返回值: 描述串，便于调试
std::string bind_current_thread(
    const std::string& cpu_ids,
    int numa_node,
    bool strict
);
```

**内部步骤**：

1. 解析 `cpu_ids` 为 `cpu_set_t`
2. `sched_setaffinity(0, sizeof(cpu_set_t), &mask)` — 绑当前线程 CPU（0=当前线程）
3. 若 `numa_node >= 0`：
   - `numa_set_bind_policy(strict ? 1 : 0)`
   - `numa_set_membind(&node_mask)` 或 `set_mempolicy(MPOL_BIND, &node_mask, ...)`
4. **不**调 `numa_migrate_pages`（避免启动延迟）
5. **不**调 `omp_set_num_threads`
6. 返回描述串：当前 tid + 绑定的 CPU 集合 + NUMA 节点

**头文件依赖**：`<sched.h>`、`<numa.h>`、`<unistd.h>`（`syscall(SYS_gettid)`）

### 4.2 Python 侧：新增绑定封装模块

**文件**：`python/sglang/srt/disaggregation/common/thread_binding.py`（新建）

**职责**：
- 读取环境变量 `SGLANG_MOONCAKE_BIND_CPU` / `SGLANG_MOONCAKE_BIND_NUMA_NODE` / `SGLANG_MOONCAKE_BIND_STRICT`
- 提供 `bind_current_thread()` Python 接口，内部调 C++ 实现
- 提供 `MooncakeCommLauncher` 上下文管理器，封装"绑定 + 拉起通信线程"的时序

**接口**：

```python
def bind_current_thread(
    cpu_ids: Optional[str] = None,
    numa_node: Optional[int] = None,
    strict: bool = True,
) -> str:
    """绑定当前线程的 CPU 和 NUMA 内存策略。

    参数若为 None，则从环境变量读取：
      - SGLANG_MOONCAKE_BIND_CPU
      - SGLANG_MOONCAKE_BIND_NUMA_NODE
      - SGLANG_MOONCAKE_BIND_STRICT
    若环境变量也未设置，则不执行任何绑定，返回空串。
    """

class MooncakeCommLauncher:
    """通信线程绑定启动器。

    用法：
        with MooncakeCommLauncher.from_env() as launcher:
            # 此后构造的 MooncakeTransferEngine / zmq.Context / MooncakeKVManager
            # 创建的所有线程都会继承当前线程的 affinity + mempolicy
            engine = MooncakeTransferEngine(...)
            ...
    """
```

### 4.3 环境变量：新增 3 个

在 `python/sglang/srt/environ.py` 的 `Environ` 类中追加：

```python
SGLANG_MOONCAKE_BIND_CPU = EnvStr(None)        # 如 "0,1,2" 或 "0-3"
SGLANG_MOONCAKE_BIND_NUMA_NODE = EnvInt(-1)    # -1=不绑内存
SGLANG_MOONCAKE_BIND_STRICT = EnvBool(True)    # 严格模式
```

### 4.4 集成点：4 处插桩

| 序号 | 文件 | 位置 | 改动 |
|---|---|---|---|
| 1 | [mooncake_transfer_engine.py:107](python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py#L107) | `MooncakeTransferEngine.__init__` 开头 | 调 `bind_current_thread()`，**在 `TransferEngine()` 构造前** |
| 2 | [mooncake_store.py:361](python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py#L361) | `MooncakeStore.__init__` 中 `MooncakeDistributedStore()` 构造前 | 同上 |
| 3 | [conn.py:156](python/sglang/srt/disaggregation/mooncake/conn.py#L156) | `MooncakeKVManager.__init__` 开头 | 同上，确保后续 Python 线程继承绑定 |
| 4 | [ipc_channels.py:34](python/sglang/srt/managers/scheduler_components/ipc_channels.py#L34) 等 | `zmq.Context()` 创建处 | **混合策略**：见 4.4.1 |

可选加固（防御性）：在每个通信线程的 target 函数开头再调一次 `bind_current_thread()`，防止 mempolicy 被中间代码覆盖。涉及 5 个 target：
- `transfer_worker` ([conn.py:1263](python/sglang/srt/disaggregation/mooncake/conn.py#L1263))
- `bootstrap_thread` ([conn.py:1504](python/sglang/srt/disaggregation/mooncake/conn.py#L1504))
- `decode_thread` ([conn.py:1603](python/sglang/srt/disaggregation/mooncake/conn.py#L1603))
- `heartbeat_checker` ([common/conn.py:885](python/sglang/srt/disaggregation/common/conn.py#L885))
- `_failed_session_probe_loop` ([conn.py:1761](python/sglang/srt/disaggregation/mooncake/conn.py#L1761))

#### 4.4.1 ZMQ 集成：混合绑定实现

ZMQ 部分采用 3.4 节的混合策略：NUMA 靠继承、CPU 靠原生 API。需封装一个 helper：

```python
# python/sglang/srt/disaggregation/common/thread_binding.py

def configure_zmq_context(ctx: "zmq.Context", cpu_ids: Optional[str] = None) -> None:
    """对 zmq.Context 应用原生 CPU affinity。

    必须在 ctx.socket() 之前调用。NUMA 绑定不在此处理——
    需在创建 ctx 之前调 bind_current_thread() 让 I/O 线程继承 mempolicy。
    """
    if not cpu_ids:
        cpu_ids = envs.SGLANG_MOONCAKE_BIND_CPU.get()
    if not cpu_ids:
        return
    for cpu in _parse_cpu_ids(cpu_ids):       # "0,1,2" -> [0,1,2]
        ctx.set(zmq.THREAD_AFFINITY_CPU_ADD, cpu)
    ctx.set(zmq.THREAD_NAME_PREFIX, b"zmq-io")
```

Scheduler 进程内 [ipc_channels.py:34](python/sglang/srt/managers/scheduler_components/ipc_channels.py#L34) 的改法：

```python
# 改动前
context = zmq.Context(2)

# 改动后
bind_current_thread()                          # NUMA 继承（必须在 Context 前）
context = zmq.Context(2)
configure_zmq_context(context)                 # CPU 原生绑定（必须在 socket 前）
```

**对 `zmq.Context.instance()` 单例**（[load_snapshot.py:384,524](python/sglang/srt/managers/load_snapshot.py#L384)）：需在首次 `instance()` 调用前调 `bind_current_thread()`，并在获取单例后立即调 `configure_zmq_context(ctx)`，确保在首次 `socket()` 前完成。若单例已被其他模块提前触发，NUMA 继承失效——此时只能依赖原生 CPU 绑定，NUMA 需通过 `numactl` 包装进程来补救。

#### 4.4.2 其他进程的 ZMQ Context

DP Controller / Detokenizer / Tokenizer 进程的 ZMQ Context 如需绑核，同样套用 4.4.1 模式。但本方案优先级是 Scheduler 进程（mooncake 线程所在），其他进程可后续迭代。

### 4.5 ThreadPoolExecutor 绑核（易遗漏）

[conn.py:192-197](python/sglang/srt/disaggregation/mooncake/conn.py#L192) 的 `ThreadPoolExecutor` 工作线程虽然会继承父线程绑定，但若想在 executor 启动后再显式加固，可用 `initializer`：

```python
self.executors = [
    concurrent.futures.ThreadPoolExecutor(
        max_workers=transfer_thread_pool_size // transfer_queue_size,
        thread_name_prefix="mooncake-exec",
        initializer=bind_current_thread,   # ← 加固
        initargs=(None, None, True),       # 从 env 读取
    )
    for _ in range(transfer_queue_size)
]
```

注意 `initargs` 必须是 tuple。

---

## 5. 注意事项

### 5.1 时序约束（最重要）

绑定必须在以下 C++ 对象构造**之前**完成：

```
bind_current_thread()
    ↓
MooncakeTransferEngine()      # C++ 启动 RDMA polling 线程
MooncakeDistributedStore()    # C++ 启动 store 工作线程
```

**ZMQ 的特殊时序**（见 3.4 节混合策略）：

```
bind_current_thread()         # NUMA mempolicy 继承（必须在 Context 前）
    ↓
zmq.Context()                 # I/O 线程继承 mempolicy
    ↓
configure_zmq_context(ctx)    # 原生 CPU affinity（必须在 socket 前）
    ↓
ctx.socket()                  # I/O 线程启动，应用 CPU affinity
```

**`zmq.Context.instance()` 单例风险**：若单例在 `bind_current_thread()` 之前被其他模块触发（如日志/监控），NUMA 继承失效，只能通过原生 CPU 绑定 + `numactl` 包装进程补救。需排查 SGLang 启动早期是否有隐式 `zmq.Context.instance()` 调用。

```
MooncakeKVManager()           # Python 拉起 transfer_worker 等
```

**若顺序颠倒，C++ 库拉起的线程会落在默认 affinity 上，Python 无法回收。**

### 5.2 不要真绑 1 个核

单核容纳 4-12 个 transfer_worker + executor + heartbeat + bootstrap/decode + RDMA polling + ZMQ I/O，会严重时片抢占，GIL 竞争加剧，可能比不绑还慢。

**建议**：

| 部署角色 | 建议核数 | 示例 |
|---|---|---|
| PREFILL 节点（线程多） | 2-4 核 | `SGLANG_MOONCAKE_BIND_CPU=32,33,34,35` |
| DECODE 节点（线程少） | 1-2 核 | `SGLANG_MOONCAKE_BIND_CPU=48,49` |

### 5.3 与推理绑核集合不能重叠

`SGLANG_SET_CPU_AFFINITY=1` 会按 GPU 线性分配物理核给推理。若 mooncake 绑定的核落在推理范围内，等于没绑。部署前需核算：

```
推理核范围 = set_gpu_proc_affinity 算出的集合
mooncake 核范围 = SGLANG_MOONCAKE_BIND_CPU
要求：两者交集为空
```

### 5.4 NUMA 跨节点访问代价

若 `SGLANG_NUMA_BIND_V2=1` 把整个 scheduler 进程绑到 NUMA 节点 0，而 mooncake 线程绑到节点 1 的 CPU + 节点 1 的内存：

| 访问类型 | 代价 |
|---|---|
| mooncake 线程访问 Python 业务对象（节点 0） | 跨 NUMA，慢 |
| mooncake 线程访问自己的传输 buffer（节点 1） | 本地，快 |

通常可接受，因为传输 buffer 是 hot path。但需性能验证。

### 5.5 已分配内存不会迁移

`set_mempolicy` 只对**调用之后的内存分配**生效。Python 解释器在调用绑定前已分配的 GC 堆、模块、CUDA context 等仍留在原 NUMA 节点。

→ 对 mooncake 传输 buffer 影响小（绑定后新分配），对 Python 业务对象有影响。

### 5.6 GIL 与单核绑定的冲突

Python 线程绑到同一核后，GIL 抢占会更剧烈。若发现绑核后吞吐下降，先尝试放宽到 2-3 个核。

### 5.7 `setup_dummy` 模式的特殊性

[mooncake_store.py:436](python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py#L436) `standalone_storage=True` 时的 `setup_dummy` 文档暗示会起独立 `mooncake_client` 进程：

| 启动方式 | 是否继承绑定 |
|---|---|
| `fork()` | ✅ 继承（子进程拷贝父进程 affinity/mempolicy） |
| `fork()+exec()` | ❌ 不继承（exec 重置地址空间但保留 affinity；mempolicy 行为依赖实现） |
| `posix_spawn` | ❌ 不继承 |

**待验证**：需用 `strace -f -e clone,fork,execve` 在实际环境观察 `setup_dummy` 的进程创建方式。

### 5.8 外部进程无法通过本方案绑定

| 进程 | 创建者 | 本方案能否覆盖 |
|---|---|---|
| `mooncake_master` | 用户手动启动 | ❌ 需 `numactl` 包装启动命令 |
| `mooncake_store_service` | 用户手动启动 | ❌ 同上 |
| 独立 `mooncake_client` | 用户手动启动 | ❌ 同上 |

外部进程的绑核方法见第 9 节。

### 5.9 异常处理

绑定失败时（如 CPU 编号超范围、NUMA 节点不存在）应：
- 记录 warning 日志
- **不抛异常**，继续启动（避免绑定失败导致服务无法启动）
- 返回错误描述串供排查

### 5.10 可观测性

建议绑定成功后日志输出：
```
[mooncake-bind] tid=12345 cpu=32,33,34,35 numa_node=1 strict=true
```

并在 `/proc/<pid>/task/*/status` 可验证（见第 8 节）。

---

## 6. 代码改动清单

| 序号 | 文件 | 类型 | 说明 |
|---|---|---|---|
| 1 | `sgl-kernel/csrc/cpu/numa_utils.cpp` | 修改 | 追加 `bind_current_thread` 函数实现 |
| 2 | `sgl-kernel/csrc/cpu/numa_utils_pybind/binding.cpp` | 修改 | 追加 pybind11 绑定 |
| 3 | `sgl-kernel/csrc/cpu/numa_utils_pybind/CMakeLists.txt` | 无需改 | `numa_utils.cpp` 已在 SOURCES 列表 |
| 4 | `python/sglang/srt/disaggregation/common/thread_binding.py` | 新建 | Python 封装层 |
| 5 | `python/sglang/srt/environ.py` | 修改 | 新增 3 个环境变量 |
| 6 | `python/sglang/srt/distributed/device_communicators/mooncake_transfer_engine.py` | 修改 | `__init__` 开头插桩 |
| 7 | `python/sglang/srt/mem_cache/storage/mooncake_store/mooncake_store.py` | 修改 | `MooncakeStore.__init__` 插桩 |
| 8 | `python/sglang/srt/disaggregation/mooncake/conn.py` | 修改 | `MooncakeKVManager.__init__` 插桩 + 5 个 target 加固 + ThreadPoolExecutor initializer |
| 9 | `python/sglang/srt/utils/network.py` | 修改 | `zmq.Context()` 创建前插桩 |
| 10 | `sgl-kernel/tests/test_mooncake_thread_binding.py` | 新建 | 单元测试（可选） |

---

## 7. 实施步骤

### 步骤 1：C++ 侧实现 `bind_current_thread`

1. 在 [numa_utils.cpp](sgl-kernel/csrc/cpu/numa_utils.cpp) 末尾追加 `bind_current_thread` 实现
2. 在 [binding.cpp](sgl-kernel/csrc/cpu/numa_utils_pybind/binding.cpp) 追加 pybind11 绑定
3. 重新构建 `.so`：
   ```bash
   bash sgl-kernel/csrc/cpu/numa_utils_pybind/build.sh
   ```
4. 验证：
   ```python
   import _numa_utils
   print(_numa_utils.bind_current_thread("0-3", 0, True))
   ```

### 步骤 2：Python 封装层

1. 新建 `python/sglang/srt/disaggregation/common/thread_binding.py`
2. 实现 `bind_current_thread()` + `MooncakeCommLauncher`
3. 在 `environ.py` 加 3 个环境变量
4. 单测：验证调用后 `os.sched_getaffinity(0)` 变化

### 步骤 3：核心插桩（4 处）

按第 4.4 节在 4 个集成点插入 `bind_current_thread()` 调用，**确保在 C++ 对象构造前**。

### 步骤 4：防御性加固（5 处 target + executor）

在 5 个 thread target 开头加 `bind_current_thread()`，给 `ThreadPoolExecutor` 加 `initializer`。

### 步骤 5：集成测试

1. 用 PD 分离小流量场景跑通
2. 用 `/proc/<pid>/task/*/status` 验证所有 mooncake 线程 affinity 落在指定核
3. 对比绑核前后的推理吞吐 + RDMA 传输延迟

### 步骤 6：文档与部署指南

更新部署文档，说明 `SGLANG_MOONCAKE_BIND_CPU` 等环境变量的用法，以及外部进程的 `numactl` 包装方式（第 9 节）。

---

## 8. 验证方法

### 8.1 绑核生效验证

```bash
# 找到 scheduler 进程 PID
PID=$(pgrep -f "sglang.srt.managers.scheduler")

# 查看进程内每个线程的 affinity + 名字
for tid in $(ls /proc/$PID/task); do
    name=$(cat /proc/$PID/task/$tid/comm)
    affinity=$(taskset -cp $tid 2>/dev/null | awk -F: '{print $2}')
    echo "tid=$tid name=$name affinity=$affinity"
done | grep -E "mooncake|Thread|ThreadPool|bootstrap|decode|heartbeat"

# 期望：所有 mooncake 相关线程的 affinity 都落在 SGLANG_MOONCAKE_BIND_CPU 指定的核上
```

### 8.2 NUMA 内存策略验证

```bash
# 查看进程的 NUMA 内存分布（要求 numactl 包）
numastat -p $PID

# 期望：mooncake 线程新分配的内存主要落在 SGLANG_MOONCAKE_BIND_NUMA_NODE 指定的节点
```

### 8.3 性能对比

| 指标 | 测试方法 | 期望 |
|---|---|---|
| 推理吞吐 | `sglang.benchmark.serving` 压测 | 不下降或略升 |
| RDMA 传输延迟 | mooncake_trace 日志 | 不显著上升 |
| CPU 利用率 | `mpstat -P ALL 1` | mooncake 核繁忙，推理核不受干扰 |
| 跨 NUMA 访问 | `perf stat -e node-loads, node-load-misses` | mooncake 相关跨节点访问不增加 |

### 8.4 单元测试（可选）

新建 `sgl-kernel/tests/test_mooncake_thread_binding.py`：
- 测试 `bind_current_thread` 返回串格式
- 测试调用后 `os.sched_getaffinity(0)` 匹配
- 测试子线程是否继承绑定（起 `threading.Thread` 检查 affinity）
- 测试无效输入的容错

---

## 9. 外部进程绑核（补充）

本方案无法覆盖的外部进程，需通过启动命令 `numactl` 包装：

### 9.1 mooncake_master

```bash
numactl --cpunodebind=0 --membind=0 \
    mooncake_master --enable_http_metadata_server=true --http_metadata_server_port=8080
```

### 9.2 mooncake_store_service

```bash
numactl --cpunodebind=0 --membind=0 \
    python -m mooncake.mooncake_store_service --config=/path/to/config.json --port=8081
```

### 9.3 独立 mooncake_client（standalone_storage=True 模式）

```bash
numactl --cpunodebind=0 --membind=0 \
    mooncake_client --config=/path/to/config.json
```

### 9.4 SGLang Scheduler 进程本身（若需要）

若想让整个 Scheduler 进程（含推理 + 通信）都在某个 NUMA 节点：

```bash
numactl --cpunodebind=0 --membind=0 \
    python -m sglang.launch_server ...
```

但通常推荐让推理用 `SGLANG_SET_CPU_AFFINITY`，通信用本方案的线程级绑定。

---

## 10. 风险与缓解

| 风险 | 影响 | 缓解 |
|---|---|---|
| C++ 库线程在绑定前构造 | RDMA polling / ZMQ I/O 线程逃逸 | 严格按第 5.1 节时序；用 `strace` 验证 |
| 单核 GIL 抢占 | 吞吐下降 | 至少绑 2-4 核；性能回归测试 |
| 与 `SGLANG_SET_CPU_AFFINITY` 重叠 | 绑核失效 | 部署前核算核范围 |
| NUMA 跨节点访问 | mooncake 访问 Python 对象变慢 | 性能验证；必要时调整 NUMA 节点选择 |
| `setup_dummy` 起独立进程 | 子进程不继承绑定 | 用 `strace` 确认；外部 `numactl` 包装 |
| 绑定失败导致启动失败 | 服务不可用 | 绑定失败只 warning 不 raise |
| 已分配内存不迁移 | 旧对象留在原节点 | 可接受；传输 buffer 是新分配的 |
| `SGLANG_NUMA_BIND_V2` 与本方案冲突 | mempolicy 被覆盖 | 二选一；或确保两者 NUMA 节点一致 |

---

## 11. 环境变量速查

| 变量 | 默认值 | 说明 |
|---|---|---|
| `SGLANG_MOONCAKE_BIND_CPU` | `None` | mooncake/zmq 线程绑定的 CPU 核，如 `"0-3"` 或 `"32,33,34,35"` |
| `SGLANG_MOONCAKE_BIND_NUMA_NODE` | `-1` | mooncake/zmq 线程绑定的 NUMA 节点，-1=不绑内存 |
| `SGLANG_MOONCAKE_BIND_STRICT` | `True` | 严格模式：内存分配必须落在指定 NUMA 节点 |
| `SGLANG_SET_CPU_AFFINITY` | `False` | 推理线程绑核（已有） |
| `SGLANG_NUMA_BIND_V2` | `True` | 子进程级 NUMA 绑定（已有） |
| `SGLANG_DISAGGREGATION_THREAD_POOL_SIZE` | `None` | mooncake 传输线程池大小（已有） |
| `SGLANG_DISAGGREGATION_QUEUE_SIZE` | `4` | mooncake 传输队列数（已有） |

---

## 12. 与现有机制的关系

```
┌──────────────────────────────────────────────────────────────┐
│  Scheduler 进程                                              │
│                                                              │
│  ┌────────────────────────────────────────────────────┐      │
│  │  推理线程（GPU 计算）                               │      │
│  │  绑核：SGLANG_SET_CPU_AFFINITY=1                   │      │
│  │  范围：CPU 0-31（示例）                            │      │
│  └────────────────────────────────────────────────────┘      │
│                                                              │
│  ┌────────────────────────────────────────────────────┐      │
│  │  通信线程（Mooncake + ZMQ）   ← 本方案             │      │
│  │  绑核：SGLANG_MOONCAKE_BIND_CPU=32,33,34,35        │      │
│  │  绑内存：SGLANG_MOONCAKE_BIND_NUMA_NODE=1          │      │
│  │  覆盖：transfer_worker / bootstrap / decode /      │      │
│  │        heartbeat / probe / RDMA polling / ZMQ I/O  │      │
│  └────────────────────────────────────────────────────┘      │
│                                                              │
│  ┌────────────────────────────────────────────────────┐      │
│  │  子进程（Tokenizer / Detokenizer / DP Controller） │      │
│  │  绑核：SGLANG_NUMA_BIND_V2=1                       │      │
│  └────────────────────────────────────────────────────┘      │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│  外部进程（用户手动启动）                                     │
│  mooncake_master / mooncake_store_service / mooncake_client   │
│  绑核：numactl --cpunodebind=0 --membind=0 ...（第 9 节）     │
└──────────────────────────────────────────────────────────────┘
```

---

## 13. 待确认事项

在实施前需通过实验确认以下两点：

### 13.1 C++ 库线程创建时机

需确认 `MooncakeDistributedStore()` 和 `TransferEngine()` 构造时是否立即 spawn 工作线程：

```bash
strace -f -e clone,fork -o trace.log python -c "
from mooncake.store import MooncakeDistributedStore
MooncakeDistributedStore()
"
grep clone trace.log
```

若构造时无 `clone`，说明线程是延迟创建，绑定时机可放宽；若有 `clone`，必须严格按第 5.1 节时序。

### 13.2 `setup_dummy` 进程创建方式

```bash
strace -f -e clone,fork,execve -o trace.log python -c "
# 模拟 standalone_storage=True 场景
...
"
grep -E "clone|fork|execve" trace.log
```

判断是 `fork()`（继承绑定）还是 `fork()+exec()`（不继承 mempolicy）。

---

## 14. 总结

本方案通过"C++ 侧实现线程级 CPU+NUMA 绑定 + Python 封装层在通信对象构造前调用 + 利用 Linux 线程继承机制覆盖 C++ 库线程"的方式，完整解决了 Mooncake/ZMQ 通信线程的绑核问题。

**核心收益**：
- 覆盖 Python 线程 + C++ 库线程（RDMA polling / ZMQ I/O）
- 同时绑定 CPU affinity + NUMA 内存策略
- 与现有 `SGLANG_SET_CPU_AFFINITY` / `SGLANG_NUMA_BIND_V2` 互补不冲突

**核心约束**：
- 绑定必须在 C++ 对象构造前完成
- 不能只绑 1 个核（GIL 竞争）
- 外部进程需单独 `numactl` 包装