# 7月22日-7月23日进展

做了mooncake和zmq组件、以及绑numa和绑核的调研。

## mooncake

[mooncake deepwiki](https://deepwiki.com/search/mooncake_5a959a3a-c73b-40bb-9920-ae1069535ab6?mode=fast)

根据deepwiki的回答，mooncake自身具有以cpu socket为级别的绑核绑内存api：

```c++
static inline int bindToSocket(int socket_id) {  
    // 检查 NUMA 是否可用  
    if (unlikely(numa_available() < 0)) {  
        LOG(WARNING) << "The platform does not support NUMA";  
        return ERR_NUMA;  
    }  
    // 获取指定 socket 的 CPU 列表并设置亲和性  
    cpu_set_t cpu_set;  
    CPU_ZERO(&cpu_set);  
    // ... 获取 CPU 列表并设置 pthread_setaffinity_np  
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpu_set);  
}
```

不过这个绑定范围是以cpu socket为单位，并不能支持我们现在所需要的场景，也就是cpu绑核只需要在一个numa分配一个核。

绑numa也许可以借助上面的方法。

### mooncake on sglang

sglang支持两种部署形式的mooncake，一种是内嵌模式，这个应该就是我们目前所需的模式，方便以cpu socket为粒度进行管控。

另一种是外置模式，用户手动拉起mooncake服务，通过sglang这边的环境变量进行配置，与外置mooncake服务建立联系。

如果我们的部署形式支持外置模式，也许能省一些工作（`MooncakeStore`相关），例如：

```shell
exec numactl --cpunodebind=0 --membind=0 python3 -m mooncake.mooncake_store_service --port=8099
```

另外，sglang中存在`MooncakeTransferEngine`、`MooncakeKVManager`等管理设施线程，在外置模式下也会进行拉起（PD分离部署情况下），还是需要在框架侧做绑核绑numa的适配。

sglang侧的mooncake设施中，也使用zmq进行管理信息的传递，后续实现绑核绑numa操作时，需要区分开来，看是否需要和其他的zmq进行隔离绑定在不同的核上。

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

通信线程全景清单

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


## zmq

根据网上查阅的资料，zmq先行主流版本支持以cpu core为单位进行绑核。不过没有api支持numa的绑定。

```python
import zmq

# 创建 context 时设置 io_threads
ctx = zmq.Context(io_threads=2)

# 设置线程亲和性（必须在创建 socket 之前）
# pyzmq 暴露了这些常量（需要 pyzmq >= 17.1, libzmq >= 4.3）
ctx.set(zmq.THREAD_AFFINITY_CPU_ADD, 4)
ctx.set(zmq.THREAD_AFFINITY_CPU_ADD, 5)
ctx.set(zmq.THREAD_AFFINITY_CPU_ADD, 6)
ctx.set(zmq.THREAD_AFFINITY_CPU_ADD, 7)

# 线程命名
ctx.set(zmq.THREAD_NAME_PREFIX, 55555)

# 之后创建 socket
sock = ctx.socket(zmq.PUSH)
sock.connect("tcp://192.168.1.100:5555")
```

可以直接使用zmq自带的接口实现绑核操作。

numa绑定操作需要额外实现。

## 绑核

python原生支持了以cpu core为单位的绑核操作。

```python
import os

# 绑定到 CPU 核心 4, 5, 6, 7
os.sched_setaffinity(0, {4, 5, 6, 7})
```

## 绑定numa

绑定numa没有主流的python包支持，有以下几种形式：

- 通过启动脚本进行控制：

```bash
numactl --cpunodebind=0 --membind=0 \
    python zmq_server.py
```

- 通过c++侧进行控制：

sglang自己是实现了一套绑numa方法的，比如`numa_utils.cpp`中的`init_cpu_threads_env`函数。

```c++
#include <numa.h>

...

  // Memory node binding
  if (numa_available() != -1) {
    TORCH_CHECK(!omp_cpu_ids.empty(), "Cannot bind memory, no CPUs specified.");
    int mem_node_id_st = numa_node_of_cpu(omp_cpu_ids.front());
    int mem_node_id_ed = numa_node_of_cpu(omp_cpu_ids.back());
    if (mem_node_id_st > mem_node_id_ed) {
      std::swap(mem_node_id_st, mem_node_id_ed);
    }

    bitmask* mask =
        numa_parse_nodestring((std::to_string(mem_node_id_st) + "-" + std::to_string(mem_node_id_ed)).c_str());
    bitmask* src_mask = numa_get_membind();

    int pid = getpid();

    // move all existing pages to the specified numa node.
    *(src_mask->maskp) = *(src_mask->maskp) ^ *(mask->maskp);
    int page_num = numa_migrate_pages(pid, src_mask, mask);
    if (page_num == -1) {
      TORCH_WARN(false, "numa_migrate_pages failed. errno: " + std::to_string(errno));
    }

    // restrict memory allocation node.
    numa_set_membind(mask);
    numa_set_strict(1);
  }

...
```

这个实现会在python中进行调用，完成对当前进程的numa绑定。

```python
def _init_cpu_threads_env(
    *, tp_size: int, tp_rank: int, local_omp_cpuid: Optional[List[int]]
) -> None:
    if _is_cpu_amx_available or _is_cpu_arm64:
        # Bind OpenMP threads to CPU cores
        torch.ops.sgl_kernel.init_cpu_threads_env(local_omp_cpuid)

        # Set local size to hint SGLang to use shared memory based AllReduce
        os.environ["LOCAL_SIZE"] = str(tp_size)
        torch.ops.sgl_kernel.initialize(tp_size, tp_rank)

    else:
        logger.warning(
            "init_cpu_threads_env and shared memory based AllReduce is disabled, only intel amx backend and arm64 are supported"
        )
```

如果需要做C++侧的绑核实现，可以进行参考。 

- python侧通过ctypes调用c++侧的绑核函数

[python-numa github](https://github.com/zakalibit/python-numa/blob/master/numa.py)

[py-numa github](https://github.com/smira/py-numa/blob/master/numa.py)

虽然没有主流的numa python包，但是社区是有一些相关实现的，可以选择性进行参考。

## 其他工具

实现了一个查看指定pid及其子线程的绑核绑numa的python脚本：

[print_pid_threads_cpubinding_numabinding.py](https://github.com/litmei/personal/blob/main/scripts/python/print_pid_threads_cpubinding_numabinding.py)

# 7月24日进展

## 规划适配路径

- 在已有环境上跑起来，检查计算面的绑核是否OK；
- 先实现绑核再实现绑numa，先做zmq的工作再做mooncake，先易后难，熟悉流程。