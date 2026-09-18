# NPU Prefill 入图方案评估

> 目标：评估 NPU 适配 Prefill CUDA Graph 的三条技术路线，选定方案并给出落地路径、验证用例与代码量估算。
>
> 结论：**选择 Breakable CUDA Graph（BCG）**。前提是先完成 §4 的 NPUGraph 分段捕获能力验证。

---

## 0. 摘要

| 方案 | NMU 适配成本 | 支持 PD 分离 / DP attn | 需要重写 `forward_extend` | 结论 |
|---|---|---|---|---|
| `tc_piecewise` | 低（200-600 行） | **否**（被 dynamo 限制挡掉） | 否 | ❌ 排除（刚需特性缺失） |
| **`breakable` (BCG)** | **低-中（200-600 行）** | **是** | **否** | ✅ **选定** |
| `full` | 中（600-1100 行） | 是 | **是** | 备选（BCG 不可行时退到此） |

核心判断依据：

1. NPU 的 decode graph 已经验证了"除 prefill attention 之外的一切"都能入图（MoE、norm、TP/DP 通信、metadata 静态化）。
2. BCG = 上述已验证部分 + 分段捕获能力 + prefill attention 走 eager。**这是成本最低的组合**。
3. BCG 让 attention 留在图外，因此**完全不需要重写 `forward_extend`** —— 而这是 Full 唯一的大成本项。

---

## 1. 三种 Prefill 入图方案

### 1.1 总览对比

| 维度 | `tc_piecewise` | `breakable` (BCG) | `full` |
|---|---|---|---|
| 捕获机制 | torch.compile 把模型切成若干 FX 子图，每片用 `torch.cuda.graph` 捕获 | 把 forward 按 break 标记切成 N 段，**每段一个独立 graph 对象**，共享同一 pool | 整段 transformer body 用**单个** graph 捕获 |
| 是否需要 torch.compile / dynamo | **需要**（核心依赖） | 不需要 | 不需要 |
| 注意力是否入图 | **否**（注册为 FX split op，图外 eager） | **否**（break 点，图外 eager） | **是**（必须图安全） |
| 捕获单元 | 每个 FX 子图 | 每个 break 区间 | 每个 `(num_tokens, chunked_prefix_variant)` bucket |
| 请求几何形状 | 任意（子图边界天然适配） | 任意（无 request-slot padding） | **固定 request 槽** (`full_prefill_max_req`)，尾部写哨兵 |
| metadata 处理 | 图外 plan | 图外 plan（默认回落普通 `init_forward_metadata`） | 图外 plan + 固定列宽 |
| LM head | 不一定 | eager tail | eager tail |
| 输出缓冲 | 各片自有 | 共享 output buffer + `copy_` | `reuse_output_buffer=True` |
| 图数量 | 子图数 × bucket | 段数 × bucket | bucket 数 × chunked-prefix variants (1/2/4/8/16) |
| 显存代价 | 中 | 中 | **最高** |
| 命中率 | 高 | **高**（无 slot padding） | 中（受 `req_slots` 上限 + 2x padding 浪费门限） |
| 成熟度（CUDA） | 新 | **默认后端**，最成熟 | opt-in / 实验性，规则表为空 |

### 1.2 `tc_piecewise`

**机制**

- 注意力被注册成 FX split op：`python/sglang/srt/layers/radix_attention.py:431`、`:176-289`
- NPU 版 FX 后端已存在并被工厂选中：`python/sglang/srt/compilation/backend.py:55-62` → `python/sglang/srt/compilation/npu_piecewise_backend.py:62-90`（`torch.npu.NPUGraph` + `torch.npu.graph`）
- 编译器可选 `eager`（`python/sglang/srt/compilation/compiler_interface.py:485-497`），因此**不需要 inductor/Triton codegen**

**优点**

1. 注意力不进图 → `forward_extend` 的 host 同步 / 动态 shape 问题**天然被绕开**。
2. NPU 侧 FX 后端骨架已存在，接线成本最低。
3. 图外 eager 注意力意味着动态 metadata 无需静态化。

**缺点**

1. **大量常用特性被 dynamo 限制挡掉**（见 §1.5），包括 PD 分离与 DP attention —— 这两项是调优刚需。
2. 依赖 torch_npu 自定义算子的 FakeTensor/meta 覆盖度，且要求 `fullgraph=True` 零 graph break —— 这是最大不确定性。
3. NPU 版 piecewise 后端是**未被验证的死代码**：相比 CUDA 版缺 warmup 短路、capture stream 传递、`graph_pool_capture_scope/replay_scope`、stream 为空时的降级（对照 `python/sglang/srt/compilation/cuda_piecewise_backend.py:149-228`）。

**特点**

- 唯一"注意力不入图但**也不需要 break 标记**"的方案。
- 收益上限与 BCG 接近，但**特性覆盖最差**。

**难点**

| 难点 | 风险 |
|---|---|
| torch.compile 在 NPU 上 trace 整个模型（绕不开 dynamo） | **高** |
| NPU piecewise backend 需与 CUDA 版对齐 | 低-中 |
| 各类 `torch.cuda.Stream/Event` 类型注解、`pool.py` 的 `is_cuda()` 门禁补漏 | 低 |

### 1.3 `breakable` (BCG)

**机制**

forward 被 `eager_on_graph(True)` 标记切成 N 段；每段是一个**独立的 graph 对象**，段间执行真正的 eager 代码。所有段共享同一个 memory pool。

关键代码：

- 分段捕获容器：`python/sglang/srt/model_executor/runner_backend_utils/breakable_cuda_graph/breakable_cuda_graph.py`
- backend：`python/sglang/srt/model_executor/runner_backend/breakable_cuda_graph_backend.py`
- break 标记（**设备无关**）：`python/sglang/srt/layers/radix_attention.py:585-590`（MHA）、`python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py:1004`（MLA）、`python/sglang/srt/layers/radix_linear_attention.py:242`（linear attn）、`python/sglang/srt/layers/moe/ep_moe/layer.py:215`（MoE A2A）

**优点**

1. **特性覆盖最全**：PD 分离、DP attn、MoE A2A、LoRA 都在白名单内。
2. **不需要重写 `forward_extend`**：attention 在图外，现有的 `.cpu().tolist()`、逐请求 python 循环全部保留。
3. **框架 90% 设备无关**：已有 `is_hip()/_is_xpu` 分支，`cuda.bindings` 有 `try/except ImportError` 保护。
4. **break 标记免费**：标记包在 generic 路径上（`unified_attention_with_output` → `get_attn_backend().forward()`），ascend 自动被包住。
5. **命中率最高**：无 request-slot padding，不受 `full_prefill_max_req` 与 2x padding 门限约束。
6. **与 CUDA 主线一致**：BCG 是 CUDA 的 prefill 默认后端，上游新特性优先落在这里。

**缺点**

1. 收益上限略低于 Full：attention 不进图，那部分 launch overhead 省不掉。
2. 依赖 `torch.npu.NPUGraph` 的**分段捕获 + pool 共享**能力（§4 待验证）。
3. CUDA 上 BCG prefill 也只有 DeepSeek-V4 opt-in 了 captured-metadata 契约，其他架构走 generic 路径 → 这条路径本身较新。

**特点**

- 段是**独立 graph 对象**而非单图内的区块，靠共享 pool 维系地址稳定性。
- 段间**输入**走弱引用（赌 pool 被 pin 住），break 点**输出**走强引用（它是下一段的静态输入地址）。

**难点**

| 难点 | 风险 | 说明 |
|---|---|---|
| NPUGraph 分段捕获能力 | **高（决定性）** | 见 §4 |
| pool 存活语义（地址稳定性） | 中 | 有兜底（弱引用改强引用） |
| `torch.npu.Stream.wait_stream` hook 的设备适配 | 低 | 捕获期需 join side stream |
| 段间 bridge buffer 显存 | 低 | 用强引用兜底时略增 |

### 1.4 `full`

**机制**

- 捕获 transformer body（`layer_model.forward`），LM head + logits_processor 走 eager tail：`python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py` 的 `_execute_body_capture`
- 固定 request 槽 `_capture_req_slots = full_prefill_max_req`，尾部槽写哨兵（`seq_lens=0`、`extend_start_loc=raw_num_tokens`）
- 捕获与 replay 前都在图外调 `init_forward_metadata_out_graph(padded_view)`
- backend：`python/sglang/srt/model_executor/runner_backend/full_cuda_graph_backend.py`（NPU 已有对应实现 `python/sglang/srt/hardware_backend/npu/graph_runner/npu_cudagraph_backend.py`，docstring 明确 "Mirrors FullCudaGraphBackend"）

**优点**

1. 特性覆盖与 BCG 同级（规则表 `rules = []`，见 `python/sglang/srt/arg_groups/cuda_graph_hook.py:351-369`）。
2. 不需要 torch.compile。
3. **注意力入图** → launch overhead 节省最彻底。
4. NPU 侧 backend 骨架已存在（decode 在用）。

**缺点**

1. **必须重写 `forward_extend` 为图安全**（去掉图内 `.cpu()/.item()/.tolist()`、动态 shape、逐请求切片）—— 最大成本项。
2. capture pool 显存代价最大（attention 中间量也固化）。
3. 固定槽位带来 padding 浪费（>2x 拒绝回放）+ `batch_size > req_slots` 直接拒绝 → 命中率受损。
4. 图数量 = bucket × chunked-prefix variants（1/2/4/8/16），capture 时间与显存同步上涨。

**特点**

- "固定形状 + 只有长度变化"是其设计核心，天然适配 Ascend FIA 的 TND + `actual_seq_lengths` 语义。
- 拒绝条件最多（`return_logprob`、`input_embeds`、`target_verify`、`capture_hidden_mode` 不足…）。

**难点**

| 难点 | 风险 |
|---|---|
| `forward_extend` 图安全化 | 中（体力活，但量大） |
| ascend `init_forward_metadata_out_graph` 扩 EXTEND + 固定列宽 block_table | 高 |
| 长度参数入图通路（单 attr rebind 限制） | **高** |
| chunked-prefix 第二套拓扑 | 中高（ascend 完全空白） |
| 静默算错（metadata 与真实 batch 不一致不报错） | 高 |

### 1.5 特性兼容矩阵（关键决策依据）

来源：`python/sglang/srt/arg_groups/cuda_graph_hook.py`

| 特性 / 配置 | `tc_piecewise` | `breakable` | `full` |
|---|---|---|---|
| **PD 分离**（prefill 角色） | ❌ 禁用 (`:240`) | ✅ | ✅ |
| **DP attention** | ❌ 禁用 (`:187`) | ✅ | ✅ |
| MoE A2A backend ≠ none | ❌ 禁用 (`:207-210`) | ✅ | ✅ |
| LoRA | ❌ 禁用 (`:211-213`) | ✅ | ✅ |
| `--enable-symm-mem` / deterministic / eplb | ❌ 禁用 (`:241-247`) | ✅ | ✅ |
| pipeline parallelism (pp>1) | ❌ 禁用 (`:189`) | ✅ | ✅ |
| multimodal 模型 | 白名单外禁用 | 白名单内 ✅ | ✅ |
| **非 CUDA 硬件 (HIP/NPU/CPU/MPS/XPU)** | ❌ **禁用 (`:190-199`)** | 无此规则 | 无此规则 |

> 注：`apply_cuda_graph_disaggregation_roles` 在 **decode 角色**下会关闭 prefill graph，故 PD 分离相关讨论只影响 P 节点。

---

## 2. 为什么选择 BCG

### 理由 1：刚需特性只有 BCG/Full 支持

PD 分离与 DP attention 是调优刚需，而它们都被 `tc_piecewise` 的黑名单挡掉（`cuda_graph_hook.py:187`、`:240`）。`tc_piecewise` 因此直接出局。

### 理由 2：BCG 完全不需要重写 `forward_extend`

这是 BCG 与 Full 的**成本分水岭**。ascend 的 `forward_extend`（`python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py:1273+`）内含：

- `seq_lens.cpu().tolist()`（`:1463`）
- 按请求的 python 循环 + 数据依赖切片（`:1470-1502`）
- `torch.empty((q.shape[0], ...))` 动态 shape

这些在 BCG 下**一行都不用改**（attention 是 break 点，跑 eager）；在 Full 下则必须整体重构（600-1500 行量级）。

### 理由 3：NPU 已验证的部分恰好覆盖了 BCG 的绝大部分内容

| BCG 需要的能力 | NPU 现状 |
|---|---|
| pool handle | ✅ `python/sglang/srt/hardware_backend/npu/graph_runner/npu_cudagraph_backend.py:69-71` 已在用 `device_module.graph_pool_handle()` |
| 传 pool + stream 捕获 | ✅ `npu_cudagraph_backend.py:113-123`：`torch.npu.graph(graph, pool=..., stream=..., auto_dispatch_capture=True)` |
| 图对象创建 | ✅ `npu_cudagraph_backend.py:96`：`torch.npu.NPUGraph()` |
| 全模型 body（含 MoE / norm / 通信）入图 | ✅ decode graph 已验证 |
| DP gather 入图 | ✅ `python/sglang/srt/hardware_backend/npu/graph_runner/npu_graph_runner.py:122-126` 明确 DP attention 下 replay 走通用 DP 图机制 |
| metadata 图外 plan + 静态 buffer + rebind | ✅ decode 已有 `init_forward_metadata_out_graph` / `_init_cuda_graph_metadata` / `_apply_cuda_graph_metadata` |

BCG 只是在此之上加两件事：**分段捕获能力** + **prefill attention 走 eager**。

### 理由 4：框架可移植性已被代码本身证明

- 捕获状态查询已有可移植分支：`breakable_cuda_graph.py:87-98`（`is_hip()` / `_is_xpu` 走 `get_device_module()`）
- 段图类已按设备分支：`breakable_cuda_graph.py:363-385`（`torch.xpu.XPUGraph` vs `torch.cuda.CUDAGraph`）
- `cuda.bindings` 有保护：`breakable_cuda_graph.py:32-35`
- pool 获取函数设备无关：`python/sglang/srt/model_executor/runner_utils/pool.py:109-115`
- `set_graph_pool_id` 是纯全局赋值，无副作用：`python/sglang/srt/distributed/device_communicators/pynccl_allocator.py:165-167`
- backend 选择和构造完全设备无关：`python/sglang/srt/model_executor/runner_backend/utils.py:107-128`

### 理由 5：break 标记免费

`eager_on_graph(True)` 直接包在 generic 路径上：

```python
# python/sglang/srt/layers/radix_attention.py:585
breakable_unified_attention_with_output = eager_on_graph(True)(unified_attention_with_output)
```

而 `unified_attention_with_output` 最终 dispatch 到 `get_attn_backend().forward()` → **ascend 自动被包住**，无需新增标记。装饰器在非捕获态是透明的（`breakable_cuda_graph.py:224-227`）。

### 理由 6：首版可以完全不碰 captured-metadata 契约

`use_captured_forward_metadata_for_breakable_cuda_graph` 基类默认 `False`（`python/sglang/srt/layers/attention/base_attn_backend.py:154`），目前只有 DSV4 开启。非 opt-in 后端在 replay 时回落普通 `init_forward_metadata`：

```python
# python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py:1178-1183
if not self.use_captured_attn_metadata:
    attn_backend.init_forward_metadata(forward_batch)
    attn_backend.prepare_prefill_shared_read_snapshot(forward_batch, num_qo_tokens=shape_key.size)
    return
```

而 `init_forward_metadata` 与 `get_cuda_graph_seq_len_fill_value`（`ascend_backend.py:864` 已实现）都是 ascend 已有的东西。

### 为什么不选 Full

| 原因 | 说明 |
|---|---|
| 唯一的大成本项它躲不开 | `forward_extend` 图安全化（600-1500 行） |
| 需要额外的 ascend metadata 契约 | `init_forward_metadata_out_graph` 扩 EXTEND + 固定列宽 block_table |
| 命中率天然受制 | `req_slots` 上限 + 2x padding 浪费门限 |
| 若日后转 BCG，那部分重写会白做 | 沉没成本风险 |

**Full 的正确定位**：BCG 的分段捕获能力若验证失败时的**退路**。

---

## 3. BCG 的实现过程

### 3.1 CUDA 现有链路（5 步）

以一次 prefill 捕获为例：

**① 取 pool、装捕获上下文** — `breakable_cuda_graph_backend.py:94-109`

```python
self._pool = get_or_create_global_graph_memory_pool(self._device_module)  # = device_module.graph_pool_handle()
set_graph_pool_id(self._pool)
with BreakableCUDAGraphCapture(cuda_graph=graph, pool=self._pool,
                               stream=self._capture_stream,
                               barrier_fn=self._tp_group.barrier):
```

**② 开第一段** — `breakable_cuda_graph.py:338-348` + `:363-385`

```python
def _begin_new_segment(self):
    graph_cls = torch.xpu.XPUGraph if _is_xpu else torch.cuda.CUDAGraph
    graph = graph_cls()                      # 每段一个独立 graph 对象
    if _is_xpu:
        graph.capture_begin(pool=self._pool)             # XPU 只收 pool
    else:
        graph.capture_begin(pool=self._pool,
                            capture_error_mode=self._capture_error_mode)
    self._current_graph = graph
```

**③ 遇到 break 点：关段 → 跑 eager → 开新段** — `breakable_cuda_graph.py:224-268`

```python
def wrapper(*args, **kwargs):
    capture = _current_capture_var.get()
    if capture is None:                    # 非捕获态 → 装饰器透明
        return inner(*args, **kwargs)

    capture._end_current_segment()         # ① capture_end()，退出捕获态
    if capture._barrier_fn is not None:
        capture._barrier_fn()              # ② rank 间重同步
    output = inner(*args, **kwargs)        # ③ 真正的 eager 执行（attention 在这里）

    captured_args = tuple(_weak_ref_if_tensor(a) for a in args)   # ④ 弱引用
    captured_output = output                                      # ⑤ 强引用（bridge buffer）

    def replay_fn():
        new_out = captured_inner(*captured_args, *…)
        return _copy_output(captured_output, new_out)
    capture.cuda_graph._break_fns.append(replay_fn)

    capture._begin_new_segment()           # ⑥ 开新段
    return output
```

**④ 关段收尾** — `breakable_cuda_graph.py:387-407`：先 join 捕获期 fork 的 side stream，再 `capture_end()`，然后 `_append_segment`。

**⑤ replay：顺序串起来** — `breakable_cuda_graph.py:284-293`

```python
def replay(self):
    for i, seg in enumerate(self._segments):
        seg.replay()
        if i < len(self._break_fns):
            self._break_fns[i]()      # 重跑 eager，结果 copy_ 回捕获时的地址
```

**时序示意**

```
捕获期:   [segment 0]─capture_end─┐
                                  ├─ eager break (attention, 任意 host 逻辑) ─┐
                                  │                                          │
                                  └──── capture_begin ── [segment 1] ─capture_end
                                             ↑ 共享同一 pool

回放期:   seg0.replay() → break_fn[0]() → seg1.replay() → break_fn[1]() → …
                          └ 重跑 eager 并 copy_ 回固定地址
```

### 3.2 NPU 需要改什么

| 文件 | 改动 | 必要性 |
|---|---|---|
| `python/sglang/srt/platforms/npu.py` | 新增 breakable 能力位（`support_piecewise_cuda_graph` 保持 `False`） | 必须 |
| `python/sglang/srt/arg_groups/cuda_graph_hook.py` | `:190-199` 的 non-CUDA 规则需按 backend 区分：不要在 breakable 路径上挡掉 NPU（建议做成 per-arch allowlist） | 必须 |
| `python/sglang/srt/model_executor/runner_backend/utils.py` | `resolve_prefill_backend` 增加 NPU 默认值分支（若走 config 显式指定则可不改） | 可选 |
| `breakable_cuda_graph/breakable_cuda_graph.py` | ① `graph_cls` 加 `torch.npu.NPUGraph`；② `_begin_new_segment`/`_end_current_segment` 的设备分支（见下方"两种分段写法"）；③ `_is_stream_capturing` 加 NPU（`:92`）；④ `wait_stream` hook 的设备类（`:105-140`） | 必须 |
| `breakable_cuda_graph/cuda_utils.py` | 无（CUDA 专用，已隔离） | — |
| `runner_backend/breakable_cuda_graph_backend.py` | 基本无需（已走 `device_module` + 设备无关的 `pool.py`） | — |
| `runner_utils/pool.py` | `GraphPoolPrecarve`（`:127-161`）用 `torch.cuda.*` / `device="cuda"`；受 `SGLANG_ENABLE_GRAPH_POOL_PRECARVE`（`environ.py:1395`，**默认 False**）门控 → 默认惰性无害；若要开启需补设备分支 | 低（可延后） |
| ascend attention | **Phase 1 无需**；Phase 3 针对 MLA/长 prefix 补 captured-metadata 契约 | 分阶段 |
| 测试 | spike 脚本 + 端到端精度回归 | 必须 |

**注意**：`NPUCudaGraphBackend` 与 `FullCudaGraphBackend` 是不同类，**BCG 不复用 `NPUCudaGraphBackend`**（后者只服务 Full 形态）。BCG 的 NPU 适配走的是 `BreakableCudaGraphBackend` + 设备分支。

**两种分段写法**（由 §4.2 Test 0 决定用哪种，两者都是 BCG，不改变方案选型）：

```python
# A) split API（CUDA/XPU 形态，BCG 现状）
graph.capture_begin(pool=self._pool, capture_error_mode=self._capture_error_mode)   # 开段
graph.capture_end()                                                                 # 关段

# B) context-manager（torch_npu 现有代码唯一用过的形态；NPUGraph 无 split API 时用）
self._seg_cm = torch.npu.graph(graph, pool=self._pool, stream=self._capture_stream)
self._seg_cm.__enter__()                        # 开段（等价于 capture_begin）
...
self._seg_cm.__exit__(None, None, None)         # 关段（等价于 capture_end）
```

### 3.3 阶段划分

| 阶段 | 内容 | 判定目标 |
|---|---|---|
| **Phase 0** | 运行 §4 的 spike 脚本 | 判定分段捕获是否可行；决定走 BCG / Full / 多图串联变体 |
| **Phase 1** | 接线 + 设备分支 + **generic metadata 路径**（`use_captured_attn_metadata=False`） | MHA 单模型单 bucket 端到端跑通 + 逐层数值对齐 eager |
| **Phase 2** | 多 bucket；PD 分离 / DP attn / MoE A2A 专项回归 | 刚需特性可用 |
| **Phase 3** | 针对 MLA / 长 prefix 架构补 captured-metadata 契约（固定列宽 + in-place refresh） | 覆盖率提升 |
| **Phase 4** | 按真实流量分布调 bucket；统计 replay 命中率；显存上限调优 | 收益验证 |

**执行顺序上的唯一硬约束**：不要先做 Full 再转 BCG —— `forward_extend` 图安全化的重写会全部白做。共享层（接线 / hook / `npu.py` / pool）两者都要，先做不吃亏。

---

## 4. NPUGraph 三个特殊场景的支持测试用例

### 4.0 待验证的三个能力

BCG 依赖 NPUGraph 的四项能力，其中三项需要实机确认：

| 编号 | 能力 | 对应测试 | 若不成立 |
|---|---|---|---|
| **U1** | 存在一种**可分段**的捕获 API：优先 `capture_begin()`/`capture_end()`（且 `capture_begin` 接受 `pool=`），退而求其次 `torch.npu.graph(graph, pool=...)` 上下文管理器可手动 `__enter__`/`__exit__` | Test 0 + A | 两者都没有 → 分段捕获不成立 → 退 Full；只有 CM 形态 → **仍是 BCG**，改接线即可 |
| **U2** | 同一 capture stream 上可**多次** begin/end，中间可夹任意 eager 执行（含 host 同步逻辑），之后能重新 begin | Test A + B | 退 Full |
| **U3** | 段图输出张量在**丢弃 Python 强引用**后，地址仍由 pool 存活保证，跨 replay 保持新鲜 | Test C / D | 弱引用改强引用 / 加 bridge buffer（+20-80 行） |

第四项（pool handle 可获取）已在 NPU decode 中被验证，无需重新测。

> **U1 的退路不能只看 `capture_begin`**：仓库里 NPU 现有代码（`npu_cudagraph_backend.py:113-123`、`vit_npu_graph_runner.py:72`、`npu_piecewise_backend.py:76`）**一律只用 `torch.npu.graph(...)` 上下文管理器**，从未调用过 `capture_begin`。因此"没有 split API"不等于"不能分段捕获" —— 手动 `__enter__`/`__exit__` 就能切段，§3.2 的 `_begin_new_segment`/`_end_current_segment` 换一种写法即可（+10~20 行）。旧版脚本把这一点误判为"退 Full"，是 §4.3 结论偏悲观的原因。

> **U3 必须用弱引用验**：只要测试里还留着段输出的 Python 强引用，地址就必然稳定，测出来的 PASS 是假阳性。`breakable_cuda_graph._weak_ref_if_tensor` 在 NPU 上落到 `torch_npu._C._weak_ref_tensor`（`weak_ref_tensor.py:9-10`），所以 NPU 侧这条链路本身是通的；spike 必须复现同样的"弱引用 + 丢强引用"形态才有判定力。

### 4.1 关于"XPU 支持则 NPU 也应该支持"的分析

用户提出的推理是合理的，但需要区分证据强度：

**支持该推理的证据**

1. `torch.xpu.XPUGraph.capture_begin` **接受 `pool=` 参数**，且这是代码注释明确记录的（`breakable_cuda_graph.py:378-380`："torch.xpu.XPUGraph.capture_begin takes only an optional pool"）→ 说明"多次 begin/end + 传 pool"这套 API 形态**不是 CUDA 独有**，PyTorch 系的图对象在多个后端上暴露了同样的方法形态。这是对 **U1 的强旁证**。
2. BCG 代码里有专门的 `_is_xpu` 分支（`:42`、`:364`、`:378`）和 HIP 的可移植捕获状态查询（`:92-94`）→ 说明作者已为非 CUDA 设备专门适配过一轮。
3. `resolve_prefill_backend`（`runner_backend/utils.py:107-128`）完全设备无关 → 只要 config 解析成 `BREAKABLE`，任何平台都会走 BCG。

**需要保留的谨慎**

1. **XPU 默认也没走 BCG**。XPU 与 NPU 同样被 `cuda_graph_hook.py:190-199` 的 "non-CUDA hardware" 规则挡掉 `tc_piecewise`，而 XPU 平台自身 `support_piecewise_cuda_graph()` 返回 `True`（`platforms/xpu.py:103-104`）→ 说明 XPU 默认路径是被禁用的，**"XPU 已跑通 BCG"缺乏证据**，那些 XPU 分支可能是防御性代码而非已验收路径。
2. **NPU 与 XPU 的能力面确实不同**：`platforms/npu.py:86-87` 的 `support_piecewise_cuda_graph()` 返回 **False**，而 XPU 返回 **True**。不能把"XPU 有"当定理推"NPU 必有"。
3. **torch_npu 是独立实现**，不是 PyTorch 上游 cuda/xpu 实现的变体，方法签名与语义需要实机确认。
4. **NPU 侧 `weak_ref_tensor` 已存在**（`weak_ref_tensor.py:9-10` 走 `torch_npu._C._weak_ref_tensor`），且 `npu_piecewise_backend.py:85` 已在用 → U3 所需的"弱引用张量"原语在 NPU 上不是空白，剩下的问题是 **pool 是否真的把这些弱引用撑住**。

**结论**：XPU 证据把 **U1 的不确定性显著降低**（方法形态大概率存在），但无法替代实机验证，尤其是 **U2（重入捕获 + eager 插缝）和 U3（pool 存活语义）** —— 这两点在 XPU 上也未必被验证过。因此 §4.2 的 spike 仍然是必要的，只是预期成功率较高。

### 4.2 Spike 脚本

将以下脚本保存为仓库根目录外的任意路径（例如 `/tmp/npu_bcg_spike.py`）并在真实 NPU 环境运行：

```bash
python /tmp/npu_bcg_spike.py
```

```python
"""
NPUGraph 分段捕获能力验证 —— BCG 适配前置 spike

验证三件事:
  U1: 同一 capture stream 上能反复 begin/end 捕获，且每次都能绑定到同一个 pool
  U2: begin 之间（非捕获态）能执行任意 eager 代码（含 .item()/.cpu() 等 host 逻辑），
      之后能重新 begin 捕获
  U3: 段图输出张量在**丢失 Python 强引用**后，地址仍因 pool 存活而稳定，
      跨 replay 保持"新鲜"

判定标准（三者缺一不可）:
  数值正确 + 不抛异常 + 不依赖 Python 强引用。
  特别是 U3 必须用弱引用张量去验：只要还留着强引用，地址就必然稳定，
  测出来的 PASS 是假阳性，掩盖了"pool 是否保住段间张量"这一唯一待验证事实。

用法:
  python npu_bcg_spike.py            # 全部测试
  python npu_bcg_spike.py --only A   # 只跑某个测试
"""
import argparse
import gc
import inspect
import sys
import traceback

import torch

try:
    import torch_npu  # noqa: F401
except ImportError:
    print("[FATAL] torch_npu 不可用")
    sys.exit(2)

DEV = "npu:0"
RESULTS = {}

# 实际生效的分段捕获形态（split API / context-manager），供结果解读使用
# desc:     生效写法
# pool_ok:  True=pool= 被接受; False=只能无 pool 捕获(共享 pool 不成立); None=未知
CAPTURE_MODE = {"desc": None, "pool_ok": None}


def weak_ref(t):
    """段图输出 -> 弱引用张量（共享 storage，但不延长其生命周期）。

    这正是 breakable 路径在 NPU 上的形态：
      breakable_cuda_graph._weak_ref_if_tensor -> weak_ref_tensors
      -> sglang/srt/compilation/weak_ref_tensor.py:10
         `from torch_npu._C import _weak_ref_tensor`
    """
    fn = getattr(getattr(torch_npu, "_C", None), "_weak_ref_tensor", None)
    if fn is None:
        raise RuntimeError("torch_npu._C._weak_ref_tensor 不可用，无法验证 U3")
    return fn(t)


class _NoSplitAPI(Exception):
    pass


def _split_begin(graph, pool):
    """CUDA 形态：graph.capture_begin(pool=...)，返回实际生效的写法。"""
    for desc, fn in (
        (
            "capture_begin(pool=pool, capture_error_mode='global')",
            lambda: graph.capture_begin(pool=pool, capture_error_mode="global"),
        ),
        ("capture_begin(pool=pool)", lambda: graph.capture_begin(pool=pool)),
        ("capture_begin()", lambda: graph.capture_begin()),
    ):
        try:
            fn()
            return desc
        except (TypeError, AttributeError):
            continue
        except Exception as e:
            # 签名存在但调用失败 -> 不能当成"没有该 API"，必须暴露出来
            raise RuntimeError(f"{desc} 抛出非签名错误: {e}") from e
    raise _NoSplitAPI("capture_begin 的三种写法均不被接受")


def begin_segment(graph, pool):
    """开始一段捕获；返回 (end_fn, mode_desc)。

    split API 优先（对应 BCG 的 _begin_new_segment / _end_current_segment）。
    若 NPUGraph 不暴露 capture_begin/capture_end，退回官方 context-manager
    `torch.npu.graph(graph, pool=...)`：手动 __enter__/__exit__ 同样能分段，
    只是 §3.2 的接线要换一种写法（+10~20 行），并非"BCG 不可行"。
    """
    if CAPTURE_MODE["desc"] != "cm":
        if hasattr(graph, "capture_begin") and hasattr(graph, "capture_end"):
            try:
                desc = _split_begin(graph, pool)
            except _NoSplitAPI:
                pass
            else:
                CAPTURE_MODE["desc"] = desc
                # U1 要求 pool= 被接受；只有 capture_begin() 可用 = 无法共享 pool
                CAPTURE_MODE["pool_ok"] = "pool=" in desc
                return graph.capture_end, desc
        CAPTURE_MODE["desc"] = "cm"

    cm = torch.npu.graph(graph, pool=pool)
    cm.__enter__()
    CAPTURE_MODE["pool_ok"] = True          # CM 形态下 pool= 是显式入参
    return (
        lambda: cm.__exit__(None, None, None),
        "cm:torch.npu.graph(graph, pool=pool)",
    )


def hr(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def record(name, ok, note=""):
    RESULTS[name] = (ok, note)
    print(f">>> {name}: {'PASS' if ok else 'FAIL/UNKNOWN'}  {note}")


# --------------------------------------------------------------------------
# Test 0: API 探测 (U1 的静态部分)
# --------------------------------------------------------------------------
def test_api_probe():
    hr("Test 0: API 探测")
    if not hasattr(torch.npu, "NPUGraph"):
        record("U1-API", False, "torch.npu.NPUGraph 不存在")
        return False

    cls = torch.npu.NPUGraph
    print(f"torch.npu.NPUGraph = {cls}")
    for name in ("capture_begin", "capture_end", "replay", "reset",
                 "update", "pool", "instantiate", "debug_dump"):
        fn = getattr(cls, name, None)
        if fn is None:
            print(f"  [MISSING] {name}")
            continue
        try:
            sig = str(inspect.signature(fn))
        except (TypeError, ValueError):
            sig = "(签名不可反射，可能是 C 扩展)"
        print(f"  [OK] {name}{sig}")

    has_split = hasattr(cls, "capture_begin") and hasattr(cls, "capture_end")
    has_cm = hasattr(torch.npu, "graph")
    print(f"\n  split API   (capture_begin/capture_end) = {has_split}")
    print(f"  ctx-manager (torch.npu.graph)           = {has_cm}")
    print(f"  torch.npu.graph_pool_handle             = {getattr(torch.npu, 'graph_pool_handle', None)}")
    print(f"  torch.npu.Stream                        = {getattr(torch.npu, 'Stream', None)}")
    print(f"  torch.npu.synchronize                   = {getattr(torch.npu, 'synchronize', None)}")

    if hasattr(torch.npu, "graph_pool_handle"):
        try:
            print(f"  graph_pool_handle() -> {torch.npu.graph_pool_handle()!r}")
        except Exception as e:
            print(f"  graph_pool_handle() 失败: {e}")

    # U3 只能靠弱引用张量来验：缺这个原语时 Test C/D 的 FAIL 不代表 NPU 不支持 pool 存活
    weak_fn = getattr(getattr(torch_npu, "_C", None), "_weak_ref_tensor", None)
    print(f"  torch_npu._C._weak_ref_tensor           = {weak_fn}")
    if weak_fn is None:
        print("  [WARN] 缺该原语 -> U3(Test C/D) 无法验证，其 FAIL 不可直接解读为'缺 pool 存活语义'")

    # 关键：两种 API 任一可用即可分段捕获；只有两者都缺才是"BCG 不可行"
    ok = has_split or has_cm
    if has_split:
        note = "split API 可用"
    elif has_cm:
        note = "仅 ctx-manager 可用（BCG 需改用 __enter__/__exit__ 分段，仍可行）"
    else:
        note = "两种 API 都不存在"
    record("U1-API", ok, note)
    return ok


# --------------------------------------------------------------------------
# 工具: 捕获流
# --------------------------------------------------------------------------
def make_capture_stream():
    return torch.npu.Stream(device=DEV)


# --------------------------------------------------------------------------
# Test A: 多次 capture_begin / capture_end (U1 + U2)
# --------------------------------------------------------------------------
def test_multi_segment():
    hr("Test A: 同一 stream 上多次 begin/end（隔离 U1/U2，段间无耦合）")
    try:
        pool = torch.npu.graph_pool_handle()
        stream = make_capture_stream()

        # 每段有**自己的**输入输出：段间没有张量依赖 -> 不可能被 U3 的失败污染
        xs = [torch.full((8,), float(v), device=DEV) for v in (1.0, 2.0, 3.0)]
        _ = xs[0] + 1
        torch.npu.synchronize()

        graphs = [torch.npu.NPUGraph() for _ in range(3)]
        outs = [None] * 3

        with torch.npu.stream(stream):
            for i, g in enumerate(graphs):
                end, used = begin_segment(g, pool)
                outs[i] = xs[i] + 1
                end()

        print(f"  捕获形态: {CAPTURE_MODE['desc']}；本段实际生效: {used}")

        # 输入改写与 replay 放在同一条流上，避免 cross-stream 竞态
        new_vals = (10.0, 20.0, 30.0)
        with torch.npu.stream(stream):
            for x, v in zip(xs, new_vals):
                x.fill_(v)
            for g in graphs:
                g.replay()
        torch.npu.synchronize()

        for i, (o, v) in enumerate(zip(outs, new_vals)):
            got = o.cpu().tolist()
            want = [v + 1.0] * 8
            if got != want:
                record("U1-多段捕获", False,
                       f"段{i} 数值错误 {got[:3]}... 期望 {want[:3]}...")
                return False

        record("U1-多段捕获", True,
               f"3 段独立 begin/end + replay 数值正确（{CAPTURE_MODE['desc']}）")
        return True
    except Exception:
        traceback.print_exc()
        record("U1-多段捕获", False, "见上方 traceback")
        return False


# --------------------------------------------------------------------------
# Test B: 段间 eager 执行，含 host 逻辑 (U2)
# --------------------------------------------------------------------------
def test_eager_gap_with_host_logic():
    hr("Test B: 段间 eager 执行（含 .item() / .cpu().tolist() 等 host 逻辑）")
    try:
        pool = torch.npu.graph_pool_handle()
        stream = make_capture_stream()
        x = torch.arange(8, dtype=torch.float32, device=DEV)

        _ = x + 1
        torch.npu.synchronize()

        g1, g2 = torch.npu.NPUGraph(), torch.npu.NPUGraph()
        host_seen = {}
        hold = {}

        with torch.npu.stream(stream):
            end, _ = begin_segment(g1, pool)
            hold["a"] = x + 1
            end()

            # ---- 非捕获态：任意 host 逻辑 ----
            # 注意：capture 期的 kernel 并未真正执行，这里读到的数值本身无意义；
            # 本测试只要求"允许执行且不报错"，数值正确性由下面的 g2 负责。
            a = hold["a"]
            host_seen["item"] = float(a.sum().item())
            host_seen["tolist"] = a.cpu().tolist()
            host_seen["shape"] = tuple(a.shape)
            host_seen["host_int_list"] = [int(v) for v in host_seen["tolist"]]
            print(f"  [in gap] item={host_seen['item']} "
                  f"host_int_list={host_seen['host_int_list']}")

            # ---- 重新进入捕获；g2 只依赖 x，不依赖 g1 的输出 ----
            end, _ = begin_segment(g2, pool)
            hold["b"] = x * 2
            end()

        # replay 验证：沟里做完 host 逻辑后还能继续捕获且结果正确
        with torch.npu.stream(stream):
            x.fill_(3.0)
            g1.replay()
            g2.replay()
        torch.npu.synchronize()

        got = hold["b"].cpu().tolist()
        want = [6.0] * 8
        ok = got == want
        record("U2-段间eager", ok,
               "沟内可做 D2H/host 逻辑，续捕段数值正确" if ok
               else f"续捕段数值错误 {got[:3]}... 期望 {want[:3]}...")
        return ok
    except Exception:
        traceback.print_exc()
        record("U2-段间eager", False, "沟内 host 逻辑或续捕失败，见 traceback")
        return False


# --------------------------------------------------------------------------
# Test C / D: 跨段数据依赖 + 弱引用（U3 的**唯一**有效验证形态）
#   C: 共享同一 pool（BCG 的实际形态）
#   D: 各自独立 pool（对照，用于判定"是否必须共享 pool"）
#
# 关键：段1 的输出降级为弱引用并**丢弃强引用**，否则地址必然稳定，
#       测不出 pool 的存活语义（这正是旧版用例的假阳性来源）。
#       段2 申请等大张量，制造"地址被复用"的机会。
# --------------------------------------------------------------------------
def _cross_segment_case(share_pool: bool):
    stream = make_capture_stream()
    x = torch.full((16,), 4.0, device=DEV)

    _ = (x + 1) * 2
    torch.npu.synchronize()

    pool_a = torch.npu.graph_pool_handle()
    pool_b = pool_a if share_pool else torch.npu.graph_pool_handle()

    g1, g2 = torch.npu.NPUGraph(), torch.npu.NPUGraph()
    box = {}

    with torch.npu.stream(stream):
        end, _ = begin_segment(g1, pool_a)
        a = x + 1
        end()

        # BCG 形态：段输出 -> 弱引用，丢掉强引用
        box["a_weak"] = weak_ref(a)
        box["a_ptr"] = a.data_ptr()
        del a
        gc.collect()

        # 段2：申请若干等大张量，制造地址复用机会
        end, _ = begin_segment(g2, pool_b)
        box["c"] = [(x * 2.0) + i for i in range(4)]
        end()

    report = []
    ok = True

    def check(tag, expect):
        nonlocal ok
        got = box["a_weak"].cpu().tolist()
        want = [expect] * 16
        good = got == want
        ok = ok and good
        addr = "ptr-same" if box["a_weak"].data_ptr() == box["a_ptr"] else "ptr-CHANGED"
        report.append(f"{tag}: {'OK' if good else f'BAD {got[:2]} want {want[:2]}'} ({addr})")

    # x=4 -> a=5：replay g1 后弱引用应读到 5
    with torch.npu.stream(stream):
        g1.replay()
    torch.npu.synchronize()
    check("g1.replay", 5.0)

    # replay g2（等大分配可能覆盖 a 的地址）后，弱引用应仍是 5
    with torch.npu.stream(stream):
        g2.replay()
    torch.npu.synchronize()
    check("g2.replay", 5.0)

    # 换输入再来一轮，排除"恰好正确"
    with torch.npu.stream(stream):
        x.fill_(7.0)
        g1.replay()
    torch.npu.synchronize()
    check("x=7 g1.replay", 8.0)

    print("  " + "; ".join(report))
    return ok


def test_cross_segment_shared_pool():
    hr("Test C: 跨段弱引用 + 共享 pool（BCG 形态）")
    try:
        ok = _cross_segment_case(share_pool=True)
        record("U3-共享pool", ok,
               "弱引用段输出跨 replay 保持新鲜" if ok
               else "弱引用失效/被覆盖（需 bridge buffer 兜底）")
        return ok
    except Exception:
        traceback.print_exc()
        record("U3-共享pool", False, "见 traceback")
        return False


def test_cross_segment_independent_pool():
    hr("Test D（对照）: 跨段弱引用 + 各自独立 pool")
    try:
        ok = _cross_segment_case(share_pool=False)
        print("  说明: D 失败是**预期可能**的——正好证明'必须共享 pool'；"
              "D 成功则说明 NPU 上独立图池也能保住地址，显存管理更灵活。")
        record("U3-对照独立pool", ok,
               "独立池也可行（可放宽）" if ok else "独立池下地址被复用（如预期）")
        return ok
    except Exception:
        traceback.print_exc()
        record("U3-对照独立pool", False, "见 traceback（如预期）")
        return False


# --------------------------------------------------------------------------
# 结果解读
# --------------------------------------------------------------------------
def summarize():
    hr("结果解读")
    u1 = RESULTS.get("U1-多段捕获", (False, ""))[0]
    u2 = RESULTS.get("U2-段间eager", (False, ""))[0]
    u3 = RESULTS.get("U3-共享pool", (False, ""))[0]
    u3d = RESULTS.get("U3-对照独立pool", (False, ""))[0]
    api = RESULTS.get("U1-API", (False, ""))[0]
    mode = CAPTURE_MODE["desc"]
    pool_ok = CAPTURE_MODE["pool_ok"]

    print(f"  U1  多次 begin/end : {'PASS' if u1 else 'FAIL'}")
    print(f"  U2  段间 eager     : {'PASS' if u2 else 'FAIL'}")
    print(f"  U3  共享 pool+弱引用: {'PASS' if u3 else 'FAIL'}")
    print(f"  U3  独立 pool(对照) : {'PASS' if u3d else 'FAIL'}")
    print(f"  捕获形态            : {mode}")
    print(f"  pool= 是否被接受    : {pool_ok}")
    print()

    if pool_ok is False:
        print("  [!] 只能做到 'capture_begin()' 无 pool 捕获：BCG 的共享 pool 语义不成立，")
        print("      U3 的 PASS 不可信（多个图各用默认池）。需按情形 2 的 bridge buffer 兜底。")

    if u1 and u2 and u3:
        print("  ==> 情形 1：BCG 可直接适配。按 §3.3 Phase 1 推进。")
        if mode == "cm":
            print("      注：NPUGraph 未暴露 split API，实现时把 §3.2 的")
            print("      _begin_new_segment/_end_current_segment 改成 torch.npu.graph 的")
            print("      手动 __enter__/__exit__（+10~20 行），不要因此退 Full。")
        if u3d:
            print("      注：Test D 也通过 -> 段间地址在独立池下同样稳定，显存管理可放宽。")
    elif u1 and u2 and not u3:
        print("  ==> 情形 2：分段捕获可行，但缺 pool 存活语义。")
        print("      兜底：把 breakable_cuda_graph.py:252 的弱引用改为强引用 /")
        print("      显式 bridge buffer（+20-80 行），仍是'适配'而非重设计。")
    else:
        print("  ==> 情形 3：分段捕获在 NPU 上不成立。")
        if not api:
            print("      原因：split API 与 torch.npu.graph 均不可用 -> 只能退 Full。")
        else:
            print("      原因：API 存在但多段/EAGER 插缝不成立（Test A/B 失败）。")
        print("      退路 a：改用 Full（单图捕获，约 600-1100 行）")
        print("      退路 b：'N 个独立图顺序 replay + 显式 bridge buffer' 的变体")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, choices=list("ABCD"))
    args = ap.parse_args()

    hr(f"环境信息\ntorch={torch.__version__}\ndevice={DEV}")
    print(f"  device_name = {torch.npu.get_device_name(0)}")

    # 任何子测试都依赖 API 探测结果，因此总是先跑 Test 0
    if not test_api_probe():
        print("\n[FATAL] 两种分段捕获 API 都不存在，后续测试无意义")
        summarize()
        return

    tests = {
        "A": test_multi_segment,
        "B": test_eager_gap_with_host_logic,
        "C": test_cross_segment_shared_pool,
        "D": test_cross_segment_independent_pool,
    }
    if args.only:
        tests[args.only]()
    else:
        for name in "ABCD":
            tests[name]()

    summarize()


if __name__ == "__main__":
    main()
```

### 4.3 判定矩阵

| Test 0 | Test A | Test B | Test C | 结论 | 后续 |
|---|---|---|---|---|---|
| ✅ | ✅ | ✅ | ✅ | **情形 1**：分段捕获 + pool 存活均可用 | 直接按 Phase 1 做 BCG（200-600 行） |
| ✅(仅 CM) | ✅ | ✅ | ✅ | **情形 1′**：无 split API，但 CM 分段可用 | 仍是 BCG：`_begin_new_segment`/`_end_current_segment` 改用 `torch.npu.graph` 手动 enter/exit（+10-20 行），**不要退 Full** |
| ✅ | ✅ | ✅ | ❌ | **情形 2**：缺 pool 存活语义 | 弱引用改强引用 / 加 bridge buffer（+20-80 行），仍属"适配" |
| ✅ | ❌ | — | — | **情形 3a**：无法多次 begin/end | 退 Full（600-1100 行） |
| ✅ | ✅ | ❌ | — | **情形 3b**：无法在捕获中间执行 eager | 同上，退 Full |
| ❌ | — | — | — | **情形 3c**：split API 与 `torch.npu.graph` **都不存在** | 只能退 Full |

**Test D 的作用**：它是一个"预期可能失败"的对照组。若 D 失败、C 成功 → 证明 pool 共享是**必要**的（U3 是关键依赖）；若 D 也成功 → 说明 NPU 上段间地址天然稳定，将来甚至可以放宽为独立图池，显存管理更灵活。

> **判定口径说明**：Test C/D 都用"弱引用段输出 + 丢强引用 + 后续段申请等大张量"来施加地址复用压力，并检查弱引用在 `g1.replay()` / `g2.replay()` / 换输入再 replay 三次观察点上的数值。**只有这种形态的 PASS 才算 U3 通过**；若改用强引用，任何实现都会 PASS，判定无效。

---

## 5. 不同前提下的代码量

### 5.1 按 spike 结果分档

| 情形 | 最小可用（Phase 0-1） | 完整能力（含 Phase 2-3） | 说明 |
|---|---|---|---|
| **情形 1**（全通过） | **200-400 行** | **450-1100 行** | 首选路径 |
| **情形 1′**（仅 CM 分段） | 210-420 行 | 460-1120 行 | 与情形 1 同，仅多 `torch.npu.graph` 手动 enter/exit 的接线 |
| **情形 2**（U3 失败） | 220-480 行 | 470-1180 行 | 仅加 bridge buffer / 强引用 |
| **情形 3**（U1/U2 失败 → 转 Full） | 600-1100 行 | 1400-2900 行 | 含 `forward_extend` 重写 |
| 情形 3 退路 b（多图串联变体） | 400-800 行 | 800-1500 行 | 新设计，不共享 pool，靠 `copy_` 传递 |

### 5.2 情形 1 的分文件明细

| 文件 | 改动内容 | 估算行数 |
|---|---|---|
| `python/sglang/srt/platforms/npu.py` | breakable 能力位 | 3-5 |
| `python/sglang/srt/arg_groups/cuda_graph_hook.py` | non-CUDA 规则按 backend 拆分（breakable 路径放行 NPU，建议 per-arch allowlist） | 5-15 |
| `python/sglang/srt/model_executor/runner_backend/utils.py` | `resolve_prefill_backend` 的 NPU 默认值（可选） | 0-10 |
| `breakable_cuda_graph/breakable_cuda_graph.py` | ① `graph_cls` = `NPUGraph`；② `capture_begin` 设备分支；③ `_is_stream_capturing` 加 NPU（`:92`）；④ `wait_stream` hook 的设备类（`:105-140`） | 30-80 |
| `runner_backend/breakable_cuda_graph_backend.py` | 基本无需（已设备无关） | 0-10 |
| `runner_utils/pool.py` | `GraphPoolPrecarve` 设备分支（仅当开启 precarve，默认关闭可延后） | 0-20 |
| ascend attention（Phase 1） | **无需改动** | 0 |
| ascend attention（Phase 3，captured-metadata 契约） | `init_forward_metadata_for_breakable_cuda_graph_capture` + `prepare_forward_metadata_for_breakable_cuda_graph_replay` + metadata in-place refresh（针对 MLA / 长 prefix） | 150-400 |
| 测试 | spike 脚本 + 端到端精度回归（replay vs eager 逐层对比） | 100-300 |
| **最小可用合计** | | **约 200-400 行 / 5-6 个文件** |
| **完整能力合计** | | **约 450-1100 行 / 6-8 个文件** |

### 5.3 与 Full 的对照

| 工作项 | BCG | Full |
|---|---|---|
| backend 接线 + 设备分支 | 30-80 行 | 40-80 行（复用 `NPUCudaGraphBackend`） |
| **`forward_extend` 图安全化** | **0** | **200-400 行** |
| ascend extend metadata 契约 | 0（Phase 1 用 generic 路径） | 150-300 行（`init_forward_metadata_out_graph` 扩 EXTEND） |
| 长度参数入图通路 | 0 | 60-120 行（单 attr rebind 限制） |
| chunked-prefix 第二拓扑 | 0 | 200-500 行 |
| fixed-context | 0 | 150-300 行 |
| 测试 | 100-300 行 | 200-400 行 |

**差距的来源就是两条**：`forward_extend` 是否要重写，以及 metadata 是否要固定列宽。

---

## 6. 风险与未决问题

| # | 风险 | 等级 | 应对 |
|---|---|---|---|
| 1 | NPUGraph 分段捕获能力不足 | **高（决定性）** | Phase 0 spike；失败退 Full 或改多图串联 |
| 2 | pool 存活语义缺失 | 中 | 弱引用改强引用（代码已有先例：break 输出即强引用 + `_copy_output`） |
| 3 | 收益上限低于 Full（attention 不进图） | 中 | prefill attention 通常 kernel 数少、单 kernel 大，而 kernel 最碎的 MoE 仍入图 → 差距预计有限，需 Phase 4 实测 |
| 4 | BCG prefill 在 CUDA 上也不算成熟（仅 DSV4 opt-in captured-metadata） | 中 | 首版限定 1-2 个目标模型 + 有限 bucket |
| 5 | 静默算错（metadata 与真实 batch 不一致不报错） | 高 | **必须**做 replay vs eager 逐层数值对比，作为入场券 |
| 6 | 命中率被拒绝条件影响 | 中 | Phase 4 按真实流量统计，不假设"开了就满载" |
| 7 | `torch.npu.Stream.wait_stream` hook 的设备适配（捕获期 side stream join） | 低 | 按 `graph_cls` 同样的方式做设备分支 |
| 8 | `GraphPoolPrecarve` 的 CUDA 硬编码 | 低 | `SGLANG_ENABLE_GRAPH_POOL_PRECARVE` 默认 False，惰性无害；开启前补设备分支 |

**未决问题（需向 torch_npu 侧确认）**

1. `torch.npu.graph` 的 `auto_dispatch_capture=True` 具体语义 —— 若它意味着"不可捕获算子自动派发/降级"，对降低整条链路门槛是加分项（但**不能替代 BCG**，因为 BCG 要的是"在指定位置断开并执行任意 eager 代码"）。需查 torch_npu 文档确认。
2. `torch.npu.NPUGraph` 的 pool 生命周期语义：`__del__` 是否释放 pool、pool 是否按"存活图数量"计数。
3. NPU 上多个 NPUGraph 共享 pool 时的显存记账方式（影响 `full_prefill_max_req` 类上限的设计）。

---

## 附录：关键代码位置索引

| 主题 | 位置 |
|---|---|
| 分段捕获容器 | `python/sglang/srt/model_executor/runner_backend_utils/breakable_cuda_graph/breakable_cuda_graph.py` |
| ├ pool / 弱引用设计说明 | `:14-23` |
| ├ 设备可移植捕获状态查询 | `:87-98` |
| ├ break 点装饰器逻辑 | `:224-268` |
| ├ replay 循环 | `:284-293` |
| ├ 开新段（设备分支） | `:363-385` |
| └ 关段 + side stream join | `:387-407` |
| BCG backend | `python/sglang/srt/model_executor/runner_backend/breakable_cuda_graph_backend.py` |
| ├ 取 pool | `:94-109` |
| └ replay | `:246-259` |
| Full backend | `python/sglang/srt/model_executor/runner_backend/full_cuda_graph_backend.py` |
| Prefill runner（三方案共用） | `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py` |
| ├ reset slot 填充 | `:453-457` |
| ├ captured-metadata 契约开关 | `:569-580` |
| ├ capture-time metadata | `:1125-1143` |
| ├ replay-time metadata 刷新 | `:1145-1190` |
| ├ BCG 无 request-slot padding | `:1953-1956` |
| └ 回放准入判定 | `:1199-1272` |
| 特性兼容规则 | `python/sglang/srt/arg_groups/cuda_graph_hook.py` |
| ├ 规则只按 resolved backend 筛选 | `:167-172` |
| ├ tc_piecewise 黑名单 | `:182-260` |
| ├ DP attention | `:187` |
| ├ 非 CUDA 硬件 | `:190-199` |
| ├ PD 分离 | `:240` |
| ├ full 规则表（空） | `:351-369` |
| └ PD 角色处理 | `:697-712` |
| break 标记 | `python/sglang/srt/layers/radix_attention.py:585-590`（MHA）<br>`python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mla.py:1004`（MLA）<br>`python/sglang/srt/layers/radix_linear_attention.py:242`（linear）<br>`python/sglang/srt/layers/moe/ep_moe/layer.py:215`（MoE A2A） |
| NPU decode graph backend | `python/sglang/srt/hardware_backend/npu/graph_runner/npu_cudagraph_backend.py` |
| ├ 取 pool | `:69-71` |
| ├ 建图 + 捕获 | `:96-124` |
| └ pool rebind | `:135-193` |
| NPU DP gather 入图（已验证） | `python/sglang/srt/hardware_backend/npu/graph_runner/npu_graph_runner.py:122-126` |
| NPU 平台能力位 | `python/sglang/srt/platforms/npu.py:82-87` |
| XPU 平台能力位 | `python/sglang/srt/platforms/xpu.py:100-104` |
| ascend attention | `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`（`forward_extend` 起点 `:1273`，`:1463`、`:1470-1502`） |
| ascend 设备侧 compact 原语 | `python/sglang/srt/hardware_backend/npu/attention/ascend_dsv4_backend.py:1395-1442` |
| backend 选择（设备无关） | `python/sglang/srt/model_executor/runner_backend/utils.py:107-128` |
| pool 工具 | `python/sglang/srt/model_executor/runner_utils/pool.py:109-161` |