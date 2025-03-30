# EfficientPyTorch
在学习张爱玲和杨占略合著的 《PyTorch性能与显存优化手册》时，记录的一些疑问。其中大部分疑问和 Claude 3.5 Sonnet、DeepSeek-R1 等讨论过。
原书代码地址参见：https://github.com/ailzhang/EfficientPyTorch

# 3.1 PyTorch 的张量数据结构
1. 张量是数据容器，提供了统一的方式来处理不同维度和形状的数据；
2. 插入一个新的维度：
```python
# x is a tensor with shape of [10, 20]
x = x[:, None, :]
# 返回张量的形状为 [10, 1, 20]
```
3. 张量的本质：对其底层数据存储的一种特定的访问和解释方式。每个张量底层都有一个 `torch.Storage` 来存储数据，多个张量可以共享同一个 `torch.Storage`.
4. 对于连续存储的矩阵，通过指定 stride 属性，可以实现高效的访问。比如通过 as_strided （跳跃访问） 函数实现矩阵转置，无需额外的数据复制。
5. 连续性检测与“使连续”：`tensor.is_contiguous()` 和 `tensor.contiguous()`(会创建副本)。

# 3.2 PyTorch 中的算子
1. Pytorch 算子库包括基础数学运算、线性代数运算、逻辑和比较运算、张量操作、AI模型层与损失函数等特殊运算。
2. 逻辑和比较运算示例：
-  逻辑与 (logical_and)、逻辑或 (logical_or)、等于(eq)、大于 (gt)、小于(lt)
3. PyTorch 提供了两种执行操作的方式：
- torch 命名空间下的算子函数
- Tensor 类的方法

```python
import torch

# 加法

x = torch.ones(4,4)
# torch 命名空间下加法
y1 = x.add(x)
# Tensor 类的加法
y2 = torch.add(x, x)
# 两种方式在数学上是等价的
assert (y1 == y2).all()

# 等于
a = torch.tensor([1, 2, 3, 4])
b = torch.tensor([1, 2, 5, 4])
# 使用 torch 命名空间
result1 = torch.eq(a, b)  # tensor([True, True, False, True])
# 使用 Tensor 类方法
result2 = a.eq(b)         # tensor([True, True, False, True])

# 逻辑
# Create boolean tensors
x = torch.tensor([True, True, False, False])
y = torch.tensor([True, False, True, False])

# Logical AND
# torch namespace
and_result1 = torch.logical_and(x, y)  
# tensor([True, False, False, False])

# tensor method
and_result2 = x.logical_and(y)
# tensor([True, False, False, False])         
```

4. 关于算子的内存分配：
- 原位操作有助于减少内存分配和避免数据的复制开销，对节省内存和提升性能都有帮助，但使用不当可能带来副作用，如backward过程中，原位操作可能导致数值错误。
- add_() 是 add() 的原位操作版本，其它算子同理；
- x += y 触发原位操作，x = x + y 则创建新张量。
- 视图操作 view() 中，输出张量与输入张量之间共享底层内存，不会造成额外内存分配。
- 基础索引与输入张量共享底层存储，高级索引则导致额外的内存分配。

```python
import torch 

x = torch.arange(200).reshape(10, 20).contiguous()

# 基础索引，返回张量和x共享底层存储
y_basic_index = x[0]
assert y_basic_index.data_ptr() == x.data_ptr()

# 高级索引，返回张量的底层存储是新开辟的
# 返回在 [0,2], [1,3], [2,4] 位置的元素
z_adv_index_int = x[torch.tensor([0, 1, 2]), torch.tensor([2, 3, 4])]

# 使用布尔张量对x进行高级索引
ind = x < 10
z_adv_index_bool = x[ind]

assert z_adv_index_int.data_ptr() != x.data_ptr()
assert z_adv_index_bool.data_ptr() != x.data_ptr()

```
- 无论基于哪种索引方式赋值，原始张量都会被改变。

5. 算子调用的开销是性能杀手之一。

```python
import torch

x1 = torch.rand(32, 32, dtype=torch.float32, device="cuda:0")
x2 = torch.rand(32, 32, dtype=torch.float32, device="cuda:0")

y = x1 @ x2
```

完整流程如下：

- 函数入口：调用Python中Tensor类的 __matmul__ 方法作为 “矩阵乘法” 算子的入口
- 算子定位：PyTorch 的 分发系统 （dispatcher）寻找底层算子实现：找到GPU上float32类型矩阵乘法计算对应的cuda函数实现
- 创建输出张量
- 底层调用，计算，并写入输出张量
- 函数返回：创建输出张量的Python对象，并返回用户。

概括一下就是: 函数入口 -> 定位算子 -> 创建张量 -> 底层调用 -> 函数返回。
除了底层调用对应的GPU实际计算耗时外，其它两端的开销都属于调用延迟，算子太碎、调用太频繁，会导致总的调用延迟较高。因此，能对张量进行整体操作的时候，尽量整体操作，尽可能避免单个、局部元素的单独操作。

# 6.3 减少 CPU 和 GPU 间的同步

```python
import torch
import torch.nn as nn
from torch.profiler import profile, ProfilerActivity


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(1000, 5000)
        self.linear2 = nn.Linear(5000, 10000)
        self.linear3 = nn.Linear(10000, 10000)
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.relu(self.linear1(x))
        output = self.relu(self.linear2(output))
        output = self.relu(self.linear3(output))
        nonzero = torch.nonzero(output)
        return nonzero


def run(data, model):
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(10):
            model(data)
    prof.export_chrome_trace("traces/PROF_nonzero.json")


data = torch.randn(1, 1000, device="cuda")
model = Model().to(torch.device("cuda"))
run(data, model)

```

**原书关于nonzero操作导致隐式同步的分析**

1. **现象解释**：
- `torch.nonzero()`触发GPU到CPU的数据拷贝（cudaMemcpyAsync）
- 这导致了隐式同步，使CPU等待GPU计算完成
- 同步是必需的，因为需要知道非零元素的数量来分配内存

2. **性能瓶颈分析**：
- 真正的耗时点在于模型的forward计算，而不是同步操作本身
- CPU必须等待GPU完成所有前序计算才能获得非零元素信息
- 如果业务逻辑需要非零元素信息，这个等待是不可避免的
- 原书中的分析更多是解释了为什么会出现这种行为，而不是指出一个需要优化的问题
- 当然，这里的统计非零元素的操作本身也是可以优化的

3. **优化思路与局限**：
- 可以将非零元素统计放在GPU上完成，利用GPU的并行特性
- 可以尝试批量处理减少同步次数，但收益取决于GPU在大batch下的效率提升
- 可以利用GPU计算的异步特性，在等待期间让CPU执行其他任务

4. **关键结论**：
- 代码中的同步行为不是性能问题，而是业务需求导致的必要开销
- 优化应该着眼于整体架构设计，而不是纠结于同步时机的调整
- 真正的优化空间在于如何更好地利用GPU-CPU并行计算的特性


# 6.4 降低程序中的额外开销

1. PyTorch 通过Python提供接口易用性，通过C++实现高性能，两层逻辑，两层开销；Python层的开销属于 “易用税”

2. 直接在GPU上创建张量，避免在CPU创建/初始化后在拷贝到GPU 

3. 大部分 PyTorch 操作默认会为返回变量创建新内存

4. 尽量使用张量的原位操作，避免显式/隐式创建新张量：

```python
# inplace
# 进程无须申请显存，直接向gpu提交任务：cudaLaunchKernel
data.mul_(2)

# non-inplace
data = data.mul(2)
```

5. 尽量使用 inplace 操作，避免显式/隐式创建新张量：

```python
data.mul_(2)
```

6. 关闭不必要的梯度计算，如模型微调场景下，预训练模型的前向；如推理场景：

```python
torch.no_grad()
model.eval()
torch.inference_mode()
with torch.inference_mode(mode=True/False):
    # 上下文管理，灵活开/关
```

# 6.5 有损性能优化

1. 数据压缩，低精度；比如图像数据，每个像素的取值范围都在 [0,255] 之间，因此 1 个字节的 unit8 足以表示，无须使用 2 - 4 字节的 float；
2. 明确数据的值域很重要，方便判断低精度压缩的程度，该用多大的压缩率，损失多大；
3. 尽量使用性能特化过的优化器，如算子融合后的 fused, for-each 等，而不是原生的 for-loop; 减少算子的调用、调度开销，提高性能。


# A1. CUDA Graph 与 torch.compile

## 1. CUDA Graph

### 1.1 基本概念
CUDA Graph 是 NVIDIA 在 CUDA 10.0 引入的特性，允许将一系列 GPU 操作（kernel 启动、内存拷贝等）预定义为图结构，一次性执行整个计算图。本质上是执行过程的优化，而非计算本身的优化。

### 1.2 工作原理
**两阶段执行模式**：
- **捕获阶段**：记录完整的 GPU 操作序列
- **重放阶段**：一次性提交整个操作序列给 GPU 执行

**具体实现流程**：
```python
# 捕获阶段
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    output = model(input_tensor)

# 重放阶段
graph.replay()
```

### 1.3 性能提升机制
1. **减少 CPU-GPU 通信开销**：传统执行每个操作都需要 CPU 参与，CUDA Graph 只需一次 CPU 调用
2. **优化调度**：GPU 可以全局规划资源分配
3. **减少 kernel 启动延迟**：消除 CPU 指令分发和驱动层处理的延迟
4. **提高 GPU 利用率**：减少 GPU 空闲等待时间

### 1.4 使用限制
1. **输入固定**：图捕获后，输入形状必须保持一致
2. **静态执行图**：不能包含动态控制流
3. **固定内存分配**：不支持动态内存分配/释放

## 2. torch.compile 

### 2.1 基本概念
torch.compile 是 PyTorch 2.0 引入的即时编译（JIT）技术，通过编译优化将 PyTorch 模型转换为更高效的计算代码。本质上是计算内容的优化。

### 2.2 工作原理
**多阶段编译优化流程**：
1. **计算图捕获**：通过 FX 图捕获框架获取模型计算图
2. **图优化**：进行算子融合、死代码消除等优化
3. **代码生成**：生成针对特定硬件的优化代码（如 Triton kernel）
4. **编译执行**：编译生成的代码并缓存供后续使用

**核心实现示例**：
```python
# 基本使用方式
optimized_model = torch.compile(
    model,
    mode="max-autotune",
    dynamic=False
)
output = optimized_model(input_tensor)
```

### 2.3 性能提升机制
1. **操作融合**：将多个小操作合并为一个大操作，减少内存访问
2. **内存优化**：减少中间结果，优化内存布局
3. **自动调优**：为特定 GPU 架构生成优化的 kernel 参数
4. **Triton kernel 生成**：使用 Triton 编程模型生成高效 CUDA kernel

### 2.4 使用限制
1. **首次编译开销**：初次编译时间可能很长（数十秒至数分钟）
2. **内存消耗**：编译过程需要较大内存
3. **兼容性**：某些自定义操作可能不兼容
4. **调试难度**：生成代码难以直接调试

## 3. SGLang 中的实现与优化策略

### 3.1 CUDA Graph 实现策略

**多 batch size 预捕获**：
```python
def get_batch_sizes_to_capture(model_runner: ModelRunner):
    # 如果未指定，使用预定义的 batch size 列表
    if capture_bs is None:
        if server_args.disable_cuda_graph_padding:
            capture_bs = list(range(1, 33)) + [64, 96, 128, 160]
        else:
            capture_bs = [1, 2, 4] + [i * 8 for i in range(1, 21)]
```

**处理可变 batch size**：
```python
def replay_prepare(self, forward_batch: ForwardBatch):
    # 向上匹配最接近的 batch size
    index = bisect.bisect_left(self.capture_bs, raw_bs)
    bs = self.capture_bs[index]
    # 当需要填充时，进行特殊处理
    if bs != raw_bs:
        self.seq_lens.fill_(1)
        self.out_cache_loc.zero_()
```

**内存管理优化**：
```python
# 重用内存池减少碎片
with torch.cuda.graph(graph, pool=global_graph_memory_pool, stream=stream):
    out = run_once()
global_graph_memory_pool = graph.pool()
```

### 3.2 torch.compile 实现策略

**选择性编译**：
```python
# 只对较小的 batch size 应用 torch.compile
compile_bs = [bs for bs in capture_bs if bs <= server_args.torch_compile_max_bs] 
    if server_args.enable_torch_compile else []
```

**模型自适应**：
```python
def _to_torch(model: torch.nn.Module, reverse: bool, num_tokens: int):
    for sub in model._modules.values():
        if isinstance(sub, CustomOp):
            if reverse:
                sub._forward_method = sub.forward_cuda
            else:
                # 特殊处理某些层
                if "FusedMoE" in sub.__class__.__name__:
                    if num_tokens == 1:
                        sub._forward_method = fused_moe_forward_native
                else:
                    sub._forward_method = sub.forward_native
```

### 3.3 两者结合策略

**分层优化应用**：
```python
with patch_model(
    self.model_runner.model,
    bs in self.compile_bs,  # 条件性使用 torch.compile
    num_tokens=bs * self.num_tokens_per_bs,
    tp_group=self.model_runner.tp_group,
) as forward:
    # 再通过 CUDA Graph 捕获优化后的执行
    graph, output_buffers = self.capture_one_batch_size(bs, forward)
```

## 4. 关键技术细节与最佳实践

### 4.1 CUDA Graph 关键技术细节

**预分配内存与输入更新**：
```python
# 预分配足够大的缓冲区
self.input_ids = torch.zeros((self.max_num_token,), dtype=torch.int64)

# 重放前更新输入
self.input_ids[:raw_num_token].copy_(forward_batch.input_ids)
```

**静态填充值处理**：
```python
# 使用特定值填充序列长度
self.seq_len_fill_value = self.model_runner.attn_backend.get_cuda_graph_seq_len_fill_value()
self.seq_lens = torch.full((self.max_bs,), self.seq_len_fill_value, dtype=torch.int32)
```

**Graph 捕获预热**：
```python
# 捕获前预热执行两次，确保稳定
for _ in range(2):
    torch.cuda.synchronize()
    self.model_runner.tp_group.barrier()
    run_once()
```

### 4.2 torch.compile 关键技术细节

**动态性控制**：
```python
yield torch.compile(
    torch.no_grad()(model.forward),  # 固定为推理模式
    mode="max-autotune-no-cudagraphs",  # 使用最大自动调优
    dynamic=False,  # 关闭动态形状支持
)
```

**自定义操作处理**：
```python
# 为自定义操作设置特定标记和实现方法
if isinstance(sub, CustomOp):
    sub._forward_method = sub.forward_native
    setattr(sub, "is_torch_compile", True)
```

**缓存管理**：
```python
# 调整缓存大小限制
torch._dynamo.config.accumulated_cache_size_limit = 1024
if hasattr(torch._dynamo.config, "cache_size_limit"):
    torch._dynamo.config.cache_size_limit = 1024
```

### 4.3 内存管理策略

**显存占用控制**：
- CUDA Graph 通常增加 10-30% 的基本模型内存占用
- 预捕获的 batch size 范围直接影响显存占用
- 使用共享内存池减少碎片

**权衡与调优参数**：
```python
# 用户可配置参数
parser.add_argument("--cuda-graph-max-bs", type=int, default=160)
parser.add_argument("--torch-compile-max-bs", type=int, default=32)
parser.add_argument("--disable-cuda-graph-padding", action="store_true")
```

## 5. 性能对比与实际应用

### 5.1 性能提升分析

**典型性能提升**：
- CUDA Graph：小 batch 场景下减少 30%-70% 推理延迟
- torch.compile：视模型结构提升 10%-30% 计算效率
- 结合使用：可达到 30%-60% 的综合性能提升

**批量大小影响**：
- 小批量（1-8）：CUDA Graph 提升最显著
- 中等批量（8-64）：两者结合效果最佳
- 大批量（>64）：torch.compile 相对贡献更大

### 5.2 应用场景适配

**适用场景**：
- 在线推理服务：低延迟、小批量
- 批处理任务：固定输入大小，重复执行
- 多模态模型：涉及多个执行步骤

**不适用场景**：
- 高度动态的批量大小和输入形状
- 需频繁修改模型参数的场景（如强化？）
- 内存极度受限的环境

### 5.3 部署建议

**关键部署参数**：
- 合理设置 `cuda-graph-max-bs` 和 `torch-compile-max-bs`
- 预热阶段分配足够时间完成编译
- 监控内存使用情况

**框架选择**：
- 延迟敏感型：优先 CUDA Graph
- 吞吐量敏感型：优先 torch.compile
- 生产环境：两者结合，依据硬件资源调整配置

## 6. 总结与展望

### 6.1 关键结论

1. **互补优势**：CUDA Graph 优化执行调度，torch.compile 优化计算内容
2. **实现策略**：预捕获多种 batch size + 向上匹配填充是处理动态批量的有效方法
3. **内存权衡**：需在性能提升和额外内存消耗间取得平衡
4. **框架集成**：SGLang 展示了两种技术如何有效结合，实现最佳性能

### 6.2 未来趋势

1. **编译优化进展**：更智能的自动调优和操作融合
2. **内存效率**：更高效的内存管理策略减少冗余
3. **动态支持**：改进对动态形状的处理能力
4. **硬件协同**：更深度结合特定硬件架构特性

## 7. RL 训练中的使用

1. 对于 CUDA Graph, 如果仅仅是模型的参数值发生了变化（如一轮训练后被更新），则通常不需要重新进行捕获；
2. 但如果参数更新的过程中，参数张量的内存地址，即地址指针发生了变化，则 cuda graph 需要重新捕获，这种情况下，使用 cuda graph 的效率就比价低；
3. 参数精度变化：如从 FP32 切换到 FP16 或 INT8，也会触发重捕获；