# EfficientPyTorch
在学习张爱玲和杨占略合著的 《PyTorch性能与显存优化手册》时，记录的一些疑问。其中大部分疑问和 Claude 3.5 Sonnet、DeepSeek-R1 等讨论过。
原书代码地址参见：https://github.com/ailzhang/EfficientPyTorch

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

