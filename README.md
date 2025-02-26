# EfficientPyTorch
在学习张爱玲和杨占略合著的 《PyTorch性能与显存优化手册》时，记录的一些疑问。其中大部分疑问和 Claude 3.5 Sonnet、DeepSeek-R1 等讨论过。
原书代码地址参见：https://github.com/ailzhang/EfficientPyTorch

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
