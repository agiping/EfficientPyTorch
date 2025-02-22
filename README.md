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

