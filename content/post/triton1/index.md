---
title: Triton学习——Vector Addition, Fused Softmax
description: Triton学习笔记（一）
slug: Triton1
date: 2025-03-06 10:23:00+0800
math: true
image: img/cover.jpg
categories:
    - 文档
    - AI Infra
tags:
    - 文档
    - AI Infra
weight: 11
---
## Reference

[教程 | Triton 中文站](https://triton.hyper.ai/docs/getting-started/tutorials)

[Tutorials — Triton documentation](https://triton-lang.org/main/getting-started/tutorials/index.html)

[并行编程语言（Triton & 九齿） - 2024 冬季大模型与人工智能系统训练营](https://opencamp.cn/InfiniTensor/camp/2024winter/stage/7)

## Vector Addition

### Compute Kernel

```python
import torch

import triton
import triton.language as tl

DEVICE = torch.device("cuda:0")


@triton.jit
def add_kernel(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               # NOTE: `constexpr` so it can be used as a shape value.
               ):
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    block_start = pid * BLOCK_SIZE # Offsets is a list of pointers  
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n_elements
    # Load x and y from DRAM
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    # Write x + y back to DRAM.
    tl.store(output_ptr + offsets, output, mask=mask)
```

> 编译遇到的第一个报错是`AttributeError: 'CudaDriver' object has no attribute 'get_active_torch_device'`，在[issue#5388](https://github.com/triton-lang/triton/issues/5388)下找到了原因和解决方案，`DEVICE = triton.runtime.driver.active.get_active_torch_device()`似乎对应了旧本版本，用`DEVICE = torch.device("cuda:0")`可以暂缓燃眉之急。

kernel的实现思路是简单的：

+ 对于输入`(input_x, input_y, output, n, BLOCK_SIZE)`，获取`programd_id`，通过`pid * BLOCK_SIZE`计算分块索引，通过`block_start + tl.arange(0, BLOCK_SIZE)`确定每个元素的索引

+ 确定`mask`，防止边界溢出

+ 加载输入元素，实现向量加法，最后写回DRAM

接下来需要一个合适的辅助函数，负责生成`output`张量和创建合适的layout

```python
def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x) # preallocate the output
    assert x.device == DEVICE and y.device == DEVICE and output.device == DEVICE
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    # We return a handle to z but, since `torch.cuda.synchronize()` hasn't been called, the kernel is still
    # running asynchronously at this point.
    return output
```

核心只有两行代码，

+ `grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )`定义一个grid匿名函数，用`triton.cdiv`计算$\lceil\frac{n}{BLOCK\_SIZE}\rceil$

+ `add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)`调用`grid(meta)`来确定网格大小并启动GPU线程块，并将`(x, y, output, n_elements, BLOCK_SIZE=1024)`传递给kernel

```python
torch.manual_seed(0)
size = 984320000
x = torch.rand(size, device=DEVICE)
y = torch.rand(size, device=DEVICE)
output_torch = x + y
output_triton = add(x, y)
print(output_torch)
print(output_triton)
print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')
```

### Benchmark

这个设计真的非常喜欢，不用费事去写一些benchmark测试代码了！

```python
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # Argument names to use as an x-axis for the plot. 用作绘图 x 轴的参数名称。
        x_vals=[2**i for i in range(12, 28, 1)],  # Different possible values for `x_name`. `x_name` 的不同可能值。
        x_log=True,  # x axis is logarithmic. x 轴为对数。
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot. 参数名称，其值对应于绘图中的不同线条。
        line_vals=['triton', 'torch'],  # Possible values for `line_arg`. `line_arg` 的可能值。
        line_names=['Triton', 'Torch'],  # Label name for the lines. 线条的标签名称。
        styles=[('blue', '-'), ('green', '-')],  # Line styles. 线条样式。
        ylabel='GB/s',  # Label name for the y-axis. y 轴标签名称。
        plot_name='vector-add-performance',  # Name for the plot. Used also as a file name for saving the plot. 绘图名称。也用作保存绘图的文件名。
        args={},  # Values for function arguments not in `x_names` and `y_name`. 不在 `x_names` 和 `y_name` 中的函数参数值。
    ))

def benchmark(size, provider):
    x = torch.rand(size, device='cuda', dtype=torch.float32)
    y = torch.rand(size, device='cuda', dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x, y), quantiles=quantiles)
    gbps = lambda ms: 3 * x.numel() * x.element_size() / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)

benchmark.run(print_data=True, show_plots=True, save_path='./')
```

以下是单卡Tesla V100-PCIE-32GB的benchmark结果，可以看出和Torch还是有一些差距的

```text
vector-add-performance:
           size      Triton       Torch
0        4096.0    9.600000    9.600000
1        8192.0   19.200000   19.200000
2       16384.0   38.400001   38.400001
3       32768.0   76.800002   76.800002
4       65536.0  127.999995  127.999995
5      131072.0  219.428568  219.428568
6      262144.0  341.333321  341.333321
7      524288.0  472.615390  474.898540
8     1048576.0  614.400016  614.400016
9     2097152.0  702.171410  702.171410
10    4194304.0  756.184613  756.184613
11    8388608.0  792.974002  793.173993
12   16777216.0  812.429770  815.800825
13   33554432.0  822.627612  824.352211
14   67108864.0  827.823144  828.695462
15  134217728.0  830.445624  831.344043
```

## Fused Softmax

TODO