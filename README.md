# David's LLM Infrastructure Journey

A learning repository focused on understanding and implementing high-performance compute kernels for Large Language Model (LLM) inference across different hardware platforms: NVIDIA (CUDA), AMD (ROCm/Triton), and Intel (oneAPI/Triton).

## 📚 Overview

This repository documents the journey of learning LLM infrastructure optimization, specifically:

- **CUDA Kernels**: Writing custom GPU kernels for NVIDIA hardware
- **Triton Kernels**: Utilizing OpenAI's Triton for portable, high-performance kernels on AMD and Intel GPUs
- **vLLM Integration**: Understanding and optimizing vLLM for multi-vendor GPU support

## 🎯 Learning Objectives

### 1. CUDA Kernel Development (NVIDIA)

CUDA (Compute Unified Device Architecture) is NVIDIA's parallel computing platform and programming model.

**Key Concepts:**
- Understanding GPU architecture (SMs, warps, threads)
- Memory hierarchy (global, shared, register)
- Kernel launch configuration and occupancy
- Common optimization patterns (tiling, coalescing, memory alignment)

**Resources:**
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [NVIDIA Deep Learning Examples](https://github.com/NVIDIA/DeepLearningExamples)

**Common CUDA Kernel Operations for LLMs:**
- Matrix multiplication (GEMM)
- Attention mechanisms (FlashAttention)
- Layer normalization
- Activation functions (GELU, SiLU)
- Quantization/dequantization kernels

### 2. Triton for AMD GPUs (ROCm)

Triton is a Python-based language that enables writing portable GPU kernels without vendor-specific code.

**AMD ROCm Platform:**
- ROCm is AMD's open-source GPU computing platform
- Triton can target AMD GPUs through the ROCm backend
- Compatible with AMD Instinct series (MI100, MI200, MI300)

**Key Advantages:**
- Write once, run on multiple GPU vendors
- Python-like syntax for easier development
- Automatic optimization and tuning
- JIT compilation for optimal performance

**Resources:**
- [OpenAI Triton Documentation](https://triton-lang.org/)
- [AMD ROCm Documentation](https://rocm.docs.amd.com/)
- [Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)
- [ROCm/Triton Integration Guide](https://github.com/ROCm/triton)

**Example Triton Operations:**
```python
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
```

### 3. Triton for Intel GPUs (oneAPI)

Intel's Data Center GPU Max series (formerly Ponte Vecchio) and Arc GPUs support Triton through Intel's oneAPI.

**Intel oneAPI Platform:**
- Unified programming model across Intel architectures
- Support for Intel Data Center GPU Max (1100, 1550)
- Triton integration through Intel Extension for PyTorch

**Key Features:**
- XMX (Xe Matrix Extensions) for accelerated matrix operations
- Unified shared memory across CPU and GPU
- SYCL-based backend for maximum performance

**Resources:**
- [Intel oneAPI Documentation](https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html)
- [Intel Extension for PyTorch](https://github.com/intel/intel-extension-for-pytorch)
- [Triton on Intel GPUs](https://github.com/intel/intel-xpu-backend-for-triton)
- [Intel Data Center GPU Max Documentation](https://www.intel.com/content/www/us/en/developer/articles/technical/intel-data-center-gpu-max-series-overview.html)

**Hardware Support:**
- Intel Data Center GPU Max 1100
- Intel Data Center GPU Max 1550
- Intel Arc A-series (limited support)

### 4. vLLM - High-Performance LLM Serving

vLLM is an efficient and easy-to-use library for LLM inference and serving.

**Key Features:**
- PagedAttention for efficient memory management
- Continuous batching for high throughput
- Fast model execution with CUDA/Triton kernels
- Multi-GPU and tensor parallelism support
- Quantization support (AWQ, GPTQ, SqueezeLLM)

**Multi-Vendor Support:**
- Primary: NVIDIA GPUs (CUDA)
- Experimental: AMD GPUs (ROCm)
- Emerging: Intel GPUs (oneAPI)

**Resources:**
- [vLLM GitHub Repository](https://github.com/vllm-project/vllm)
- [vLLM Documentation](https://docs.vllm.ai/)
- [PagedAttention Paper](https://arxiv.org/abs/2309.06180)

## 🚀 Getting Started

### Prerequisites

**For NVIDIA GPUs:**
```bash
# Install CUDA Toolkit (12.1+)
# Download from: https://developer.nvidia.com/cuda-downloads

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**For AMD GPUs:**
```bash
# Install ROCm (5.7+)
# Follow: https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html

# Install PyTorch with ROCm support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
```

**For Intel GPUs:**
```bash
# Install Intel oneAPI Base Toolkit
# Download from: https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html

# Install Intel Extension for PyTorch
pip install intel-extension-for-pytorch
pip install oneccl_bind_pt --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
```

**Install Triton:**
```bash
pip install triton
```

**Install vLLM:**
```bash
# For NVIDIA
pip install vllm

# For AMD ROCm
pip install vllm  # Use ROCm-compatible wheel

# For Intel (experimental)
# Follow Intel-specific installation instructions
```

## 📖 Learning Path

### Beginner
1. Understand GPU architecture basics
2. Write simple CUDA kernels (vector addition, matrix operations)
3. Learn Triton syntax with basic examples
4. Run simple vLLM inference examples

### Intermediate
1. Implement optimized kernels (fused operations, shared memory usage)
2. Port CUDA kernels to Triton
3. Profile and benchmark kernel performance
4. Understand vLLM's PagedAttention mechanism

### Advanced
1. Implement custom attention mechanisms (FlashAttention, PagedAttention)
2. Optimize kernels for specific hardware (AMD MI300, Intel Max 1550)
3. Contribute to vLLM with multi-vendor kernel implementations
4. Benchmark and tune for production workloads

## 🛠️ Development Tools

### Profiling and Debugging
- **NVIDIA**: `nsys`, `ncu`, `nvprof`
- **AMD**: `rocprof`, `rocgdb`, `omniperf`
- **Intel**: `VTune Profiler`, `Advisor`
- **Triton**: Built-in profiler and autotuner

### Performance Analysis
```python
# Triton autotuner example
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(...):
    ...
```

## 📊 Hardware Comparison

| Feature | NVIDIA (CUDA) | AMD (ROCm) | Intel (oneAPI) |
|---------|---------------|------------|----------------|
| Maturity | Excellent | Good | Emerging |
| Triton Support | Native | Good | Experimental |
| vLLM Support | Full | Experimental | Limited |
| Ecosystem | Largest | Growing | Developing |
| Documentation | Extensive | Good | Improving |

## 🔗 Additional Resources

### Papers
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [FlashAttention](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2](https://arxiv.org/abs/2307.08691)
- [Efficient Memory Management for LLM Serving (PagedAttention)](https://arxiv.org/abs/2309.06180)

### Communities
- [CUDA Programming Discord](https://discord.gg/nvidia)
- [ROCm Developer Community](https://community.amd.com/)
- [Intel oneAPI Forums](https://community.intel.com/t5/oneAPI/ct-p/oneapi)
- [vLLM Discord](https://discord.gg/vllm)

### Courses
- [NVIDIA DLI: Fundamentals of Accelerated Computing with CUDA](https://www.nvidia.com/en-us/training/)
- [AMD ROCm Learning Resources](https://www.amd.com/en/graphics/servers-solutions-rocm-ml)
- [Intel GPU Programming Guide](https://www.intel.com/content/www/us/en/docs/oneapi/programming-guide/current/intel-gpu-programming-guide.html)

## 🤝 Contributing

This is a personal learning repository, but contributions, corrections, and suggestions are welcome!

## 📝 License

This repository is for educational purposes. Individual resources and examples may have their own licenses.

## 🙏 Acknowledgments

- OpenAI for Triton
- vLLM team for the excellent inference framework
- NVIDIA, AMD, and Intel for their GPU platforms and documentation
- The broader GPU computing community
