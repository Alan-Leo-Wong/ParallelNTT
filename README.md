# CUDA-NTT

An accelerated NTT to compute the product of two polynomials, based CUDA and multi-threaded CPU. 

# Usage

The project uses CUDA by default, additional options are as follows:

- `-normal`: Force single-threaded NTT on the CPU.
- `-cpu`: Force multi-threaded NTT on the CPU instead of GPU.  
- `-n <number>`: Degree(log) of the first polynomial. 
- `-m <number>`: Degree(log) of the second polynomial.
- `-iter <number>`: The number of iterations to run the program. 

# Dependencies

The project does **NOT** support MSVC currently, and it has the following build dependencies:

- [NVIDIA CUDA 11.5 Toolkit (or higher)](https://developer.nvidia.com/cuda-toolkit) for CUDA.
- [OpenMP](https://www.openmp.org/) for CPU multi-threading.

# Build

```powershell
git clone https://github.com/KMushaL/ParallelNTT.git
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="your_cuda_compute_capability"
make -j number_of_cores
```

