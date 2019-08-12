# rocALUTION
rocALUTION is a sparse linear algebra library with focus on exploring fine-grained parallelism on top of AMD's Radeon Open Compute [ROCm][] runtime and toolchains, targeting modern CPU and GPU platforms. Based on C++ and [HIP][], it provides a portable, generic and flexible design that allows seamless integration with other scientific software packages.

## Documentation
The latest rocALUTION documentation and API description can be found [here][].

## Quickstart rocALUTION build

#### CMake 3.5 or later
All compiler specifications are determined automatically. The compilation process can be performed by
```
# Clone rocALUTION using git
git clone https://github.com/ROCmSoftwarePlatform/rocALUTION.git

# Go to rocALUTION directory, create and change to build directory
cd rocALUTION; mkdir build; cd build

# Configure rocALUTION
# Build options:
#   SUPPORT_HIP    - build rocALUTION with HIP support (ON)
#   SUPPORT_OMP    - build rocALUTION with OpenMP support (ON)
#   SUPPORT_MPI    - build rocALUTION with MPI (multi-node) support (OFF)
#   BUILD_SHARED   - build rocALUTION as shared library (ON, recommended)
#   BUILD_EXAMPLES - build rocALUTION examples (ON)
cmake .. -DSUPPORT_HIP=ON

# Build
make
```

#### Simple test
You can test the installation by running a CG solver on a Laplace matrix:
```
cd rocALUTION; cd build
wget ftp://math.nist.gov/pub/MatrixMarket2/Harwell-Boeing/laplace/gr_30_30.mtx.gz
gzip -d gr_30_30.mtx.gz
./clients/staging/cg gr_30_30.mtx
```

## Overview

#### Backends
rocALUTION offers various backends for different (parallel) hardware.
*  Host
*  [OpenMP][] - designed for multi-core CPUs
*  [HIP][]    - designed for ROCm compatible devices
*  [MPI][]    - designed for multi-node clusters and multi-GPU setups

#### Easy to use
Syntax and structure of the library provide fast learning curves. With the help of the examples, anyone can try out the library - no knowledge in multi-core or GPU programming is required.

#### Requirements
There are no hardware requirements to install and run rocALUTION. If GPU devices are available, rocALUTION will use them.
In order to use rocALUTION on GPU devices, you will need to make sure that [rocBLAS][] and [rocSPARSE][] libraries are installed on your system. You can install them from ROCm repository, from github releases tab or manually compile them yourself.

#### Iterative solvers
*  Fixed-Point iteration schemes - Jacobi, (Symmetric) Gauss-Seidel, SOR, SSOR
*  Krylov subspace methods - CR, CG, BiCGStab, BiCGStab(*l*), GMRES, IDR, QMRCGSTAB, Flexible CG/GMRES
*  Mixed-precision defect-correction scheme
*  Chebyshev iteration scheme
*  Multigrid - geometric and algebraic

#### Preconditioners
*  Matrix splitting schemes - Jacobi, (multi-colored) (symmetric) Gauss-Seidel, SOR, SSOR
*  Factorization schemes    - ILU(*0*), ILU(*p*) (based on levels), ILU(*p,q*) (power(*q*)-pattern method), multi-elimination ILU (nested/recursive), ILUT (based on threshold), IC(*0*)
*  Approximate Inverses - Chebyshev matrix-valued polynomial, SPAI, FSAI, TNS
*  Diagonal-based preconditioner for Saddle-point problems
*  Block-type of sub-preconditioners/solvers
*  (Restricted) Additive Schwarz
*  Variable type of preconditioners

#### Sparse matrix formats
*  Compressed Sparse Row (CSR)
*  Modified Compressed Sparse Row (MCSR)
*  Dense (DENSE)
*  Coordinate (COO)
*  ELL
*  Diagonal (DIA)
*  Hybrid ELL+COO (HYB)

#### Generic and robust design
rocALUTION is based on a generic and robust design, allowing expansion in the direction of new solvers and preconditioners and support for various hardware types. Furthermore, the design of the library allows the use of all solvers as preconditioners in other solvers, for example you can define a CG solver with a multi-elimination preconditioner, where the last-block is preconditioned with another Chebyshev iteration method which is preconditioned with a multi-colored symmetric Gauss-Seidel scheme.

#### Portable code and results
All code based on rocALUTION is portable and independent of the hardware, it will compile and run on any supported platform. All solvers and preconditioners are based on a single source code implementation, which delivers portable results across all backends (variations are possible due to different rounding modes on the hardware). The only difference which you can see for a hardware change is the performance variation.



[ROCm]: https://github.com/RadeonOpenCompute/ROCm
[HIP]: https://github.com/GPUOpen-ProfessionalCompute-Tools/HIP/
[OpenMP]: http://www.openmp.org/
[MPI]: https://www.open-mpi.org/
[rocBLAS]: https://github.com/ROCmSoftwarePlatform/rocBLAS
[rocSPARSE]: https://github.com/ROCmSoftwarePlatform/rocSPARSE
[here]: https://rocalution.readthedocs.io
