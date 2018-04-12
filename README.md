# rocALUTION
rocALUTION is a library on top of AMD's Radeon Open Compute [ROCm][] runtime and toolchains that enables you to perform various sparse iterative solvers and preconditioners on multi/many-core CPU and GPU devices. Based on C++ and [HIP][], it provides a generic and flexible design that allows seamless integration with other scientific software packages.

## Quickstart rocALUTION build

#### CMake
All compiler specifications are determined automatically. The compilation process can be performed by
*   mkdir build
*   cd build
*   cmake .. -DSUPPORT_HIP=ON -DSUPPORT_OMP=ON -DSUPPORT_MPI=OFF
*   make

#### Simple test
You can test the installation by running a CG solver on a Laplace matrix:
*   wget ftp://math.nist.gov/pub/MatrixMarket2/Harwell−Boeing/laplace/gr3030.mtx.gz
*   gzip -d gr_30_30.mtx.gz
*   ./cg gr_30_30.mtx

[ROCm]: https://github.com/RadeonOpenCompute/ROCm
[HIP]: https://github.com/GPUOpen-ProfessionalCompute-Tools/HIP/
