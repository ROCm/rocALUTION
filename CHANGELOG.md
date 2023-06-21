# Change Log for rocALUTION

Full documentation for rocALUTION is available at [rocalution.readthedocs.io](https://rocalution.readthedocs.io/en/latest/).

## rocALUTION 2.1.11 for ROCm 5.7.0
### Added
- Added support for gfx940, gfx941 and gfx942
### Improved
- Fixed OpenMP runtime issue with Windows toolchain

## rocALUTION 2.1.9 for ROCm 5.6.0
### Improved
- Fixed synchronization issues in level 1 routines

## rocALUTION 2.1.8 for ROCm 5.5.0
### Added
- Added build support for Navi32
### Improved
- Fixed a typo in MPI backend
- Fixed a bug with the backend when HIP support is disabled
- Fixed a bug in SAAMG hierarchy building on HIP backend
- Improved SAAMG hierarchy build performance on HIP backend
### Changed
- LocalVector::GetIndexValues(ValueType\*) is deprecated, use LocalVector::GetIndexValues(const LocalVector&, LocalVector\*) instead
- LocalVector::SetIndexValues(const ValueType\*) is deprecated, use LocalVector::SetIndexValues(const LocalVector&, const LocalVector&) instead
- LocalMatrix::RSDirectInterpolation(const LocalVector&, const LocalVector&, LocalMatrix\*, LocalMatrix\*) is deprecated, use LocalMatrix::RSDirectInterpolation(const LocalVector&, const LocalVector&, LocalMatrix\*) instead
- LocalMatrix::RSExtPIInterpolation(const LocalVector&, const LocalVector&, bool, float, LocalMatrix\*, LocalMatrix\*) is deprecated, use LocalMatrix::RSExtPIInterpolation(const LocalVector&, const LocalVector&, bool, LocalMatrix\*) instead
- LocalMatrix::RugeStueben() is deprecated
- LocalMatrix::AMGSmoothedAggregation(ValueType, const LocalVector&, const LocalVector&, LocalMatrix\*, LocalMatrix\*, int) is deprecated, use LocalMatrix::AMGAggregation(ValueType, const LocalVector&, const LocalVector&, LocalMatrix\*, int) instead
- LocalMatrix::AMGAggregation(const LocalVector&, LocalMatrix\*, LocalMatrix\*) is deprecated, use LocalMatrix::AMGAggregation(const LocalVector&, LocalMatrix\*) instead

## rocALUTION 2.1.3 for ROCm 5.4.0
### Added
- Added build support for Navi31 and Navi33
- Added support for non-squared global matrices
### Improved
- Fixed a memory leak in MatrixMult on HIP backend
- Global structures can now be used with a single process
### Changed
- Switched GTest death test style to 'threadsafe'
- GlobalVector::GetGhostSize() is deprecated and will be removed
- ParallelManager::GetGlobalSize(), ParallelManager::GetLocalSize(), ParallelManager::SetGlobalSize() and ParallelManager::SetLocalSize() are deprecated and will be removed
- Vector::GetGhostSize() is deprecated and will be removed
- Multigrid::SetOperatorFormat(unsigned int) is deprecated and will be removed, use Multigrid::SetOperatorFormat(unsigned int, int) instead
- RugeStuebenAMG::SetCouplingStrength(ValueType) is deprecated and will be removed, use SetStrengthThreshold(float) instead

## rocALUTION 2.1.0 for ROCm 5.3.0
### Added
- Benchmarking tool
- Ext+I Interpolation with sparsify strategies added for RS-AMG
### Improved
- ParallelManager

## rocALUTION 2.0.3 for ROCm 5.2.0
### Added
- Packages for test and benchmark executables on all supported OSes using CPack.

## rocALUTION 2.0.2 for ROCm 5.1.0
### Added
- Added out-of-place matrix transpose functionality
- Added LocalVector<bool>

## rocALUTION 2.0.1 for ROCm 5.0.0
### Changed
- Removed deprecated GlobalPairwiseAMG class, please use PairwiseAMG instead.
- Changed to C++ 14 Standard
### Improved
- Added sanitizer option
- Improved documentation

## rocALUTION 1.13.2 for ROCm 4.5.0
### Added
- Address sanitizer build option added
- Enabling beta support for Windows 10
### Changed
- Deprecated GlobalPairwiseAMG, please use PairwiseAMG instead. GlobalPairwiseAMG will be removed in a future major release.
- Packaging split into a runtime package called rocalution and a development package called rocalution-devel. The development package depends on runtime. The runtime package suggests the development package for all supported OSes except CentOS 7 to aid in the transition. The suggests feature in packaging is introduced as a deprecated feature and will be removed in a future rocm release.
### Improved
- (A)MG smoothing and convergence rate improvement
- Improved IDR Gram-Schmidt process
- (A)MG solving phase optimization

## rocALUTION 1.12.1 for ROCm 4.3.0
### Added
- support for gfx90a target
- support for gfx1030 target
### Improved
- install script
### Known Issues
- none

## rocALUTION 1.11.5 for ROCm 4.2.0
### Added
- none
### Known Issues
- none

## rocALUTION 1.11.5 for ROCm 4.1.0
### Added
- none
### Known Issues
- none

## rocALUTION 1.11.5 for ROCm 4.0.0
### Added
- Add changelog
- Fixing NaN issues
- update to debian package name
- bcsr format support.
- cmake files adjustments.

## rocALUTION 1.10 for ROCm 3.9
### Added
- rocRAND to support GPU sampling of random data.
### Known Issues
- none

## rocALUTION 1.9.3 for ROCm 3.8
### Added
- csr2dense and dense2csr to HIP backend.
### Known Issues
- none

## rocALUTION 1.9.1 for ROCm 3.7
### Added
- none
### Known Issues
- none

## rocALUTION 1.9.1 for ROCm 3.6
### Added
- none
### Known Issues
- none

## rocALUTION 1.9.1 for ROCm 3.5
### Added
- static build
- BCSR matrix format for SpMV
- Bug fixing in conversion from CSR to HYB format.
### Known Issues
- none
