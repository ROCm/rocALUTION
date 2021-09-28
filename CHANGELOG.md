# Change Log for rocALUTION

Full documentation for rocALUTION is available at [rocalution.readthedocs.io](https://rocalution.readthedocs.io/en/latest/).

## (Unreleased) rocALUTION 1.13.0
### Added
- Enabling beta support for Windows 10
### Changed
- Packaging split into a runtime package called rocalution and a development package called rocalution-devel. The development package depends on runtime. The runtime package suggests the development package for all supported OSes except CentOS 7 to aid in the transition. The suggests feature in packaging is introduced as a deprecated feature and will be removed in a future rocm release.
### Improved
- (A)MG solving phase optimization

## (Unreleased) rocALUTION 1.12.3
### Added
- Address sanitizer build option added
### Changed
- Deprecated GlobalPairwiseAMG, please use PairwiseAMG instead. GlobalPairwiseAMG will be removed in a future major release.
### Improved
- (A)MG smoothing and convergence rate improvement
- Improved IDR Gram-Schmidt process

## [rocALUTION 1.12.1 for ROCm 4.3.0]
### Added
- support for gfx90a target
- support for gfx1030 target
### Improved
- install script
### Known Issues
- none

## [rocALUTION 1.11.5 for ROCm 4.2.0]
### Added
- none
### Known Issues
- none

## [rocALUTION 1.11.5 for ROCm 4.1.0]
### Added
- none
### Known Issues
- none

## [rocALUTION 1.11.5 for ROCm 4.0.0]
### Added
- Add changelog
- Fixing NaN issues
- update to debian package name
- bcsr format support.
- cmake files adjustments.

## [rocALUTION 1.10 for ROCm 3.9]
### Added
- rocRAND to support GPU sampling of random data.
### Known Issues
- none

## [rocALUTION 1.9.3 for ROCm 3.8]
### Added
- csr2dense and dense2csr to HIP backend.
### Known Issues
- none

## [rocALUTION 1.9.1 for ROCm 3.7]
### Added
- none
### Known Issues
- none

## [rocALUTION 1.9.1 for ROCm 3.6]
### Added
- none
### Known Issues
- none

## [rocALUTION 1.9.1 for ROCm 3.5]
### Added
- static build
- BCSR matrix format for SpMV
- Bug fixing in conversion from CSR to HYB format.
### Known Issues
- none
