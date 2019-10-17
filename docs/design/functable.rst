*******************
Functionality Table
*******************

The following tables give an overview whether a rocALUTION routine is implemented on host backend, accelerator backend, or both.

LocalMatrix and LocalVector classes
===================================
All matrix operations (except SpMV) require a CSR matrix.

.. note:: If the input matrix is not a CSR matrix, an internal conversion will be performed to CSR format, followed by a back conversion to the previous format after the operation.
          In this case, a warning message on verbosity level 2 will be printed.

======================== =============================================================================== ======== =======
**LocalMatrix function** **Comment**                                                                     **Host** **HIP**
======================== =============================================================================== ======== =======
`GetFormat`              Obtain the matrix format                                                        Yes      Yes
`Check`                  Check the matrix for structure and value validity                               Yes      No
`AllocateCSR`            Allocate CSR matrix                                                             Yes      Yes
`AllocateBCSR`           Allocate BCSR matrix                                                            Yes      Yes
`AllocateMCSR`           Allocate MCSR matrix                                                            Yes      Yes
`AllocateCOO`            Allocate COO matrix                                                             Yes      Yes
`AllocateDIA`            Allocate DIA matrix                                                             Yes      Yes
`AllocateELL`            Allocate ELL matrix                                                             Yes      Yes
`AllocateHYB`            Allocate HYB matrix                                                             Yes      Yes
`AllocateDENSE`          Allocate DENSE matrix                                                           Yes      Yes
`SetDataPtrCSR`          Initialize matrix with externally allocated CSR data                            Yes      Yes
`SetDataPtrMCSR`         Initialize matrix with externally allocated MCSR data                           Yes      Yes
`SetDataPtrCOO`          Initialize matrix with externally allocated COO data                            Yes      Yes
`SetDataPtrDIA`          Initialize matrix with externally allocated DIA data                            Yes      Yes
`SetDataPtrELL`          Initialize matrix with externally allocated ELL data                            Yes      Yes
`SetDataPtrDENSE`        Initialize matrix with externally allocated DENSE data                          Yes      Yes
`LeaveDataPtrCSR`        Direct Memory access                                                            Yes      Yes
`LeaveDataPtrMCSR`       Direct Memory access                                                            Yes      Yes
`LeaveDataPtrCOO`        Direct Memory access                                                            Yes      Yes
`LeaveDataPtrDIA`        Direct Memory access                                                            Yes      Yes
`LeaveDataPtrELL`        Direct Memory access                                                            Yes      Yes
`LeaveDataPtrDENSE`      Direct Memory access                                                            Yes      Yes
`Zeros`                  Set all matrix entries to zero                                                  Yes      Yes
`Scale`                  Scale all matrix non-zeros                                                      Yes      Yes
`ScaleDiagonal`          Scale matrix diagonal                                                           Yes      Yes
`ScaleOffDiagonal`       Scale matrix off-diagonal entries                                               Yes      Yes
`AddScalar`              Add scalar to all matrix non-zeros                                              Yes      Yes
`AddScalarDiagonal`      Add scalar to matrix diagonal                                                   Yes      Yes
`AddScalarOffDiagonal`   Add scalar to matrix off-diagonal entries                                       Yes      Yes
`ExtractSubMatrix`       Extract sub-matrix                                                              Yes      Yes
`ExtractSubMatrices`     Extract array of non-overlapping sub-matrices                                   Yes      Yes
`ExtractDiagonal`        Extract matrix diagonal                                                         Yes      Yes
`ExtractInverseDiagonal` Extract inverse matrix diagonal                                                 Yes      Yes
`ExtractL`               Extract lower triangular matrix                                                 Yes      Yes
`ExtractU`               Extract upper triangular matrix                                                 Yes      Yes
`Permute`                (Forward) permute the matrix                                                    Yes      Yes
`PermuteBackward`        (Backward) permute the matrix                                                   Yes      Yes
`CMK`                    Create CMK permutation vector                                                   Yes      No
`RCMK`                   Create reverse CMK permutation vector                                           Yes      No
`ConnectivityOrder`      Create connectivity (increasing nnz per row) permutation vector                 Yes      No
`MultiColoring`          Create multi-coloring decomposition of the matrix                               Yes      No
`MaximalIndependentSet`  Create maximal independent set decomposition of the matrix                      Yes      No
`ZeroBlockPermutation`   Create permutation where zero diagonal entries are mapped to the last block     Yes      No
`ILU0Factorize`          Create ILU(0) factorization                                                     Yes      No
`LUFactorize`            Create LU factorization                                                         Yes      No
`ILUTFactorize`          Create ILU(t,m) factorization                                                   Yes      No
`ILUpFactorize`          Create ILU(p) factorization                                                     Yes      No
`ICFactorize`            Create IC factorization                                                         Yes      No
`QRDecompose`            Create QR decomposition                                                         Yes      No
`ReadFileMTX`            Read matrix from matrix market file                                             Yes      No
`WriteFileMTX`           Write matrix to matrix market file                                              Yes      No
`ReadFileCSR`            Read matrix from binary file                                                    Yes      No
`WriteFileCSR`           Write matrix to binary file                                                     Yes      No
`CopyFrom`               Copy matrix (values and structure) from another LocalMatrix                     Yes      Yes
`CopyFromAsync`          Copy matrix asynchronously                                                      Yes      Yes
`CloneFrom`              Clone an entire matrix (values, structure and backend) from another LocalMatrix Yes      Yes
`UpdateValuesCSR`        Update CSR matrix values (structure remains identical)                          Yes      Yes
`CopyFromCSR`            Copy (import) CSR matrix                                                        Yes      Yes
`CopyToCSR`              Copy (export) CSR matrix                                                        Yes      Yes
`CopyFromCOO`            Copy (import) COO matrix                                                        Yes      Yes
`CopyToCOO`              Copy (export) COO matrix                                                        Yes      Yes
`CopyFromHostCSR`        Allocate and copy (import) a CSR matrix from host                               Yes      No
`ConvertToCSR`           Convert a matrix to CSR format                                                  Yes      No
`ConvertToMCSR`          Convert a matrix to MCSR format                                                 Yes      No
`ConvertToBCSR`          Convert a matrix to BCSR format                                                 Yes      No
`ConvertToCOO`           Convert a matrix to COO format                                                  Yes      Yes
`ConvertToELL`           Convert a matrix to ELL format                                                  Yes      Yes
`ConvertToDIA`           Convert a matrix to DIA format                                                  Yes      Yes
`ConvertToHYB`           Convert a matrix to HYB format                                                  Yes      Yes
`ConvertToDENSE`         Convert a matrix to DENSE format                                                Yes      No
`ConvertTo`              Convert a matrix                                                                Yes
`SymbolicPower`          Perform symbolic power computation (structure only)                             Yes      No
`MatrixAdd`              Matrix addition                                                                 Yes      No
`MatrixMult`             Multiply two matrices                                                           Yes      No
`DiagonalMatrixMult`     Multiply matrix with diagonal matrix (stored in LocalVector)                    Yes      Yes
`DiagonalMatrixMultL`    Multiply matrix with diagonal matrix (stored in LocalVector) from left          Yes      Yes
`DiagonalMatrixMultR`    Multiply matrix with diagonal matrix (stored in LocalVector) from right         Yes      Yes
`Gershgorin`             Compute the spectrum approximation with Gershgorin circles theorem              Yes      No
`Compress`               Delete all entries where `abs(a_ij) <= drop_off`                                Yes      Yes
`Transpose`              Transpose the matrix                                                            Yes      No
`Sort`                   Sort the matrix indices                                                         Yes      No
`Key`                    Compute a unique matrix key                                                     Yes      No
`ReplaceColumnVector`    Replace a column vector of a matrix                                             Yes      No
`ReplaceRowVector`       Replace a row vector of a matrix                                                Yes      No
`ExtractColumnVector`    Extract a column vector of a matrix                                             Yes      No
`ExtractRowVector`       Extract a row vector of a matrix                                                Yes      No
======================== =============================================================================== ======== =======

=========================== ===================================================================== ======== =======
**LocalVector function**    **Comment**                                                           **Host** **HIP**
=========================== ===================================================================== ======== =======
`GetSize`                   Obtain vector size                                                    Yes      Yes
`Check`                     Check vector for valid entries                                        Yes      No
`Allocate`                  Allocate vector                                                       Yes      Yes
`Sync`                      Synchronize                                                           Yes      Yes
`SetDataPtr`                Initialize vector with external data                                  Yes      Yes
`LeaveDataPtr`              Direct Memory Access                                                  Yes      Yes
`Zeros`                     Set vector entries to zero                                            Yes      Yes
`Ones`                      Set vector entries to one                                             Yes      Yes
`SetValues`                 Set vector entries to scalar                                          Yes      Yes
`SetRandomUniform`          Initialize vector with uniformly distributed random numbers           Yes      No
`SetRandomNorm`             Initialize vector with normally distributed random numbers            Yes      No
`ReadFileASCII`             Read vector for ASCII file                                            Yes      No
`WriteFileASCII`            Write vector to ASCII file                                            Yes      No
`ReadFileBinary`            Read vector from binary file                                          Yes      No
`WriteFileBinary`           Write vector to binary file                                           Yes      No
`CopyFrom`                  Copy vector (values) from another LocalVector                         Yes      Yes
`CopyFromAsync`             Copy vector asynchronously                                            Yes      Yes
`CopyFromFloat`             Copy vector from another LocalVector<float>                           Yes      Yes
`CopyFromDouble`            Copy vector from another LocalVector<double>                          Yes      Yes
`CopyFromPermute`           Copy vector under specified (forward) permutation                     Yes      Yes
`CopyFromPermuteBackward`   Copy vector under specified (backward) permutation                    Yes      Yes
`CloneFrom`                 Clone vector (values and backend descriptor) from another LocalVector Yes      Yes
`CopyFromData`              Copy (import) vector from array                                       Yes      Yes
`CopyToData`                Copy (export) vector to array                                         Yes      Yes
`Permute`                   (Foward) permute vector in-place                                      Yes      Yes
`PermuteBackward(Backward)` permute vector in-place                                               Yes      Yes
`AddScale`                  `y = a * x + y`                                                       Yes      Yes
`ScaleAdd`                  `y = x + a * y`                                                       Yes      Yes
`ScaleAddScale`             `y = b * x + a * y`                                                   Yes      Yes
`ScaleAdd2`                 `z = a * x + b * y + c * z`                                           Yes      Yes
`Scale`                     `x = a * x`                                                           Yes      Yes
`ExclusiveScan`             Compute exclusive sum                                                 Yes      No
`Dot`                       Compute dot product                                                   Yes      Yes
`DotNonConj`                Compute non-conjugated dot product                                    Yes      Yes
`Norm`                      Compute L2 norm                                                       Yes      Yes
`Reduce`                    Obtain the sum of all vector entries                                  Yes      Yes
`Asum`                      Obtain the absolute sum of all vector entries                         Yes      Yes
`Amax`                      Obtain the absolute maximum entry of the vector                       Yes      Yes
`PointWiseMult`             Perform point wise multiplication of two vectors                      Yes      Yes
`Power`                     Compute vector power                                                  Yes      Yes
=========================== ===================================================================== ======== =======

Solver and Preconditioner classes
=================================

.. note:: The building phase of the iterative solver also depends on the selected preconditioner.

================================ ================= ======== =======
**Solver**                       **Functionality** **Host** **HIP**
================================ ================= ======== =======
:cpp:class:`CG <rocalution::CG>` Building          Yes      Yes
:cpp:class:`CG <rocalution::CG>` Solving           Yes      Yes
`FCG`                            Building          Yes      Yes
`FCG`                            Solving           Yes      Yes
:cpp:class:`CR <rocalution::CR>` Building          Yes      Yes
:cpp:class:`CR <rocalution::CR>` Solving           Yes      Yes
`BiCGStab`                       Building          Yes      Yes
`BiCGStab`                       Solving           Yes      Yes
`BiCGStab(l)`                    Building          Yes      Yes
`BiCGStab(l)`                    Solving           Yes      Yes
`QMRCGStab`                      Building          Yes      Yes
`QMRCGStab`                      Solving           Yes      Yes
`GMRES`                          Building          Yes      Yes
`GMRES`                          Solving           Yes      Yes
`FGMRES`                         Building          Yes      Yes
`FGMRES`                         Solving           Yes      Yes
`Chebyshev`                      Building          Yes      Yes
`Chebyshev`                      Solving           Yes      Yes
`Mixed-Precision`                Building          Yes      Yes
`Mixed-Precision`                Solving           Yes      Yes
`Fixed-Point Iteration`          Building          Yes      Yes
`Fixed-Point Iteration`          Solving           Yes      Yes
`AMG (Plain Aggregation)`        Building          Yes      No
`AMG (Plain Aggregation)`        Solving           Yes      Yes
`AMG (Smoothed Aggregation)`     Building          Yes      No
`AMG (Smoothed Aggregation)`     Solving           Yes      Yes
`AMG (Ruge Stueben)`             Building          Yes      No
`AMG (Ruge Stueben)`             Solving           Yes      Yes
`AMG (Pairwise Aggregation)`     Building          Yes      No
`AMG (Pairwise Aggregation)`     Solving           Yes      Yes
:cpp:class:`LU <rocalution::LU>` Building          Yes      No
:cpp:class:`LU <rocalution::LU>` Solving           Yes      No
:cpp:class:`QR <rocalution::QR>` Building          Yes      No
:cpp:class:`QR <rocalution::QR>` Solving           Yes      No
`Inversion`                      Building          Yes      No
`Inversion`                      Solving           Yes      Yes
================================ ================= ======== =======

========================= ================= ======== =======
**Preconditioner**        **Functionality** **Host** **HIP**
========================= ================= ======== =======
`Jacobi`                  Building          Yes      Yes
`Jacobi`                  Solving           Yes      Yes
`BlockJacobi`             Building          Yes      Yes
`BlockJacobi`             Solving           Yes      Yes
`MultiColoredILU(0,1)`    Building          Yes      Yes
`MultiColoredILU(0,1)`    Solving           Yes      Yes
`MultiColoredILU(>0, >1)` Building          Yes      No
`MultiColoredILU(>0, >1)` Solving           Yes      Yes
`MultiElimination(I)LU`   Building          Yes      No
`MultiElimination(I)LU`   Solving           Yes      Yes
`ILU(0)`                  Building          Yes      Yes
`ILU(0)`                  Solving           Yes      Yes
`ILU(>0)`                 Building          Yes      No
`ILU(>0)`                 Solving           Yes      No
`ILUT`                    Building          Yes      No
`ILUT`                    Solving           Yes      No
`IC(0)`                   Building          Yes      No
`IC(0)`                   Solving           Yes      No
`FSAI`                    Building          Yes      No
`FSAI`                    Solving           Yes      Yes
`SPAI`                    Building          Yes      No
`SPAI`                    Solving           Yes      Yes
`Chebyshev`               Building          Yes      No
`Chebyshev`               Solving           Yes      Yes
`MultiColored(S)GS`       Building          Yes      No
`MultiColored(S)GS`       Solving           Yes      Yes
`(S)GS`                   Building          Yes      No
`(S)GS`                   Solving           Yes      No
`(R)AS`                   Building          Yes      Yes
`(R)AS`                   Solving           Yes      Yes
`BlockPreconditioner`     Building          Yes      Yes
`BlockPreconditioner`     Solving           Yes      Yes
`SaddlePoint`             Building          Yes      No
`SaddlePoint`             Solving           Yes      Yes
========================= ================= ======== =======
