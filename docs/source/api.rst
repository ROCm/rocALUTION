.. toctree::
   :maxdepth: 4 
   :caption: Contents:

===
API
===

This section provides details of the library API

Host Utility Functions
**********************
.. doxygenfunction:: rocalution::allocate_host
.. doxygenfunction:: rocalution::free_host
.. doxygenfunction:: rocalution::set_to_zero_host
.. doxygenfunction:: rocalution::rocalution_time

Backend Manager
***************
.. doxygenfunction:: rocalution::init_rocalution
.. doxygenfunction:: rocalution::stop_rocalution
.. doxygenfunction:: rocalution::set_device_rocalution
.. doxygenfunction:: rocalution::set_omp_threads_rocalution
.. doxygenfunction:: rocalution::set_omp_affinity_rocalution
.. doxygenfunction:: rocalution::set_omp_threshold_rocalution
.. doxygenfunction:: rocalution::info_rocalution(void)
.. doxygenfunction:: rocalution::info_rocalution(const struct Rocalution_Backend_Descriptor)
.. doxygenfunction:: rocalution::disable_accelerator_rocalution
.. doxygenfunction:: rocalution::_rocalution_sync

Base Rocalution
***************
.. doxygenclass:: rocalution::BaseRocalution
.. doxygenfunction:: rocalution::BaseRocalution::MoveToAccelerator
.. doxygenfunction:: rocalution::BaseRocalution::MoveToHost
.. doxygenfunction:: rocalution::BaseRocalution::MoveToAcceleratorAsync
.. doxygenfunction:: rocalution::BaseRocalution::MoveToHostAsync
.. doxygenfunction:: rocalution::BaseRocalution::Sync
.. doxygenfunction:: rocalution::BaseRocalution::CloneBackend(const BaseRocalution<ValueType>&)
.. doxygenfunction:: rocalution::BaseRocalution::Info
.. doxygenfunction:: rocalution::BaseRocalution::Clear

Operator
********
.. doxygenclass:: rocalution::Operator
.. doxygenfunction:: rocalution::Operator::GetM
.. doxygenfunction:: rocalution::Operator::GetN
.. doxygenfunction:: rocalution::Operator::GetNnz
.. doxygenfunction:: rocalution::Operator::GetLocalM
.. doxygenfunction:: rocalution::Operator::GetLocalN
.. doxygenfunction:: rocalution::Operator::GetLocalNnz
.. doxygenfunction:: rocalution::Operator::GetGhostM
.. doxygenfunction:: rocalution::Operator::GetGhostN
.. doxygenfunction:: rocalution::Operator::GetGhostNnz
.. doxygenfunction:: rocalution::Operator::Apply(const LocalVector<ValueType>&, LocalVector<ValueType> *) const
.. doxygenfunction:: rocalution::Operator::ApplyAdd(const LocalVector<ValueType>&, ValueType, LocalVector<ValueType> *) const
.. doxygenfunction:: rocalution::Operator::Apply(const GlobalVector<ValueType>&, GlobalVector<ValueType> *) const
.. doxygenfunction:: rocalution::Operator::ApplyAdd(const GlobalVector<ValueType>&, ValueType, GlobalVector<ValueType> *) const

Vector
******
.. doxygenclass:: rocalution::Vector
.. doxygenfunction:: rocalution::Vector::GetSize
.. doxygenfunction:: rocalution::Vector::GetLocalSize
.. doxygenfunction:: rocalution::Vector::GetGhostSize
.. doxygenfunction:: rocalution::Vector::Check
.. doxygenfunction:: rocalution::Vector::Zeros
.. doxygenfunction:: rocalution::Vector::Ones
.. doxygenfunction:: rocalution::Vector::SetValues
.. doxygenfunction:: rocalution::Vector::SetRandomUniform
.. doxygenfunction:: rocalution::Vector::SetRandomNormal
.. doxygenfunction:: rocalution::Vector::ReadFileASCII
.. doxygenfunction:: rocalution::Vector::WriteFileASCII
.. doxygenfunction:: rocalution::Vector::ReadFileBinary
.. doxygenfunction:: rocalution::Vector::WriteFileBinary
.. doxygenfunction:: rocalution::Vector::CopyFrom(const LocalVector<ValueType>&)
.. doxygenfunction:: rocalution::Vector::CopyFrom(const GlobalVector<ValueType>&)
.. doxygenfunction:: rocalution::Vector::CopyFromAsync
.. doxygenfunction:: rocalution::Vector::CopyFromFloat
.. doxygenfunction:: rocalution::Vector::CopyFromDouble
.. doxygenfunction:: rocalution::Vector::CopyFrom(const LocalVector<ValueType>&, int, int, int)
.. doxygenfunction:: rocalution::Vector::CloneFrom(const LocalVector<ValueType>&)
.. doxygenfunction:: rocalution::Vector::CloneFrom(const GlobalVector<ValueType>&)
.. doxygenfunction:: rocalution::Vector::AddScale(const LocalVector<ValueType>&, ValueType)
.. doxygenfunction:: rocalution::Vector::AddScale(const GlobalVector<ValueType>&, ValueType)
.. doxygenfunction:: rocalution::Vector::ScaleAdd(ValueType, const LocalVector<ValueType>&)
.. doxygenfunction:: rocalution::Vector::ScaleAdd(ValueType, const GlobalVector<ValueType>&)
.. doxygenfunction:: rocalution::Vector::ScaleAddScale(ValueType, const LocalVector<ValueType>&, ValueType)
.. doxygenfunction:: rocalution::Vector::ScaleAddScale(ValueType, const GlobalVector<ValueType>&, ValueType)
.. doxygenfunction:: rocalution::Vector::ScaleAddScale(ValueType, const LocalVector<ValueType>&, ValueType, int, int, int)
.. doxygenfunction:: rocalution::Vector::ScaleAddScale(ValueType, const GlobalVector<ValueType>&, ValueType, int, int, int)
.. doxygenfunction:: rocalution::Vector::ScaleAdd2(ValueType, const LocalVector<ValueType>&, ValueType, const LocalVector<ValueType>&, ValueType)
.. doxygenfunction:: rocalution::Vector::ScaleAdd2(ValueType, const GlobalVector<ValueType>&, ValueType, const GlobalVector<ValueType>&, ValueType)
.. doxygenfunction:: rocalution::Vector::Scale
.. doxygenfunction:: rocalution::Vector::Dot(const LocalVector<ValueType>&) const
.. doxygenfunction:: rocalution::Vector::Dot(const GlobalVector<ValueType>&) const
.. doxygenfunction:: rocalution::Vector::DotNonConj(const LocalVector<ValueType>&) const
.. doxygenfunction:: rocalution::Vector::DotNonConj(const GlobalVector<ValueType>&) const
.. doxygenfunction:: rocalution::Vector::Norm
.. doxygenfunction:: rocalution::Vector::Reduce
.. doxygenfunction:: rocalution::Vector::Asum
.. doxygenfunction:: rocalution::Vector::Amax
.. doxygenfunction:: rocalution::Vector::PointWiseMult(const LocalVector<ValueType>&)
.. doxygenfunction:: rocalution::Vector::PointWiseMult(const GlobalVector<ValueType>&)
.. doxygenfunction:: rocalution::Vector::PointWiseMult(const LocalVector<ValueType>&, const LocalVector<ValueType>&)
.. doxygenfunction:: rocalution::Vector::PointWiseMult(const GlobalVector<ValueType>&, const GlobalVector<ValueType>&)
.. doxygenfunction:: rocalution::Vector::Power

Local Matrix
************
.. doxygenclass:: rocalution::LocalMatrix
.. doxygenfunction:: rocalution::LocalMatrix::GetFormat
.. doxygenfunction:: rocalution::LocalMatrix::Check
.. doxygenfunction:: rocalution::LocalMatrix::AllocateCSR
.. doxygenfunction:: rocalution::LocalMatrix::AllocateBCSR
.. doxygenfunction:: rocalution::LocalMatrix::AllocateMCSR
.. doxygenfunction:: rocalution::LocalMatrix::AllocateCOO
.. doxygenfunction:: rocalution::LocalMatrix::AllocateDIA
.. doxygenfunction:: rocalution::LocalMatrix::AllocateELL
.. doxygenfunction:: rocalution::LocalMatrix::AllocateHYB
.. doxygenfunction:: rocalution::LocalMatrix::AllocateDENSE
.. doxygenfunction:: rocalution::LocalMatrix::SetDataPtrCOO
.. doxygenfunction:: rocalution::LocalMatrix::SetDataPtrCSR
.. doxygenfunction:: rocalution::LocalMatrix::SetDataPtrMCSR
.. doxygenfunction:: rocalution::LocalMatrix::SetDataPtrELL
.. doxygenfunction:: rocalution::LocalMatrix::SetDataPtrDIA
.. doxygenfunction:: rocalution::LocalMatrix::SetDataPtrDENSE
.. doxygenfunction:: rocalution::LocalMatrix::LeaveDataPtrCOO
.. doxygenfunction:: rocalution::LocalMatrix::LeaveDataPtrCSR
.. doxygenfunction:: rocalution::LocalMatrix::LeaveDataPtrMCSR
.. doxygenfunction:: rocalution::LocalMatrix::LeaveDataPtrELL
.. doxygenfunction:: rocalution::LocalMatrix::LeaveDataPtrDIA
.. doxygenfunction:: rocalution::LocalMatrix::LeaveDataPtrDENSE
.. doxygenfunction:: rocalution::LocalMatrix::Zeros
.. doxygenfunction:: rocalution::LocalMatrix::Scale
.. doxygenfunction:: rocalution::LocalMatrix::ScaleDiagonal
.. doxygenfunction:: rocalution::LocalMatrix::ScaleOffDiagonal
.. doxygenfunction:: rocalution::LocalMatrix::AddScalar
.. doxygenfunction:: rocalution::LocalMatrix::AddScalarDiagonal
.. doxygenfunction:: rocalution::LocalMatrix::AddScalarOffDiagonal
.. doxygenfunction:: rocalution::LocalMatrix::ExtractSubMatrix
.. doxygenfunction:: rocalution::LocalMatrix::ExtractSubMatrices
.. doxygenfunction:: rocalution::LocalMatrix::ExtractDiagonal
.. doxygenfunction:: rocalution::LocalMatrix::ExtractInverseDiagonal
.. doxygenfunction:: rocalution::LocalMatrix::ExtractU
.. doxygenfunction:: rocalution::LocalMatrix::ExtractL
.. doxygenfunction:: rocalution::LocalMatrix::Permute
.. doxygenfunction:: rocalution::LocalMatrix::PermuteBackward
.. doxygenfunction:: rocalution::LocalMatrix::CMK
.. doxygenfunction:: rocalution::LocalMatrix::RCMK
.. doxygenfunction:: rocalution::LocalMatrix::ConnectivityOrder
.. doxygenfunction:: rocalution::LocalMatrix::MultiColoring
.. doxygenfunction:: rocalution::LocalMatrix::MaximalIndependentSet
.. doxygenfunction:: rocalution::LocalMatrix::ZeroBlockPermutation
.. doxygenfunction:: rocalution::LocalMatrix::ILU0Factorize
.. doxygenfunction:: rocalution::LocalMatrix::LUFactorize
.. doxygenfunction:: rocalution::LocalMatrix::ILUTFactorize
.. doxygenfunction:: rocalution::LocalMatrix::ILUpFactorize
.. doxygenfunction:: rocalution::LocalMatrix::LUAnalyse
.. doxygenfunction:: rocalution::LocalMatrix::LUAnalyseClear
.. doxygenfunction:: rocalution::LocalMatrix::LUSolve
.. doxygenfunction:: rocalution::LocalMatrix::ICFactorize
.. doxygenfunction:: rocalution::LocalMatrix::LLAnalyse
.. doxygenfunction:: rocalution::LocalMatrix::LLAnalyseClear
.. doxygenfunction:: rocalution::LocalMatrix::LLSolve(const LocalVector<ValueType>&, LocalVector<ValueType> *) const
.. doxygenfunction:: rocalution::LocalMatrix::LLSolve(const LocalVector<ValueType>&, const LocalVector<ValueType>&, LocalVector<ValueType> *) const
.. doxygenfunction:: rocalution::LocalMatrix::LAnalyse
.. doxygenfunction:: rocalution::LocalMatrix::LAnalyseClear
.. doxygenfunction:: rocalution::LocalMatrix::LSolve
.. doxygenfunction:: rocalution::LocalMatrix::UAnalyse
.. doxygenfunction:: rocalution::LocalMatrix::UAnalyseClear
.. doxygenfunction:: rocalution::LocalMatrix::USolve
.. doxygenfunction:: rocalution::LocalMatrix::Householder
.. doxygenfunction:: rocalution::LocalMatrix::QRDecompose
.. doxygenfunction:: rocalution::LocalMatrix::QRSolve
.. doxygenfunction:: rocalution::LocalMatrix::Invert
.. doxygenfunction:: rocalution::LocalMatrix::ReadFileMTX
.. doxygenfunction:: rocalution::LocalMatrix::WriteFileMTX
.. doxygenfunction:: rocalution::LocalMatrix::ReadFileCSR
.. doxygenfunction:: rocalution::LocalMatrix::WriteFileCSR
.. doxygenfunction:: rocalution::LocalMatrix::CopyFrom
.. doxygenfunction:: rocalution::LocalMatrix::CopyFromAsync
.. doxygenfunction:: rocalution::LocalMatrix::CloneFrom
.. doxygenfunction:: rocalution::LocalMatrix::UpdateValuesCSR
.. doxygenfunction:: rocalution::LocalMatrix::CopyFromCSR
.. doxygenfunction:: rocalution::LocalMatrix::CopyToCSR
.. doxygenfunction:: rocalution::LocalMatrix::CopyFromCOO
.. doxygenfunction:: rocalution::LocalMatrix::CopyToCOO
.. doxygenfunction:: rocalution::LocalMatrix::CopyFromHostCSR
.. doxygenfunction:: rocalution::LocalMatrix::CreateFromMap(const LocalVector<int>&, int, int)
.. doxygenfunction:: rocalution::LocalMatrix::CreateFromMap(const LocalVector<int>&, int, int, LocalMatrix<ValueType> *)
.. doxygenfunction:: rocalution::LocalMatrix::ConvertToCSR
.. doxygenfunction:: rocalution::LocalMatrix::ConvertToMCSR
.. doxygenfunction:: rocalution::LocalMatrix::ConvertToBCSR
.. doxygenfunction:: rocalution::LocalMatrix::ConvertToCOO
.. doxygenfunction:: rocalution::LocalMatrix::ConvertToELL
.. doxygenfunction:: rocalution::LocalMatrix::ConvertToDIA
.. doxygenfunction:: rocalution::LocalMatrix::ConvertToHYB
.. doxygenfunction:: rocalution::LocalMatrix::ConvertToDENSE
.. doxygenfunction:: rocalution::LocalMatrix::ConvertTo
.. doxygenfunction:: rocalution::LocalMatrix::SymbolicPower
.. doxygenfunction:: rocalution::LocalMatrix::MatrixAdd
.. doxygenfunction:: rocalution::LocalMatrix::MatrixMult
.. doxygenfunction:: rocalution::LocalMatrix::DiagonalMatrixMult
.. doxygenfunction:: rocalution::LocalMatrix::DiagonalMatrixMultL
.. doxygenfunction:: rocalution::LocalMatrix::DiagonalMatrixMultR
.. doxygenfunction:: rocalution::LocalMatrix::Gershgorin
.. doxygenfunction:: rocalution::LocalMatrix::Compress
.. doxygenfunction:: rocalution::LocalMatrix::Transpose
.. doxygenfunction:: rocalution::LocalMatrix::Sort
.. doxygenfunction:: rocalution::LocalMatrix::Key
.. doxygenfunction:: rocalution::LocalMatrix::ReplaceColumnVector
.. doxygenfunction:: rocalution::LocalMatrix::ReplaceRowVector
.. doxygenfunction:: rocalution::LocalMatrix::ExtractColumnVector
.. doxygenfunction:: rocalution::LocalMatrix::ExtractRowVector
.. doxygenfunction:: rocalution::LocalMatrix::AMGConnect
.. doxygenfunction:: rocalution::LocalMatrix::AMGAggregate
.. doxygenfunction:: rocalution::LocalMatrix::AMGSmoothedAggregation
.. doxygenfunction:: rocalution::LocalMatrix::AMGAggregation
.. doxygenfunction:: rocalution::LocalMatrix::RugeStueben
.. doxygenfunction:: rocalution::LocalMatrix::FSAI
.. doxygenfunction:: rocalution::LocalMatrix::SPAI
.. doxygenfunction:: rocalution::LocalMatrix::InitialPairwiseAggregation(ValueType, int&, LocalVector<int> *, int&, int **, int&, int) const
.. doxygenfunction:: rocalution::LocalMatrix::InitialPairwiseAggregation(const LocalMatrix<ValueType>&, ValueType, int&, LocalVector<int> *, int&, int **, int&, int) const
.. doxygenfunction:: rocalution::LocalMatrix::FurtherPairwiseAggregation(ValueType, int&, LocalVector<int> *, int&, int **, int&, int) const
.. doxygenfunction:: rocalution::LocalMatrix::FurtherPairwiseAggregation(const LocalMatrix<ValueType>&, ValueType, int&, LocalVector<int> *, int&, int **, int&, int) const
.. doxygenfunction:: rocalution::LocalMatrix::CoarsenOperator

Local Stencil
*************
.. doxygenclass:: rocalution::LocalStencil
.. doxygenfunction:: rocalution::LocalStencil::LocalStencil(unsigned int)
.. doxygenfunction:: rocalution::LocalStencil::GetNDim
.. doxygenfunction:: rocalution::LocalStencil::SetGrid

Global Matrix
*************
.. doxygenclass:: rocalution::GlobalMatrix
.. doxygenfunction:: rocalution::GlobalMatrix::GlobalMatrix(const ParallelManager&)
.. doxygenfunction:: rocalution::GlobalMatrix::Check
.. doxygenfunction:: rocalution::GlobalMatrix::AllocateCSR
.. doxygenfunction:: rocalution::GlobalMatrix::AllocateCOO
.. doxygenfunction:: rocalution::GlobalMatrix::SetParallelManager
.. doxygenfunction:: rocalution::GlobalMatrix::SetDataPtrCSR
.. doxygenfunction:: rocalution::GlobalMatrix::SetDataPtrCOO
.. doxygenfunction:: rocalution::GlobalMatrix::SetLocalDataPtrCSR
.. doxygenfunction:: rocalution::GlobalMatrix::SetLocalDataPtrCOO
.. doxygenfunction:: rocalution::GlobalMatrix::SetGhostDataPtrCSR
.. doxygenfunction:: rocalution::GlobalMatrix::SetGhostDataPtrCOO
.. doxygenfunction:: rocalution::GlobalMatrix::LeaveDataPtrCSR
.. doxygenfunction:: rocalution::GlobalMatrix::LeaveDataPtrCOO
.. doxygenfunction:: rocalution::GlobalMatrix::LeaveLocalDataPtrCSR
.. doxygenfunction:: rocalution::GlobalMatrix::LeaveLocalDataPtrCOO
.. doxygenfunction:: rocalution::GlobalMatrix::LeaveGhostDataPtrCSR
.. doxygenfunction:: rocalution::GlobalMatrix::LeaveGhostDataPtrCOO
.. doxygenfunction:: rocalution::GlobalMatrix::CloneFrom
.. doxygenfunction:: rocalution::GlobalMatrix::CopyFrom
.. doxygenfunction:: rocalution::GlobalMatrix::ConvertToCSR
.. doxygenfunction:: rocalution::GlobalMatrix::ConvertToMCSR
.. doxygenfunction:: rocalution::GlobalMatrix::ConvertToBCSR
.. doxygenfunction:: rocalution::GlobalMatrix::ConvertToCOO
.. doxygenfunction:: rocalution::GlobalMatrix::ConvertToELL
.. doxygenfunction:: rocalution::GlobalMatrix::ConvertToDIA
.. doxygenfunction:: rocalution::GlobalMatrix::ConvertToHYB
.. doxygenfunction:: rocalution::GlobalMatrix::ConvertToDENSE
.. doxygenfunction:: rocalution::GlobalMatrix::ConvertTo
.. doxygenfunction:: rocalution::GlobalMatrix::ReadFileMTX
.. doxygenfunction:: rocalution::GlobalMatrix::WriteFileMTX
.. doxygenfunction:: rocalution::GlobalMatrix::ReadFileCSR
.. doxygenfunction:: rocalution::GlobalMatrix::WriteFileCSR
.. doxygenfunction:: rocalution::GlobalMatrix::Sort
.. doxygenfunction:: rocalution::GlobalMatrix::ExtractInverseDiagonal
.. doxygenfunction:: rocalution::GlobalMatrix::Scale
.. doxygenfunction:: rocalution::GlobalMatrix::InitialPairwiseAggregation
.. doxygenfunction:: rocalution::GlobalMatrix::FurtherPairwiseAggregation
.. doxygenfunction:: rocalution::GlobalMatrix::CoarsenOperator

Local Vector
************
.. doxygenclass:: rocalution::LocalVector
.. doxygenfunction:: rocalution::LocalVector::Allocate
.. doxygenfunction:: rocalution::LocalVector::SetDataPtr
.. doxygenfunction:: rocalution::LocalVector::LeaveDataPtr
.. doxygenfunction:: rocalution::LocalVector::operator[](int)
.. doxygenfunction:: rocalution::LocalVector::operator[](int) const
.. doxygenfunction:: rocalution::LocalVector::CopyFromPermute
.. doxygenfunction:: rocalution::LocalVector::CopyFromPermuteBackward
.. doxygenfunction:: rocalution::LocalVector::CopyFromData
.. doxygenfunction:: rocalution::LocalVector::CopyToData
.. doxygenfunction:: rocalution::LocalVector::Permute
.. doxygenfunction:: rocalution::LocalVector::PermuteBackward
.. doxygenfunction:: rocalution::LocalVector::Restriction
.. doxygenfunction:: rocalution::LocalVector::Prolongation
.. doxygenfunction:: rocalution::LocalVector::SetIndexArray
.. doxygenfunction:: rocalution::LocalVector::GetIndexValues
.. doxygenfunction:: rocalution::LocalVector::SetIndexValues
.. doxygenfunction:: rocalution::LocalVector::GetContinuousValues
.. doxygenfunction:: rocalution::LocalVector::SetContinuousValues
.. doxygenfunction:: rocalution::LocalVector::ExtractCoarseMapping
.. doxygenfunction:: rocalution::LocalVector::ExtractCoarseBoundary

Global Vector
*************
.. doxygenclass:: rocalution::GlobalVector
.. doxygenfunction:: rocalution::GlobalVector::GlobalVector(const ParallelManager&)
.. doxygenfunction:: rocalution::GlobalVector::Allocate
.. doxygenfunction:: rocalution::GlobalVector::SetParallelManager
.. doxygenfunction:: rocalution::GlobalVector::operator[](int)
.. doxygenfunction:: rocalution::GlobalVector::operator[](int) const
.. doxygenfunction:: rocalution::GlobalVector::SetDataPtr
.. doxygenfunction:: rocalution::GlobalVector::LeaveDataPtr
.. doxygenfunction:: rocalution::GlobalVector::Restriction
.. doxygenfunction:: rocalution::GlobalVector::Prolongation

Parallel Manager
****************
.. doxygenclass:: rocalution::ParallelManager
.. doxygenfunction:: rocalution::ParallelManager::SetMPICommunicator
.. doxygenfunction:: rocalution::ParallelManager::Clear
.. doxygenfunction:: rocalution::ParallelManager::GetGlobalSize
.. doxygenfunction:: rocalution::ParallelManager::GetLocalSize
.. doxygenfunction:: rocalution::ParallelManager::GetNumReceivers
.. doxygenfunction:: rocalution::ParallelManager::GetNumSenders
.. doxygenfunction:: rocalution::ParallelManager::GetNumProcs
.. doxygenfunction:: rocalution::ParallelManager::SetGlobalSize
.. doxygenfunction:: rocalution::ParallelManager::SetLocalSize
.. doxygenfunction:: rocalution::ParallelManager::SetBoundaryIndex
.. doxygenfunction:: rocalution::ParallelManager::SetReceivers
.. doxygenfunction:: rocalution::ParallelManager::SetSenders
.. doxygenfunction:: rocalution::ParallelManager::LocalToGlobal
.. doxygenfunction:: rocalution::ParallelManager::GlobalToLocal
.. doxygenfunction:: rocalution::ParallelManager::Status
.. doxygenfunction:: rocalution::ParallelManager::ReadFileASCII
.. doxygenfunction:: rocalution::ParallelManager::WriteFileASCII

Solvers
*******
.. doxygenclass:: rocalution::Solver
.. doxygenfunction:: rocalution::Solver::SetOperator
.. doxygenfunction:: rocalution::Solver::ResetOperator
.. doxygenfunction:: rocalution::Solver::Print
.. doxygenfunction:: rocalution::Solver::Solve
.. doxygenfunction:: rocalution::Solver::SolveZeroSol
.. doxygenfunction:: rocalution::Solver::Clear
.. doxygenfunction:: rocalution::Solver::Build
.. doxygenfunction:: rocalution::Solver::BuildMoveToAcceleratorAsync
.. doxygenfunction:: rocalution::Solver::Sync
.. doxygenfunction:: rocalution::Solver::ReBuildNumeric
.. doxygenfunction:: rocalution::Solver::MoveToHost
.. doxygenfunction:: rocalution::Solver::MoveToAccelerator
.. doxygenfunction:: rocalution::Solver::Verbose

Iterative Linear Solvers
````````````````````````
.. doxygenclass:: rocalution::IterativeLinearSolver
.. doxygenfunction:: rocalution::IterativeLinearSolver::Init(double, double, double, int)
.. doxygenfunction:: rocalution::IterativeLinearSolver::Init(double, double, double, int, int)
.. doxygenfunction:: rocalution::IterativeLinearSolver::InitMinIter
.. doxygenfunction:: rocalution::IterativeLinearSolver::InitMaxIter
.. doxygenfunction:: rocalution::IterativeLinearSolver::InitTol
.. doxygenfunction:: rocalution::IterativeLinearSolver::SetResidualNorm
.. doxygenfunction:: rocalution::IterativeLinearSolver::RecordResidualHistory
.. doxygenfunction:: rocalution::IterativeLinearSolver::RecordHistory
.. doxygenfunction:: rocalution::IterativeLinearSolver::Verbose
.. doxygenfunction:: rocalution::IterativeLinearSolver::Solve
.. doxygenfunction:: rocalution::IterativeLinearSolver::SetPreconditioner
.. doxygenfunction:: rocalution::IterativeLinearSolver::GetIterationCount
.. doxygenfunction:: rocalution::IterativeLinearSolver::GetCurrentResidual
.. doxygenfunction:: rocalution::IterativeLinearSolver::GetSolverStatus
.. doxygenfunction:: rocalution::IterativeLinearSolver::GetAmaxResidualIndex

.. doxygenclass:: rocalution::FixedPoint
.. doxygenfunction:: rocalution::FixedPoint::SetRelaxation

.. doxygenclass:: rocalution::MixedPrecisionDC
.. doxygenfunction:: rocalution::MixedPrecisionDC::Set

.. doxygenclass:: rocalution::Chebyshev
.. doxygenfunction:: rocalution::Chebyshev::Set

Krylov Subspace Solvers
-----------------------
.. doxygenclass:: rocalution::BiCGStab
.. doxygenclass:: rocalution::BiCGStabl
.. doxygenfunction:: rocalution::BiCGStabl::SetOrder
.. doxygenclass:: rocalution::CG
.. doxygenclass:: rocalution::CR
.. doxygenclass:: rocalution::FCG
.. doxygenclass:: rocalution::GMRES
.. doxygenfunction:: rocalution::GMRES::SetBasisSize
.. doxygenclass:: rocalution::FGMRES
.. doxygenfunction:: rocalution::FGMRES::SetBasisSize
.. doxygenclass:: rocalution::IDR
.. doxygenfunction:: rocalution::IDR::SetShadowSpace
.. doxygenfunction:: rocalution::IDR::SetRandomSeed
.. doxygenclass:: rocalution::QMRCGStab

MultiGrid Solvers
-----------------
.. doxygenclass:: rocalution::BaseMultiGrid
.. doxygenfunction:: rocalution::BaseMultiGrid::SetSolver
.. doxygenfunction:: rocalution::BaseMultiGrid::SetSmoother
.. doxygenfunction:: rocalution::BaseMultiGrid::SetSmootherPreIter
.. doxygenfunction:: rocalution::BaseMultiGrid::SetSmootherPostIter
.. doxygenfunction:: rocalution::BaseMultiGrid::SetRestrictOperator
.. doxygenfunction:: rocalution::BaseMultiGrid::SetProlongOperator
.. doxygenfunction:: rocalution::BaseMultiGrid::SetOperatorHierarchy
.. doxygenfunction:: rocalution::BaseMultiGrid::SetScaling
.. doxygenfunction:: rocalution::BaseMultiGrid::SetHostLevels
.. doxygenfunction:: rocalution::BaseMultiGrid::SetCycle
.. doxygenfunction:: rocalution::BaseMultiGrid::SetKcycleFull
.. doxygenfunction:: rocalution::BaseMultiGrid::InitLevels

.. doxygenclass:: rocalution::MultiGrid

.. doxygenclass:: rocalution::BaseAMG
.. doxygenfunction:: rocalution::BaseAMG::ClearLocal
.. doxygenfunction:: rocalution::BaseAMG::BuildHierarchy
.. doxygenfunction:: rocalution::BaseAMG::BuildSmoothers
.. doxygenfunction:: rocalution::BaseAMG::SetCoarsestLevel
.. doxygenfunction:: rocalution::BaseAMG::SetManualSmoothers
.. doxygenfunction:: rocalution::BaseAMG::SetManualSolver
.. doxygenfunction:: rocalution::BaseAMG::SetDefaultSmootherFormat
.. doxygenfunction:: rocalution::BaseAMG::SetOperatorFormat
.. doxygenfunction:: rocalution::BaseAMG::GetNumLevels

.. doxygenclass:: rocalution::UAAMG
.. doxygenfunction:: rocalution::UAAMG::SetCouplingStrength
.. doxygenfunction:: rocalution::UAAMG::SetOverInterp

.. doxygenclass:: rocalution::SAAMG
.. doxygenfunction:: rocalution::SAAMG::SetCouplingStrength
.. doxygenfunction:: rocalution::SAAMG::SetInterpRelax

.. doxygenclass:: rocalution::RugeStuebenAMG
.. doxygenfunction:: rocalution::RugeStuebenAMG::SetCouplingStrength

.. doxygenclass:: rocalution::PairwiseAMG
.. doxygenfunction:: rocalution::PairwiseAMG::SetBeta
.. doxygenfunction:: rocalution::PairwiseAMG::SetOrdering
.. doxygenfunction:: rocalution::PairwiseAMG::SetCoarseningFactor

.. doxygenclass:: rocalution::GlobalPairwiseAMG
.. doxygenfunction:: rocalution::GlobalPairwiseAMG::SetBeta
.. doxygenfunction:: rocalution::GlobalPairwiseAMG::SetOrdering
.. doxygenfunction:: rocalution::GlobalPairwiseAMG::SetCoarseningFactor

Direct Solvers
``````````````
.. doxygenclass:: rocalution::DirectLinearSolver

.. doxygenclass:: rocalution::Inversion
.. doxygenclass:: rocalution::LU
.. doxygenclass:: rocalution::QR

Preconditioners
***************
.. doxygenclass:: rocalution::Preconditioner

.. doxygenclass:: rocalution::AIChebyshev
.. doxygenfunction:: rocalution::AIChebyshev::Set

.. doxygenclass:: rocalution::FSAI
.. doxygenfunction:: rocalution::FSAI::Set(int)
.. doxygenfunction:: rocalution::FSAI::Set(const OperatorType&)
.. doxygenfunction:: rocalution::FSAI::SetPrecondMatrixFormat

.. doxygenclass:: rocalution::SPAI
.. doxygenfunction:: rocalution::SPAI::SetPrecondMatrixFormat

.. doxygenclass:: rocalution::TNS
.. doxygenfunction:: rocalution::TNS::Set
.. doxygenfunction:: rocalution::TNS::SetPrecondMatrixFormat

.. doxygenclass:: rocalution::AS
.. doxygenfunction:: rocalution::AS::Set
.. doxygenclass:: rocalution::RAS

.. doxygenclass:: rocalution::BlockJacobi
.. doxygenfunction:: rocalution::BlockJacobi::Set

.. doxygenclass:: rocalution::BlockPreconditioner
.. doxygenfunction:: rocalution::BlockPreconditioner::Set
.. doxygenfunction:: rocalution::BlockPreconditioner::SetDiagonalSolver
.. doxygenfunction:: rocalution::BlockPreconditioner::SetLSolver
.. doxygenfunction:: rocalution::BlockPreconditioner::SetExternalLastMatrix
.. doxygenfunction:: rocalution::BlockPreconditioner::SetPermutation

.. doxygenclass:: rocalution::Jacobi
.. doxygenclass:: rocalution::GS
.. doxygenclass:: rocalution::SGS

.. doxygenclass:: rocalution::ILU
.. doxygenfunction:: rocalution::ILU::Set

.. doxygenclass:: rocalution::ILUT
.. doxygenfunction:: rocalution::ILUT::Set(double)
.. doxygenfunction:: rocalution::ILUT::Set(double, int)

.. doxygenclass:: rocalution::IC

.. doxygenclass:: rocalution::VariablePreconditioner
.. doxygenfunction:: rocalution::VariablePreconditioner::SetPreconditioner

.. doxygenclass:: rocalution::MultiColored
.. doxygenfunction:: rocalution::MultiColored::SetPrecondMatrixFormat
.. doxygenfunction:: rocalution::MultiColored::SetDecomposition

.. doxygenclass:: rocalution::MultiColoredSGS
.. doxygenfunction:: rocalution::MultiColoredSGS::SetRelaxation

.. doxygenclass:: rocalution::MultiColoredGS

.. doxygenclass:: rocalution::MultiColoredILU
.. doxygenfunction:: rocalution::MultiColoredILU::Set(int)
.. doxygenfunction:: rocalution::MultiColoredILU::Set(int, int, bool)

.. doxygenclass:: rocalution::MultiElimination
.. doxygenfunction:: rocalution::MultiElimination::GetSizeDiagBlock
.. doxygenfunction:: rocalution::MultiElimination::GetLevel
.. doxygenfunction:: rocalution::MultiElimination::Set
.. doxygenfunction:: rocalution::MultiElimination::SetPrecondMatrixFormat

.. doxygenclass:: rocalution::DiagJacobiSaddlePointPrecond
.. doxygenfunction:: rocalution::DiagJacobiSaddlePointPrecond::Set
