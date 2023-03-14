*******
Solvers
*******

Code Structure
==============
.. doxygenclass:: rocalution::Solver

It provides an interface for

.. doxygenfunction:: rocalution::Solver::SetOperator
.. doxygenfunction:: rocalution::Solver::Build
.. doxygenfunction:: rocalution::Solver::Clear
.. doxygenfunction:: rocalution::Solver::Solve
.. doxygenfunction:: rocalution::Solver::Print
.. doxygenfunction:: rocalution::Solver::ReBuildNumeric
.. doxygenfunction:: rocalution::Solver::MoveToHost
.. doxygenfunction:: rocalution::Solver::MoveToAccelerator

Iterative Linear Solvers
========================
.. doxygenclass:: rocalution::IterativeLinearSolver

It provides an interface for

.. doxygenfunction:: rocalution::IterativeLinearSolver::Init(double, double, double, int)
.. doxygenfunction:: rocalution::IterativeLinearSolver::Init(double, double, double, int, int)
.. doxygenfunction:: rocalution::IterativeLinearSolver::InitMinIter
.. doxygenfunction:: rocalution::IterativeLinearSolver::InitMaxIter
.. doxygenfunction:: rocalution::IterativeLinearSolver::InitTol
.. doxygenfunction:: rocalution::IterativeLinearSolver::RecordResidualHistory
.. doxygenfunction:: rocalution::IterativeLinearSolver::RecordHistory
.. doxygenfunction:: rocalution::IterativeLinearSolver::Verbose
.. doxygenfunction:: rocalution::IterativeLinearSolver::SetPreconditioner
.. doxygenfunction:: rocalution::IterativeLinearSolver::SetResidualNorm
.. doxygenfunction:: rocalution::IterativeLinearSolver::GetAmaxResidualIndex
.. doxygenfunction:: rocalution::IterativeLinearSolver::GetSolverStatus

Building and Solving Phase
==========================
Each iterative solver consists of a building step and a solving step. During the building step all necessary auxiliary data is allocated and the preconditioner is constructed. After that, the user can call the solving procedure, the solving step can be called several times.

When the initial matrix associated with the solver is on the accelerator, the solver will try to build everything on the accelerator. However, some preconditioners and solvers (such as FSAI and AMG) need to be constructed on the host before they can be transferred to the accelerator. If the initial matrix is on the host and we want to run the solver on the accelerator then we need to move the solver to the accelerator as well as the matrix, the right-hand-side and the solution vector.

.. note:: If you have a preconditioner associate with the solver, it will be moved automatically to the accelerator when you move the solver.

.. code-block:: cpp

  // CG solver
  CG<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType> ls;
  // Multi-Colored ILU preconditioner
  MultiColoredILU<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType> p;

  // Move matrix and vectors to the accelerator
  mat.MoveToAccelerator();
  rhs.MoveToAccelerator();
  x.MoveToAccelerator();

  // Set mat to be the operator
  ls.SetOperator(mat);
  // Set p as the preconditioner of ls
  ls.SetPreconditioner(p);

  // Build the solver and preconditioner on the accelerator
  ls.Build();

  // Compute the solution on the accelerator
  ls.Solve(rhs, &x);

.. code-block:: cpp

  // CG solver
  CG<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType> ls;
  // Multi-Colored ILU preconditioner
  MultiColoredILU<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType> p;

  // Set mat to be the operator
  ls.SetOperator(mat);
  // Set p as the preconditioner of ls
  ls.SetPreconditioner(p);

  // Build the solver and preconditioner on the host
  ls.Build();

  // Move matrix and vectors to the accelerator
  mat.MoveToAccelerator();
  rhs.MoveToAccelerator();
  x.MoveToAccelerator();

  // Move linear solver to the accelerator
  ls.MoveToAccelerator();

  // Compute the solution on the accelerator
  ls.Solve(rhs, &x);


Clear Function and Destructor
=============================
The :cpp:func:`rocalution::Solver::Clear` function clears all the data which is in the solver, including the associated preconditioner. Thus, the solver is not anymore associated with this preconditioner.

.. note:: The preconditioner is not deleted (via destructor), only a :cpp:func:`rocalution::Preconditioner::Clear` is called.

.. note:: When the destructor of the solver class is called, it automatically calls the *Clear()* function. Be careful, when declaring your solver and preconditioner in different places - we highly recommend to manually call the *Clear()* function of the solver and not to rely on the destructor of the solver.

Numerical Update
================
Some preconditioners require two phases in the their construction: an algebraic (e.g. compute a pattern or structure) and a numerical (compute the actual values) phase. In cases, where the structure of the input matrix is a constant (e.g. Newton-like methods) it is not necessary to fully re-construct the preconditioner. In this case, the user can apply a numerical update to the current preconditioner and pass the new operator with :cpp:func:`rocalution::Solver::ReBuildNumeric`. If the preconditioner/solver does not support the numerical update, then a full :cpp:func:`rocalution::Solver::Clear` and :cpp:func:`rocalution::Solver::Build` will be performed.

Fixed-Point Iteration
=====================
.. doxygenclass:: rocalution::FixedPoint
.. doxygenfunction:: rocalution::FixedPoint::SetRelaxation

Krylov Subspace Solvers
=======================

CG
--
.. doxygenclass:: rocalution::CG

For further details, see :cite:`SAAD`.

CR
--
.. doxygenclass:: rocalution::CR

For further details, see :cite:`SAAD`.

GMRES
-----
.. doxygenclass:: rocalution::GMRES
.. doxygenfunction:: rocalution::GMRES::SetBasisSize

For further details, see :cite:`SAAD`.

FGMRES
------
.. doxygenclass:: rocalution::FGMRES
.. doxygenfunction:: rocalution::FGMRES::SetBasisSize

For further details, see :cite:`SAAD`.

BiCGStab
--------
.. doxygenclass:: rocalution::BiCGStab

For further details, see :cite:`SAAD`.

IDR
---
.. doxygenclass:: rocalution::IDR
.. doxygenfunction:: rocalution::IDR::SetShadowSpace

For further details, see :cite:`IDR1` and :cite:`IDR2`.

FCG
---
.. doxygenclass:: rocalution::FCG

For further details, see :cite:`fcg`.

QMRCGStab
---------
.. doxygenclass:: rocalution::QMRCGStab

For further details, see :cite:`qmrcgstab`.

BiCGStab(l)
-----------
.. doxygenclass:: rocalution::BiCGStabl
.. doxygenfunction:: rocalution::BiCGStabl::SetOrder

For further details, see :cite:`bicgstabl`.

Chebyshev Iteration Scheme
==========================
.. doxygenclass:: rocalution::Chebyshev

For further details, see :cite:`templates`.

Mixed-Precision Defect Correction Scheme
========================================
.. doxygenclass:: rocalution::MixedPrecisionDC

MultiGrid Solvers
=================
The library provides algebraic multigrid as well as a skeleton for geometric multigrid methods. The BaseMultigrid class itself is not constructing the data for the method. It contains the solution procedure for V, W and K-cycles. The AMG has two different versions for Local (non-MPI) and for Global (MPI) type of computations.

.. doxygenclass:: rocalution::BaseMultiGrid

Geometric MultiGrid
-------------------
.. doxygenclass:: rocalution::MultiGrid

For further details, see :cite:`Trottenberg2003`.

Algebraic MultiGrid
-------------------
.. doxygenclass:: rocalution::BaseAMG
.. doxygenfunction:: rocalution::BaseAMG::BuildHierarchy
.. doxygenfunction:: rocalution::BaseAMG::BuildSmoothers
.. doxygenfunction:: rocalution::BaseAMG::SetCoarsestLevel
.. doxygenfunction:: rocalution::BaseAMG::SetManualSmoothers
.. doxygenfunction:: rocalution::BaseAMG::SetManualSolver
.. doxygenfunction:: rocalution::BaseAMG::SetDefaultSmootherFormat
.. doxygenfunction:: rocalution::BaseAMG::SetOperatorFormat
.. doxygenfunction:: rocalution::BaseAMG::GetNumLevels

Unsmoothed Aggregation AMG
==========================
.. doxygenclass:: rocalution::UAAMG
.. doxygenfunction:: rocalution::UAAMG::SetCouplingStrength
.. doxygenfunction:: rocalution::UAAMG::SetOverInterp

For further details, see :cite:`stueben`.

Smoothed Aggregation AMG
========================
.. doxygenclass:: rocalution::SAAMG
.. doxygenfunction:: rocalution::SAAMG::SetCouplingStrength
.. doxygenfunction:: rocalution::SAAMG::SetInterpRelax

For further details, see :cite:`vanek`.

Ruge-Stueben AMG
================
.. doxygenclass:: rocalution::RugeStuebenAMG
.. doxygenfunction:: rocalution::RugeStuebenAMG::SetCouplingStrength

For further details, see :cite:`stueben`.

Pairwise AMG
============
.. doxygenclass:: rocalution::PairwiseAMG
.. doxygenfunction:: rocalution::PairwiseAMG::SetBeta
.. doxygenfunction:: rocalution::PairwiseAMG::SetOrdering
.. doxygenfunction:: rocalution::PairwiseAMG::SetCoarseningFactor

For further details, see :cite:`pairwiseamg`.

Direct Linear Solvers
=====================
.. doxygenclass:: rocalution::DirectLinearSolver
.. doxygenclass:: rocalution::LU
.. doxygenclass:: rocalution::QR
.. doxygenclass:: rocalution::Inversion

.. note:: These methods can only be used with local-type problems.
