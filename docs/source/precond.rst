###############
Preconditioners
###############
In this chapter, all preconditioners are presented. All preconditioners support local operators. They can be used as a global preconditioner via block-jacobi scheme which works locally on each interior matrix. To provide fast application, all preconditioners require extra memory to keep the approximated operator.

.. doxygenclass:: rocalution::Preconditioner

Code Structure
==============
The preconditioners provide a solution to the system :math:`Mz = r`, where either the solution :math:`z` is directly computed by the approximation scheme or it is iteratively obtained with :math:`z = 0` initial guess.

Jacobi Method
=============
.. doxygenclass:: rocalution::Jacobi
.. note:: Damping parameter :math:`\omega` can be adjusted by :cpp:func:`rocalution::FixedPoint::SetRelaxation`.

(Symmetric) Gauss-Seidel / (S)SOR Method
========================================
.. doxygenclass:: rocalution::GS
.. doxygenclass:: rocalution::SGS
.. note:: Relaxation parameter :math:`\omega` can be adjusted by :cpp:func:`rocalution::FixedPoint::SetRelaxation`.

Incomplete Factorizations
=========================

ILU
---
.. doxygenclass:: rocalution::ILU
.. doxygenfunction:: rocalution::ILU::Set

ILUT
----
.. doxygenclass:: rocalution::ILUT
.. doxygenfunction:: rocalution::ILUT::Set(double)
.. doxygenfunction:: rocalution::ILUT::Set(double, int)

IC
--
.. doxygenclass:: rocalution::IC

AI Chebyshev
============
.. doxygenclass:: rocalution::AIChebyshev
.. doxygenfunction:: rocalution::AIChebyshev::Set

FSAI
====
.. doxygenclass:: rocalution::FSAI
.. doxygenfunction:: rocalution::FSAI::Set(int)
.. doxygenfunction:: rocalution::FSAI::Set(const OperatorType&)
.. doxygenfunction:: rocalution::FSAI::SetPrecondMatrixFormat

SPAI
====
.. doxygenclass:: rocalution::SPAI
.. doxygenfunction:: rocalution::SPAI::SetPrecondMatrixFormat

TNS
===
.. doxygenclass:: rocalution::TNS
.. doxygenfunction:: rocalution::TNS::Set
.. doxygenfunction:: rocalution::TNS::SetPrecondMatrixFormat

MultiColored Preconditioners
============================
.. doxygenclass:: rocalution::MultiColored
.. doxygenfunction:: rocalution::MultiColored::SetPrecondMatrixFormat
.. doxygenfunction:: rocalution::MultiColored::SetDecomposition

MultiColored (Symmetric) Gauss-Seidel / (S)SOR
----------------------------------------------
.. doxygenclass:: rocalution::MultiColoredGS
.. doxygenclass:: rocalution::MultiColoredSGS
.. doxygenfunction:: rocalution::MultiColoredSGS::SetRelaxation
.. note:: The preconditioner matrix format can be changed using :cpp:func:`rocalution::MultiColored::SetPrecondMatrixFormat`.

MultiColored Power(q)-pattern method ILU(p,q)
---------------------------------------------
.. doxygenclass:: rocalution::MultiColoredILU
.. doxygenfunction:: rocalution::MultiColoredILU::Set(int)
.. doxygenfunction:: rocalution::MultiColoredILU::Set(int, int, bool)
.. note:: The preconditioner matrix format can be changed using :cpp:func:`rocalution::MultiColored::SetPrecondMatrixFormat`.

Multi-Elimination Incomplete LU
===============================
.. doxygenclass:: rocalution::MultiElimination
.. doxygenfunction:: rocalution::MultiElimination::GetSizeDiagBlock
.. doxygenfunction:: rocalution::MultiElimination::GetLevel
.. doxygenfunction:: rocalution::MultiElimination::Set
.. doxygenfunction:: rocalution::MultiElimination::SetPrecondMatrixFormat

Diagonal Preconditioner for Saddle-Point Problems
=================================================
.. doxygenclass:: rocalution::DiagJacobiSaddlePointPrecond
.. doxygenfunction:: rocalution::DiagJacobiSaddlePointPrecond::Set

(Restricted) Additive Schwarz Preconditioner
============================================
.. doxygenclass:: rocalution::AS
.. doxygenfunction:: rocalution::AS::Set
.. doxygenclass:: rocalution::RAS

The overlapped area is shown in :numref:`AS`.

.. _AS:
.. figure:: ../fig/AS.png
  :alt: 4 block additive schwarz
  :align: center

  Example of a 4 block-decomposed matrix - Additive Schwarz with overlapping preconditioner (left) and Restricted Additive Schwarz preconditioner (right).

Block-Jacobi (MPI) Preconditioner
=================================
.. doxygenclass:: rocalution::BlockJacobi
.. doxygenfunction:: rocalution::BlockJacobi::Set

The Block-Jacobi (MPI) preconditioner is shown in :numref:`BJ`.

.. _BJ:
.. figure:: ../fig/BJ.png
  :alt: 4 block jacobi
  :align: center

  Example of a 4 block-decomposed matrix - Block-Jacobi preconditioner.

Block Preconditioner
====================
.. doxygenclass:: rocalution::BlockPreconditioner
.. doxygenfunction:: rocalution::BlockPreconditioner::Set
.. doxygenfunction:: rocalution::BlockPreconditioner::SetDiagonalSolver
.. doxygenfunction:: rocalution::BlockPreconditioner::SetLSolver
.. doxygenfunction:: rocalution::BlockPreconditioner::SetExternalLastMatrix
.. doxygenfunction:: rocalution::BlockPreconditioner::SetPermutation


Variable Preconditioner
=======================
.. doxygenclass:: rocalution::VariablePreconditioner
.. doxygenfunction:: rocalution::VariablePreconditioner::SetPreconditioner

