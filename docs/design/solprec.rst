.. _rocalution_solprec:

***************************
Solvers and Preconditioners
***************************

Most of the solvers can be performed on linear operators, e.g. :cpp:class:`LocalMatrix <rocalution::LocalMatrix>`, :cpp:class:`LocalStencil <rocalution::LocalStencil>` and :cpp:class:`GlobalMatrix <rocalution::GlobalMatrix>` - i.e. the solvers can be performed locally (on a shared memory system) or in a distributed manner (on a cluster) via MPI.
All solvers and preconditioners need three template parameters - Operators, Vectors and Scalar type.
The Solver class is purely virtual and provides an interface for

- :cpp:func:`SetOperator <rocalution::Solver::SetOperator>` to set the operator, i.e. the user can pass the matrix here.
- :cpp:func:`Build <rocalution::Solver::Build>` to build the solver (including preconditioners, sub-solvers, etc.).
  The user need to specify the operator first before building the solver.
- :cpp:func:`Solve <rocalution::Solver::Solve>` to solve the sparse linear system.
  The user need to pass a right-hand side and a solution / initial guess vector.
- :cpp:func:`Print <rocalution::Solver::Print>` to show solver information.
- :cpp:func:`ReBuildNumeric <rocalution::Solver::ReBuildNumeric>` to only re-build the solver numerically (if possible).
- :cpp:func:`MoveToHost <rocalution::Solver::MoveToHost>` and :cpp:func:`MoveToAccelerator <rocalution::Solver::MoveToAccelerator>` to offload the solver (including preconditioners and sub-solvers) to the host / accelerator.

.. _solvers:
.. figure:: ../fig/solvers.png
   :alt: solver and preconditioner classes
   :align: center

   Solver and preconditioner classes.
