.. _rocalution_opvec:

*********************
Operators and Vectors
*********************

The main objects in rocALUTION are linear operators and vectors.
All objects can be moved to an accelerator at run time.
The linear operators are defined as local or global matrices (i.e. on a single node or distributed/multi-node) and local stencils (i.e. matrix-free linear operations).
The only template parameter of the operators and vectors is the data type (ValueType).
The operator data type could be float, double, complex float or complex double, while the vector data type can be int, float, double, complex float or complex double (int is used mainly for the permutation vectors).
In the current version, cross ValueType object operations are not supported.
:numref:`operators` gives an overview of supported operators and vectors.

.. _operators:
.. figure:: ../fig/operators.png
  :alt: operator and vector classes
  :align: center

  Operator and vector classes.

Each of the objects contain a local copy of the hardware descriptor created by the :cpp:func:`init_rocalution <rocalution::init_rocalution>` function.
This allows the user to modify it according to his needs and to obtain two or more objects with different hardware specifications (e.g. different amount of OpenMP threads, HIP block sizes, etc.).

Local Operators and Vectors
===========================
By Local Operators and Vectors, rocALUTION refers to the LocalMatrix, LocalStencil and LocalVector class.
A local object is called local, because it will always stay on a single system.
The system can contain several CPUs via UMA or NUMA memory system or an accelerator.

Global Operators and Vectors
============================
By Global Operators and Vectors, rocALUTION refers to the GlobalMatrix and GlobalVector class.
A global object is called global, because it can stay on a single or on multiple nodes in a network.
For this type of communication, MPI is used.
