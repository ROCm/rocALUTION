.. meta::
   :description: A sparse linear algebra library with focus on exploring fine-grained parallelism on top of the AMD ROCm runtime and toolchains
   :keywords: rocALUTION, ROCm, library, API, tool

.. _design-philosophy:

*********************
Design and Philosophy
*********************
rocALUTION is written in C++ and HIP.

The main idea of the rocALUTION objects is that they are separated from the actual hardware specification.
Once you declare a matrix, a vector or a solver they are initially allocated on the host (CPU).
Then, every object can be moved to a selected accelerator by a simple function call.
The whole execution mechanism is based on run-time type information (RTTI), which allows you to select where and how you want to perform the operations at run time.
This is in contrast to the template-based libraries, which need this information at compile time.

The philosophy of the library is to abstract the hardware-specific functions and routines from the actual program, that describes the algorithm.
It is hard and almost impossible for most of the large simulation software based on sparse computation, to adapt and port their implementation in order to use every new technology.
On the other hand, the new high performance accelerators and devices have the capability to decrease the computational time significantly in many critical parts.

This abstraction layer of the hardware specific routines is the core of the rocALUTION design.
It is built to explore fine-grained level of parallelism suited for multi/many-core devices.
This is in contrast to most of the parallel sparse libraries available which are mainly based on domain decomposition techniques.
Thus, the design of the iterative solvers the preconditioners is very different.
Another cornerstone of rocALUTION is the native support of accelerators - the memory allocation, transfers and specific hardware functions are handled internally in the library.

rocALUTION helps you to use accelerator technologies but does not force you to use them.
Even if you offload your algorithms and solvers to the accelerator device, the same source code can be compiled and executed in a system without any accelerator.

Naturally, not all routines and algorithms can be performed efficiently on many-core systems (i.e. on accelerators).
To provide full functionality, the library has internal mechanisms to check if a particular routine is implemented on the accelerator.
If not, the object is moved to the host and the routine is computed there.
This guarantees that your code will run with any accelerator, regardless of the available functionality for it.
