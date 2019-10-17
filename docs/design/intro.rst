************
Introduction
************

Overview
========
rocALUTION is a sparse linear algebra library with focus on exploring fine-grained parallelism, targeting modern processors and accelerators including multi/many-core CPU and GPU platforms. The main goal of this package is to provide a portable library for iterative sparse methods on state of the art hardware. rocALUTION can be seen as middle-ware between different parallel backends and application specific packages.

The major features and characteristics of the library are

* Various backends
    * Host - fallback backend, designed for CPUs
    * GPU/HIP - accelerator backend, designed for HIP capable AMD GPUs
    * OpenMP - designed for multi-core CPUs
    * MPI - designed for multi-node and multi-GPU configurations
* Easy to use
    The syntax and structure of the library provide easy learning curves. With the help of the examples, anyone can try out the library - no knowledge in HIP, OpenMP or MPI programming required.
* No special hardware requirements
    There are no hardware requirements to install and run rocALUTION. If a GPU device and HIP is available, the library will use them.
* Variety of iterative solvers
    * Fixed-Point iteration - Jacobi, Gauss-Seidel, Symmetric-Gauss Seidel, SOR and SSOR
    * Krylov subspace methods - CR, CG, BiCGStab, BiCGStab(l), GMRES, IDR, QMRCGSTAB, Flexible CG/GMRES
    * Mixed-precision defect-correction scheme
    * Chebyshev iteration
    * Multiple MultiGrid schemes, geometric and algebraic
* Various preconditioners
    * Matrix splitting - Jacobi, (Multi-colored) Gauss-Seidel, Symmetric Gauss-Seidel, SOR, SSOR
    * Factorization - ILU(0), ILU(p) (based on levels), ILU(p,q) (power(q)-pattern method), Multi-Elimination ILU (nested/recursive), ILUT (based on threshold) and IC(0)
    * Approximate Inverse - Chebyshev matrix-valued polynomial, SPAI, FSAI and TNS
    * Diagonal-based preconditioner for Saddle-point problems
    * Block-type of sub-preconditioners/solvers
    * Additive Schwarz and Restricted Additive Schwarz
    * Variable type preconditioners
* Generic and robust design
    rocALUTION is based on a generic and robust design allowing expansion in the direction of new solvers and preconditioners and support for various hardware types. Furthermore, the design of the library allows the use of all solvers as preconditioners in other solvers. For example you can easily define a CG solver with a Multi-Elimination preconditioner, where the last-block is preconditioned with another Chebyshev iteration method which is preconditioned with a multi-colored Symmetric Gauss-Seidel scheme.
* Portable code and results
    All code based on rocALUTION is portable and independent of HIP or OpenMP. The code will compile and run everywhere. All solvers and preconditioners are based on a single source code, which delivers portable results across all supported backends (variations are possible due to different rounding modes on the hardware). The only difference which you can see for a hardware change is the performance variation.
* Support for several sparse matrix formats
    Compressed Sparse Row (CSR), Modified Compressed Sparse Row (MCSR), Dense (DENSE), Coordinate (COO), ELL, Diagonal (DIA), Hybrid format of ELL and COO (HYB).

The code is open-source under MIT license, see :ref:`rocalution_license` and hosted on the `GitHub rocALUTION page <https://github.com/ROCmSoftwarePlatform/rocALUTION>`_.

.. _rocalution_license:

License
=======

rocALUTION is distributed as open-source under the following license:

MIT License

Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

.. _rocalution_contributing:

Contributing
============

Contribution License Agreement
------------------------------

#. The code I am contributing is mine, and I have the right to license it.
#. By submitting a pull request for this project I am granting you a license to distribute said code under the MIT License for the project.

How to contribute
-----------------
Code contriubtion guidelines closely follow the model of GitHub pull-requests.
This repository follows the git flow workflow, which dictates a `master` branch where releases are cut, and a `develop` branch which serves as an integration branch for new code.

Pull-request guidelines
-----------------------
- Target the `**develop**` branch for integration.
- Ensure code builds successfully.
- Do not break existing test cases.
- New functionality will only be merged with new unit tests.

  - New unit tests should integrate within the existing `googletest framework <https://github.com/google/googletest/blob/master/googletest/docs/primer.md>`_.
  - Tests must have good code coverage.
  - Performance must approach the compute bound limit or memory bound limit.

StyleGuide
----------
This project follows the `CPP Core guidelines <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md>`_, with few modifications or additions noted below.
All pull-requests should in good faith attempt to follow the guidelines stated therein, but we recognize that the content is lengthy.
Below we list our primary concerns when reviewing pull-requests.

**Interface**

- All public APIs are C89 compatible; all other library code should use C++14.
- Our minimum supported compiler is clang 3.6.
- Avoid CamelCase.
- This rule applies specifically to publicly visible APIs, but is also encouraged (not mandated) for internal code.

**Philosophy**

- `P.2 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rp-Cplusplus>`_: Write in ISO Standard C++ (especially to support Windows, Linux and MacOS platforms).
- `P.5 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rp-compile-time>`_: Prefer compile-time checking to run-time checking.

**Implementation**

- `SF.1 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rs-file-suffix>`_: Use a .cpp suffix for code files and .h for interface files if your project doesn't already follow another convention.
- We modify this rule:

  - .h: C header files.
  - .hpp: C++ header files.

- `SF.5 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rs-consistency>`_: A .cpp file must include the .h file(s) that defines its interface.
- `SF.7 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rs-using-directive>`_: Don't put a using-directive in a header file.
- `SF.8 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rs-guards>`_: Use #include guards for all .h files.
- `SF.21 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rs-unnamed>`_: Don't use an unnamed (anonymous) namespace in a header.
- `SL.10 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rsl-arrays>`_: Prefer using STL array or vector instead of a C array.
- `C.9 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rc-private>`_: Minimize exposure of members.
- `F.3 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rf-single>`_: Keep functions short and simple.
- `F.21 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rf-out-multi>`_: To return multiple 'out' values, prefer returning a tuple.
- `R.1 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rr-raii>`_: Manage resources automatically using RAII (this includes unique_ptr & shared_ptr).
- `ES.11 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Res-auto>`_:  Use auto to avoid redundant repetition of type names.
- `ES.20 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Res-always>`_: Always initialize an object.
- `ES.23 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Res-list>`_: Prefer the {} initializer syntax.
- `ES.49 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Res-casts-named>`_: If you must use a cast, use a named cast.
- `CP.1 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#S-concurrency>`_: Assume that your code will run as part of a multi-threaded program.
- `I.2 <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Ri-global>`_: Avoid global variables.

**Format**

C and C++ code is formatted using clang-format.
To format a file, use

::

  /opt/rocm/hcc/bin/clang-format -style=file -i <file>

To format all files, run the following script in rocALUTION directory:

::

  #!/bin/bash

  find . -iname '*.h' \
  -o -iname '*.hpp' \
  -o -iname '*.cpp' \
  -o -iname '*.h.in' \
  -o -iname '*.hpp.in' \
  -o -iname '*.cpp.in' \
  -o -iname '*.cl' \
  | grep -v 'build' \
  | xargs -n 1 -P 8 -I{} /opt/rocm/hcc/bin/clang-format -style=file -i {}
