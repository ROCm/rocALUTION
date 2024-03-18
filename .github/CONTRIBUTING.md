<head>
  <meta charset="UTF-8">
  <meta name="description" content="Contributing to rocALUTION">
  <meta name="keywords" content="ROCm, contributing, rocALUTION">
</head>

# Contributing to rocALUTION #

AMD welcomes contributions to rocALUTION from the community. Whether those contributions are bug reports, bug fixes, documentation additions, performance notes, or other improvements, we value collaboration with our users. We can build better solutions together. Please follow these details to help ensure your contributions will be successfully accepted.

Our code contriubtion guidelines closely follow the model of [GitHub pull-requests](https://help.github.com/articles/using-pull-requests/).  This repository follows the [git flow](http://nvie.com/posts/a-successful-git-branching-model/) workflow, which dictates a /master branch where releases are cut, and a /develop branch which serves as an integration branch for new code.

## Issue Discussion ##

Please use the GitHub Issues tab to notify us of issues.

* Use your best judgement for issue creation. If your issue is already listed, upvote the issue and
  comment or post to provide additional details, such as how you reproduced this issue.
* If you're not sure if your issue is the same, err on the side of caution and file your issue.
  You can add a comment to include the issue number (and link) for the similar issue. If we evaluate
  your issue as being the same as the existing issue, we'll close the duplicate.
* If your issue doesn't exist, use the issue template to file a new issue.
  * When filing an issue, be sure to provide as much information as possible, including script output so
    we can collect information about your configuration. This helps reduce the time required to
    reproduce your issue.
  * Check your issue regularly, as we may require additional information to successfully reproduce the
    issue.
* You may also open an issue to ask questions to the maintainers about whether a proposed change
  meets the acceptance criteria, or to discuss an idea pertaining to the library.

## Acceptance Criteria ##

rocALUTION is a sparse linear algebra library with focus on exploring fine-grained parallelism on top of the AMD ROCm runtime and toolchains, targeting modern CPU and GPU platforms. Based on C++ and HIP, it provides a portable, generic and flexible design that allows seamless integration with other scientific software packages.

In rocALUTION we are interested in contributions that:
* Fix bugs, improve documentation, enhance testing, reduce complexity.
* Improve the performance of existing routines.
* Add missing functionality such as new multigrid solvers, iterative solvers, direct solvers, or preconditioners.
* Extending new or existing functionality to work with MPI or accelerators (such as GPU devices).

We encourage contributors to leverage the GitHub "Issues" tab to discuss possible additions they would like to add.

### Exceptions ###

rocALUTION places a heavy emphasis on being high performance. Because of this, contributions that add new routines (or that modify existing routines) must do so from the perspective that they offer high performance in relation to the hardware they are run on. Furthermore, all routines added to rocalution must have at a minimum a host solution as all routines must have the ability to fall back to a host solution if a GPU accelerator is not avaiable. Because compile times, binary sizes, and general library complexity are important considerations, we reserve the right to make decisions on whether a proposed routine is too niche or specialized to be worth including.

## Code Structure ##

The following is the structure of the rocALUTION library in the GitHub repository. A more detailed description of the directory structure can be found in the rocALUTION [documentation](https://rocm.docs.amd.com/projects/rocALUTION/en/latest/design/orga.html).

The `src/` directory contains the library source code. This is broken up into three sub-directories:
* `src/base`
* `src/solvers`
* `src/utils`

The `src/base` Contains source code related to rocALUTION's vector, matrix, and stencil operator types as well as classes related to parallel management. This directory is further broken up into:
* `src/base/hip` Contains HIP implementations of vector, matrix, and stencil operators.
* `src/base/host` Contains host implementations of vector, matrix, and stencil operators.

The `src/solvers` directory contains all the source code related to direct (`src/solvers/direct`), krylov (`src/solvers/krylov`), and multigrid solvers `src/solvers/multigrid`.

The `src/utils` directory contains source code related to logging, memory allocation, math and timing functions.

The `clients/` directory contains the testing and benchmarking code as well as all the samples demonstrating rocALUTION usage.

The `docs/` directory contains all of the documentation files.

## Coding Style ##

In general, follow the style of the surrounding code. C and C++ code is formatted using `clang-format`. Use the clang-format version installed with ROCm (found in the `/opt/rocm/llvm/bin` directory). Please do not use your system's built-in `clang-format`, as this is a different version that may result in incorrect results.

To format a file, use:

```
/opt/rocm/llvm/bin/clang-format -style=file -i <path-to-source-file>
```

To format all files, run the following script in rocALUTION directory:

```
#!/bin/bash
git ls-files -z *.cc *.cpp *.h *.hpp *.cl *.h.in *.hpp.in *.cpp.in | xargs -0 /opt/rocm/llvm/bin/clang-format  -style=file -i
```

Also, githooks can be installed to format the code per-commit:

```
./.githooks/install
```

## Pull Request Guidelines ##

When you create a pull request, you should target the default branch. Our current default branch is the **develop** branch, which serves as our integration branch.

By submitting a pull request, you acknowlege and agree with the CLA below:

Contribution License Agreement
1. The code I am contributing is mine, and I have the right to license it.
2. By submitting a pull request for this project I am granting you a license to distribute said code under the MIT License for the project.

### Deliverables ###

When raising a PR in rocALUTION here are some important things to include:

1. For each new file in the repository, Please include the licensing header
```
/* ************************************************************************
* Copyright (C) 20xx Advanced Micro Devices, Inc. All rights Reserved.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE.
*
* ************************************************************************ */
```
and adjust the date to the current year. When simply modifying a file, the date should automatically be updated pre-commit as long as the githook has been installed (./.githooks/install).

2. When adding a new routine, please make sure you are also adding appropriate testing code. These new unit tests should integrate within the existing [googletest framework](https://github.com/google/googletest/blob/master/googletest/docs/primer.md). This typically involves adding the following files:

* testing_<routine_name>.hpp file in the directory `clients/include/`
* test_<routine_name>.cpp file in directory `clients/tests/`

See existing tests for guidance when adding your own.

3. When modifiying an existing routine, add appropriate testing to test_<routine_name>.cpp file in directory `clients/tests/`.

4. Tests must have good code coverage.

5. At a minimum, rocALUTION must have a host solution for each direct, iterative, multigrid, or preconditioner. If you add a accelerator solution (say using HIP targetting GPU devices) please also add a fall back host solution.

6. Ensure code builds successfully. This includes making sure that the code can compile, that the code is properly formatted, and that all tests pass.

7. Do not break existing test cases

### Process ###

When a PR is raised targetting the develop branch in rocALUTION, CI will be automatically triggered. This will:

* Test that the PR passes static analysis (i.e ensure clang formatting rules have been followed).
* Test that the documentation can be properly built
* Ensure that the PR compiles on different OS and GPU device architecture combinations
* Ensure that all tests pass on different OS and GPU device architecture combinations

Feel free to ask questions on your PR regarding any CI failures you encounter.

* Reviewers are listed in the CODEOWNERS file
