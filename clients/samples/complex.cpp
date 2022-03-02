/* ************************************************************************
 * Copyright (c) 2020 Advanced Micro Devices, Inc.
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

#include <complex>
#include <cstdlib>
#include <iostream>
#include <rocalution/rocalution.hpp>

using namespace rocalution;

int main(int argc, char* argv[])
{
    // Check command line parameters
    if(argc == 1)
    {
        std::cerr << argv[0] << " <matrix> [Num threads]" << std::endl;
        exit(1);
    }

    // Initialize rocALUTION
    init_rocalution();

    // Check command line parameters for number of OMP threads
    if(argc > 2)
    {
        set_omp_threads_rocalution(atoi(argv[2]));
    }

    // Print rocALUTION info
    info_rocalution();

    // rocALUTION objects
    LocalVector<std::complex<double>> x;
    LocalVector<std::complex<double>> rhs;
    LocalVector<std::complex<double>> e;
    LocalMatrix<std::complex<double>> mat;

    // Read matrix from MTX file
    mat.ReadFileMTX(std::string(argv[1]));

    // Move objects to accelerator
    mat.MoveToAccelerator();
    x.MoveToAccelerator();
    rhs.MoveToAccelerator();

    // Allocate vectors
    x.Allocate("x", mat.GetN());
    rhs.Allocate("rhs", mat.GetM());
    e.Allocate("e", mat.GetN());

    // Initialize e with some values
    for(int i = 0; i < mat.GetN(); ++i)
        e[i] = std::complex<double>(1.0, -1.0);

    e.MoveToAccelerator();

    // Linear Solver
    IDR<LocalMatrix<std::complex<double>>, LocalVector<std::complex<double>>, std::complex<double>>
        ls;

    // Preconditioner
    Jacobi<LocalMatrix<std::complex<double>>,
           LocalVector<std::complex<double>>,
           std::complex<double>>
        p;

    // Initialize rhs such that A 1 = rhs
    mat.Apply(e, &rhs);

    // Initial zero guess
    x.Zeros();

    // Set solver operator
    ls.SetOperator(mat);
    // Set solver preconditioner
    ls.SetPreconditioner(p);

    // Build solver
    ls.Build();

    // Verbosity output
    ls.Verbose(1);

    // Print matrix info
    mat.Info();

    // Start time measurement
    double tick, tack;
    tick = rocalution_time();

    // Solve A x = rhs
    ls.Solve(rhs, &x);

    // Stop time measurement
    tack = rocalution_time();
    std::cout << "Solver execution:" << (tack - tick) / 1e6 << " sec" << std::endl;

    // Clear solver
    ls.Clear();

    // Compute error L2 norm
    e.ScaleAdd(-1.0, x);
    std::complex<double> error = e.Norm();
    std::cout << "||e - x||_2 = " << error << std::endl;

    // Stop rocALUTION platform
    stop_rocalution();

    return 0;
}
