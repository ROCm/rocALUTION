/* ************************************************************************
 * Copyright (c) 2018 Advanced Micro Devices, Inc.
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

#include <cstdlib>
#include <iostream>
#include <rocalution.hpp>

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
    LocalVector<double> x;
    LocalVector<double> rhs;
    LocalVector<double> e;
    LocalMatrix<double> mat;

    // Read matrix from MTX file
    mat.ReadFileMTX(std::string(argv[1]));

    // Allocate vectors
    x.Allocate("x", mat.GetN());
    rhs.Allocate("rhs", mat.GetM());
    e.Allocate("e", mat.GetN());

    // Initialize rhs such that A 1 = rhs
    e.Ones();
    mat.Apply(e, &rhs);

    // Initial zero guess
    x.Zeros();

    // Linear Solver
    GMRES<LocalMatrix<double>, LocalVector<double>, double> ls;

    // Preconditioner
    AS<LocalMatrix<double>, LocalVector<double>, double> p; // Additive Schwarz
    //  RAS<LocalMatrix<double>, LocalVector<double>, double > p; // Restricted Additive Schwarz

    // Second level preconditioners
    Solver<LocalMatrix<double>, LocalVector<double>, double>** p2;

    int n = 2;
    p2    = new Solver<LocalMatrix<double>, LocalVector<double>, double>*[n];

    for(int i = 0; i < n; ++i)
    {
        MultiColoredILU<LocalMatrix<double>, LocalVector<double>, double>* mc;
        mc    = new MultiColoredILU<LocalMatrix<double>, LocalVector<double>, double>;
        p2[i] = mc;
    }

    double tick, tack;

    // Initialize preconditioner
    p.Set(n, 4, p2);

    // Set solver operator
    ls.SetOperator(mat);
    // Set solver preconditioner
    ls.SetPreconditioner(p);

    // Verbosity output
    //  ls.Verbose(2);

    // Build solver
    ls.Build();

    // Move solver to the accelerator
    mat.MoveToAccelerator();
    x.MoveToAccelerator();
    rhs.MoveToAccelerator();
    e.MoveToAccelerator();
    ls.MoveToAccelerator();

    // Print matrix info
    mat.Info();

    // Start solving time measurement
    tick = rocalution_time();

    // Solve A x = rhs
    ls.Solve(rhs, &x);

    // Stop solving time measurement
    tack = rocalution_time();
    std::cout << "Solver took: " << (tack - tick) / 1e6 << " sec" << std::endl;

    // Clear the solver
    ls.Clear();

    // Free all allocated data
    for(int i = 0; i < n; ++i)
    {
        delete p2[i];
        p2[i] = NULL;
    }

    delete[] p2;
    p2 = NULL;

    // Compute error L2 norm
    e.ScaleAdd(-1.0, x);
    double error = e.Norm();
    std::cout << "||e - x||_2 = " << error << std::endl;

    // Stop rocALUTION platform
    stop_rocalution();

    return 0;
}
