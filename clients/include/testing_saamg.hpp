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

#pragma once
#ifndef TESTING_SAAMG_HPP
#define TESTING_SAAMG_HPP

#include "utility.hpp"

#include <rocalution.hpp>

using namespace rocalution;

static bool check_residual(float res)
{
    return (res < 1e-2f);
}

static bool check_residual(double res)
{
    return (res < 1e-5);
}

template <typename T>
bool testing_saamg(Arguments argus)
{
    int          ndim      = argus.size;
    int          pre_iter  = argus.pre_smooth;
    int          post_iter = argus.post_smooth;
    std::string  smoother  = argus.smoother;
    unsigned int format    = argus.format;
    int          cycle     = argus.cycle;
    bool         scaling   = argus.ordering;

    // Initialize rocALUTION platform
    set_device_rocalution(device);
    init_rocalution();

    // rocALUTION structures
    LocalMatrix<T> A;
    LocalVector<T> x;
    LocalVector<T> b;
    LocalVector<T> e;

    // Generate A
    int* csr_ptr = NULL;
    int* csr_col = NULL;
    T*   csr_val = NULL;

    int nrow = gen_2d_laplacian(ndim, &csr_ptr, &csr_col, &csr_val);
    int nnz  = csr_ptr[nrow];

    A.SetDataPtrCSR(&csr_ptr, &csr_col, &csr_val, "A", nnz, nrow, nrow);

    // Move data to accelerator
    A.MoveToAccelerator();
    x.MoveToAccelerator();
    b.MoveToAccelerator();
    e.MoveToAccelerator();

    // Allocate x, b and e
    x.Allocate("x", A.GetN());
    b.Allocate("b", A.GetM());
    e.Allocate("e", A.GetN());

    // b = A * 1
    e.Ones();
    A.Apply(e, &b);

    // Random initial guess
    x.SetRandomUniform(12345ULL, -4.0, 6.0);

    // Solver
    FCG<LocalMatrix<T>, LocalVector<T>, T> ls;

    // AMG
    SAAMG<LocalMatrix<T>, LocalVector<T>, T> p;

    // Setup SAAMG
    p.SetCoarsestLevel(200);
    p.SetCycle(cycle);
    p.SetOperator(A);
    p.SetManualSmoothers(true);
    p.SetManualSolver(true);
    p.SetScaling(scaling);
    p.BuildHierarchy();

    // Get number of hierarchy levels
    int levels = p.GetNumLevels();

    // Coarse grid solver
    FCG<LocalMatrix<T>, LocalVector<T>, T> cgs;
    cgs.Verbose(0);

    // Smoother for each level
    IterativeLinearSolver<LocalMatrix<T>, LocalVector<T>, T>** sm
        = new IterativeLinearSolver<LocalMatrix<T>, LocalVector<T>, T>*[levels - 1];

    for(int i = 0; i < levels - 1; ++i)
    {
        FixedPoint<LocalMatrix<T>, LocalVector<T>, T>* fp
            = new FixedPoint<LocalMatrix<T>, LocalVector<T>, T>;
        sm[i] = fp;

        Preconditioner<LocalMatrix<T>, LocalVector<T>, T>* smooth;

        if(smoother == "FSAI")
            smooth = new FSAI<LocalMatrix<T>, LocalVector<T>, T>;
        else if(smoother == "SPAI")
            smooth = new SPAI<LocalMatrix<T>, LocalVector<T>, T>;
        else
            return false;

        sm[i]->SetPreconditioner(*smooth);
        sm[i]->Verbose(0);
    }

    p.SetSmoother(sm);
    p.SetSolver(cgs);
    p.SetSmootherPreIter(pre_iter);
    p.SetSmootherPostIter(post_iter);
    p.SetOperatorFormat(format);
    p.InitMaxIter(1);
    p.Verbose(0);

    ls.Verbose(0);
    ls.SetOperator(A);
    ls.SetPreconditioner(p);

    ls.Init(1e-8, 0.0, 1e+8, 10000);
    ls.Build();

    // Matrix format
    A.ConvertTo(format);

    ls.Solve(b, &x);

    // Verify solution
    x.ScaleAdd(-1.0, e);
    T nrm2 = x.Norm();

    bool success = check_residual(nrm2);

    // Clean up
    ls.Clear(); // TODO

    // Stop rocALUTION platform
    stop_rocalution();

    return success;
}

#endif // TESTING_SAAMG_HPP
