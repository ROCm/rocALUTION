/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once
#ifndef TESTING_UTILITY_HPP
#define TESTING_UTILITY_HPP

#include <string>
#include <algorithm>

/* ============================================================================================ */
/*! \brief  Generate 2D laplacian on unit square in CSR format */
template <typename T>
int gen_2d_laplacian(int ndim, int** rowptr, int** col, T** val)
{
    if(ndim == 0)
    {
        return 0;
    }

    int n       = ndim * ndim;
    int nnz_mat = n * 5 - ndim * 4;

    *rowptr = new int[n + 1];
    *col    = new int[nnz_mat];
    *val    = new T[nnz_mat];

    int nnz = 0;

    // Fill local arrays
    for(int i = 0; i < ndim; ++i)
    {
        for(int j = 0; j < ndim; ++j)
        {
            int idx = i * ndim + j;
            (*rowptr)[idx]       = nnz;
            // if no upper boundary element, connect with upper neighbor
            if(i != 0)
            {
                (*col)[nnz] = idx - ndim;
                (*val)[nnz] = static_cast<T>(-1);
                ++nnz;
            }
            // if no left boundary element, connect with left neighbor
            if(j != 0)
            {
                (*col)[nnz] = idx - 1;
                (*val)[nnz] = static_cast<T>(-1);
                ++nnz;
            }
            // element itself
            (*col)[nnz] = idx;
            (*val)[nnz] = static_cast<T>(4);
            ++nnz;
            // if no right boundary element, connect with right neighbor
            if(j != ndim - 1)
            {
                (*col)[nnz] = idx + 1;
                (*val)[nnz] = static_cast<T>(-1);
                ++nnz;
            }
            // if no lower boundary element, connect with lower neighbor
            if(i != ndim - 1)
            {
                (*col)[nnz] = idx + ndim;
                (*val)[nnz] = static_cast<T>(-1);
                ++nnz;
            }
        }
    }
    (*rowptr)[n] = nnz;

    return n;
}

/* ============================================================================================ */

/*! \brief Class used to parse command arguments in both client & gtest   */

// has to compile with option "-std=c++11", and this rocsparse library uses c++11 everywhere
// c++11 allows intilization of member of a struct

class Arguments
{
    public:

    // MPI variables
    int rank         = 0;
    int dev_per_node = 1;

    // OpenMP variables
    int omp_nthreads  = 8;
    int omp_affinity  = true;
    int omp_threshold = 50000;

    // Accelerator variables
    int dev     = 0;
    int use_acc = true;

    // Structure variables
    int size       = 100;
    int index      = 50;
    int chunk_size = 20;

    // Computation variables
    double alpha = 1.0;
    double beta  = 0.0;
    double gamma = 0.0;

    // Solver variables
    std::string solver = "";
    std::string precond = "";

    unsigned int format;

    Arguments& operator=(const Arguments& rhs)
    {
        this->rank         = rhs.rank;
        this->dev_per_node = rhs.dev_per_node;

        this->omp_nthreads  = rhs.omp_nthreads;
        this->omp_affinity  = rhs.omp_affinity;
        this->omp_threshold = rhs.omp_threshold;

        this->dev     = rhs.dev;
        this->use_acc = rhs.use_acc;

        this->size       = rhs.size;
        this->index      = rhs.index;
        this->chunk_size = rhs.chunk_size;

        this->alpha = rhs.alpha;
        this->beta  = rhs.beta;
        this->gamma = rhs.gamma;

        this->solver  = rhs.solver;
        this->precond = rhs.precond;

        this->format = rhs.format;

        return *this;
    }
};

#endif // TESTING_UTILITY_HPP
