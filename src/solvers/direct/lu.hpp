/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#ifndef ROCALUTION_DIRECT_LU_HPP_
#define ROCALUTION_DIRECT_LU_HPP_

#include "../solver.hpp"

namespace rocalution {

/** \ingroup solver_module
  * \class LU
  * \brief LU Decomposition
  * \details
  * Lower-Upper Decomposition factors a given square matrix into lower and upper
  * triangular matrix, such that \f$A = LU\f$.
  *
  * \tparam OperatorType - can be LocalMatrix
  * \tparam VectorType - can be LocalVector
  * \tparam ValueType - can be float, double, std::complex<float> or std::complex<double>
  */
template <class OperatorType, class VectorType, typename ValueType>
class LU : public DirectLinearSolver<OperatorType, VectorType, ValueType>
{
    public:
    LU();
    virtual ~LU();

    virtual void Print(void) const;

    virtual void Build(void);
    virtual void Clear(void);

    protected:
    virtual void Solve_(const VectorType& rhs, VectorType* x);

    virtual void PrintStart_(void) const;
    virtual void PrintEnd_(void) const;

    virtual void MoveToHostLocalData_(void);
    virtual void MoveToAcceleratorLocalData_(void);

    private:
    OperatorType lu_;
};

} // namespace rocalution

#endif // ROCALUTION_DIRECT_LU_HPP_
