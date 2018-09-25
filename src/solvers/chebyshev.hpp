/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#ifndef ROCALUTION_KRYLOV_CHEBYSHEV_HPP_
#define ROCALUTION_KRYLOV_CHEBYSHEV_HPP_

#include "solver.hpp"

#include <vector>

namespace rocalution {

/** \ingroup solver_module
  * \class Chebyshev
  * \brief Chebyshev Iteration Scheme
  * \details
  * The Chebyshev Iteration scheme (also known as acceleration scheme) is similar to the
  * CG method but requires minimum and maximum eigenvalues of the operator, see "Barrett,
  * R., Berry, M., Chan, T. F., Demmel, J., Donato, J., Dongarra, J., Eijkhout, V.,
  * Pozo, R., Romine, C., and der Vorst, H. V. Templates for the Solution of Linear
  * Systems: Building Blocks for Iterative Methods, 2 ed. SIAM, Philadelphia, PA, 1994."
  * for details.
  * 
  * \tparam OperatorType - can be LocalMatrix, GlobalMatrix or LocalStencil
  * \tparam VectorType - can be LocalVector or GlobalVector
  * \tparam ValueType - can be float, double, std::complex<float> or std::complex<double>
  */
template <class OperatorType, class VectorType, typename ValueType>
class Chebyshev : public IterativeLinearSolver<OperatorType, VectorType, ValueType>
{
    public:
    Chebyshev();
    virtual ~Chebyshev();

    virtual void Print(void) const;

    /** \brief Set the minimum and maximum eigenvalues of the operator */
    void Set(ValueType lambda_min, ValueType lambda_max);

    virtual void Build(void);
    virtual void ReBuildNumeric(void);
    virtual void Clear(void);

    protected:
    virtual void SolveNonPrecond_(const VectorType& rhs, VectorType* x);
    virtual void SolvePrecond_(const VectorType& rhs, VectorType* x);

    virtual void PrintStart_(void) const;
    virtual void PrintEnd_(void) const;

    virtual void MoveToHostLocalData_(void);
    virtual void MoveToAcceleratorLocalData_(void);

    private:
    bool init_lambda_;
    ValueType lambda_min_, lambda_max_;

    VectorType r_, z_;
    VectorType p_;
};

} // namespace rocalution

#endif // ROCALUTION_KRYLOV_CHEBYSHEV_HPP_
