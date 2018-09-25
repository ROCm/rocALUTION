/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#ifndef ROCALUTION_MIXED_PRECISION_HPP_
#define ROCALUTION_MIXED_PRECISION_HPP_

#include "solver.hpp"

namespace rocalution {

/** \ingroup solver_module
  * \class MixedPrecisionDC
  * \brief Mixed-Precision Defect Correction Scheme
  * \details
  * The Mixed-Precision solver is based on a defect-correction scheme. The current
  * implementation of the library is using host based correction in double precision and
  * accelerator computation in single precision. The solver is implemeting the scheme
  * \f[
  *   x_{k+1} = x_{k} + A^{-1} r_{k},
  * \f]
  * where the computation of the residual \f$r_{k} = b - Ax_{k}\f$ and the update
  * \f$x_{k+1} = x_{k} + d_{k}\f$ are performed on the host in double precision. The
  * computation of the residual system \f$Ad_{k} = r_{k}\f$ is performed on the
  * accelerator in single precision. In addition to the setup functions of the iterative
  * solver, the user need to specify the inner (\f$Ad_{k} = r_{k}\f$) solver.
  * 
  * \tparam OperatorTypeH - can be LocalMatrix
  * \tparam VectorTypeH - can be LocalVector
  * \tparam ValueTypeH - can be double
  * \tparam OperatorTypeL - can be LocalMatrix
  * \tparam VectorTypeL - can be LocalVector
  * \tparam ValueTypeL - can be float
  */
template <class OperatorTypeH,
          class VectorTypeH,
          typename ValueTypeH,
          class OperatorTypeL,
          class VectorTypeL,
          typename ValueTypeL>
class MixedPrecisionDC : public IterativeLinearSolver<OperatorTypeH, VectorTypeH, ValueTypeH>
{
    public:
    MixedPrecisionDC();
    virtual ~MixedPrecisionDC();

    virtual void Print(void) const;

    /** \brief Set the inner solver for \f$Ad_{k} = r_{k}\f$ */
    void Set(Solver<OperatorTypeL, VectorTypeL, ValueTypeL>& Solver_L);

    virtual void Build(void);
    virtual void ReBuildNumeric(void);
    virtual void Clear(void);

    protected:
    virtual void SolveNonPrecond_(const VectorTypeH& rhs, VectorTypeH* x);
    virtual void SolvePrecond_(const VectorTypeH& rhs, VectorTypeH* x);

    virtual void PrintStart_(void) const;
    virtual void PrintEnd_(void) const;

    virtual void MoveToHostLocalData_(void);
    virtual void MoveToAcceleratorLocalData_(void);

    private:
    Solver<OperatorTypeL, VectorTypeL, ValueTypeL>* Solver_L_;

    VectorTypeH r_h_;
    VectorTypeL r_l_;

    VectorTypeH* x_h_;
    VectorTypeL d_l_;
    VectorTypeH d_h_;

    const OperatorTypeH* op_h_;
    OperatorTypeL* op_l_;
};

} // namespace rocalution

#endif // ROCALUTION_MIXED_PRECISION_HPP_
