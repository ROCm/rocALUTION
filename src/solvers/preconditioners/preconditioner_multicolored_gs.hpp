/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 *
 * ************************************************************************ */

#ifndef ROCALUTION_PRECONDITIONER_MULTICOLORED_GS_HPP_
#define ROCALUTION_PRECONDITIONER_MULTICOLORED_GS_HPP_

#include "../solver.hpp"
#include "preconditioner.hpp"
#include "preconditioner_multicolored.hpp"
#include "../../base/local_vector.hpp"

#include <vector>

namespace rocalution {

/** \ingroup precond_module
  * \class MultiColoredSGS
  * \brief Multi-Colored Symmetric Gauss-Seidel / SSOR Preconditioner
  * \details
  * The Multi-Colored Symmetric Gauss-Seidel / SSOR preconditioner is based on the
  * splitting of the original matrix. Higher parallelism in solving the forward and
  * backward substitution is obtained by performing a multi-colored decomposition.
  * Details on the Symmetric Gauss-Seidel / SSOR algorithm can be found in the SGS
  * preconditioner.
  *
  * \tparam OperatorType - can be LocalMatrix
  * \tparam VectorType - can be LocalVector
  * \tparam ValueType - can be float, double, std::complex<float> or std::complex<double>
  */
template <class OperatorType, class VectorType, typename ValueType>
class MultiColoredSGS : public MultiColored<OperatorType, VectorType, ValueType>
{
    public:
    MultiColoredSGS();
    virtual ~MultiColoredSGS();

    virtual void Print(void) const;

    virtual void ReBuildNumeric(void);

    /** \brief Set the relaxation parameter for the SOR/SSOR scheme */
    void SetRelaxation(ValueType omega);

    protected:
    virtual void PostAnalyse_(void);

    virtual void SolveL_(void);
    virtual void SolveD_(void);
    virtual void SolveR_(void);
    virtual void Solve_(const VectorType& rhs, VectorType* x);

    /** \brief Relaxation parameter */
    ValueType omega_;
};

/** \ingroup precond_module
  * \class MultiColoredGS
  * \brief Multi-Colored Gauss-Seidel / SOR Preconditioner
  * \details
  * The Multi-Colored Symmetric Gauss-Seidel / SOR preconditioner is based on the
  * splitting of the original matrix. Higher parallelism in solving the forward
  * substitution is obtained by performing a multi-colored decomposition. Details on the
  * Gauss-Seidel / SOR algorithm can be found in the GS preconditioner.
  *
  * \tparam OperatorType - can be LocalMatrix
  * \tparam VectorType - can be LocalVector
  * \tparam ValueType - can be float, double, std::complex<float> or std::complex<double>
  */
template <class OperatorType, class VectorType, typename ValueType>
class MultiColoredGS : public MultiColoredSGS<OperatorType, VectorType, ValueType>
{
    public:
    MultiColoredGS();
    virtual ~MultiColoredGS();

    virtual void Print(void) const;

    protected:
    virtual void PostAnalyse_(void);

    virtual void SolveL_(void);
    virtual void SolveD_(void);
    virtual void SolveR_(void);
    virtual void Solve_(const VectorType& rhs, VectorType* x);
};

} // namespace rocalution

#endif // ROCALUTION_PRECONDITIONER_MULTICOLORED_GS_HPP_
