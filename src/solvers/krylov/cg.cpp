#include "../../utils/def.hpp"
#include "cg.hpp"
#include "../iter_ctrl.hpp"

#include "../../base/local_matrix.hpp"
#include "../../base/local_stencil.hpp"
#include "../../base/local_vector.hpp"

#include "../../base/global_matrix.hpp"
#include "../../base/global_vector.hpp"

#include "../../utils/log.hpp"
#include "../../utils/math_functions.hpp"

#include <math.h>
#include <complex>

namespace rocalution {

template <class OperatorType, class VectorType, typename ValueType>
CG<OperatorType, VectorType, ValueType>::CG()
{
    LOG_DEBUG(this, "CG::CG()", "default constructor");
}

template <class OperatorType, class VectorType, typename ValueType>
CG<OperatorType, VectorType, ValueType>::~CG()
{
    LOG_DEBUG(this, "CG::~CG()", "destructor");

    this->Clear();
}

template <class OperatorType, class VectorType, typename ValueType>
void CG<OperatorType, VectorType, ValueType>::Print(void) const
{
    if(this->precond_ == NULL)
    {
        LOG_INFO("CG solver");
    }
    else
    {
        LOG_INFO("PCG solver, with preconditioner:");
        this->precond_->Print();
    }
}

template <class OperatorType, class VectorType, typename ValueType>
void CG<OperatorType, VectorType, ValueType>::PrintStart_(void) const
{
    if(this->precond_ == NULL)
    {
        LOG_INFO("CG (non-precond) linear solver starts");
    }
    else
    {
        LOG_INFO("PCG solver starts, with preconditioner:");
        this->precond_->Print();
    }
}

template <class OperatorType, class VectorType, typename ValueType>
void CG<OperatorType, VectorType, ValueType>::PrintEnd_(void) const
{
    if(this->precond_ == NULL)
    {
        LOG_INFO("CG (non-precond) ends");
    }
    else
    {
        LOG_INFO("PCG ends");
    }
}

template <class OperatorType, class VectorType, typename ValueType>
void CG<OperatorType, VectorType, ValueType>::Build(void)
{
    LOG_DEBUG(this, "CG::Build()", this->build_ << " #*# begin");

    if(this->build_ == true)
    {
        this->Clear();
    }

    assert(this->build_ == false);

    this->build_ = true;

    assert(this->op_ != NULL);
    assert(this->op_->GetM() == this->op_->GetN());
    assert(this->op_->GetM() > 0);

    if(this->precond_ != NULL)
    {
        this->precond_->SetOperator(*this->op_);

        this->precond_->Build();

        this->z_.CloneBackend(*this->op_);
        this->z_.Allocate("z", this->op_->GetM());
    }

    this->r_.CloneBackend(*this->op_);
    this->r_.Allocate("r", this->op_->GetM());

    this->p_.CloneBackend(*this->op_);
    this->p_.Allocate("p", this->op_->GetM());

    this->q_.CloneBackend(*this->op_);
    this->q_.Allocate("q", this->op_->GetM());

    LOG_DEBUG(this, "CG::Build()", this->build_ << " #*# end");
}

template <class OperatorType, class VectorType, typename ValueType>
void CG<OperatorType, VectorType, ValueType>::BuildMoveToAcceleratorAsync(void)
{
    LOG_DEBUG(this, "CG::BuildMoveToAcceleratorAsync()", this->build_ << " #*# begin");

    if(this->build_ == true)
    {
        this->Clear();
    }

    assert(this->build_ == false);

    this->build_ = true;

    assert(this->op_ != NULL);
    assert(this->op_->GetM() == this->op_->GetN());
    assert(this->op_->GetM() > 0);

    if(this->precond_ != NULL)
    {
        this->precond_->SetOperator(*this->op_);

        this->precond_->BuildMoveToAcceleratorAsync();

        this->z_.CloneBackend(*this->op_);
        this->z_.Allocate("z", this->op_->GetM());
        this->z_.MoveToAcceleratorAsync();
    }

    this->r_.CloneBackend(*this->op_);
    this->r_.Allocate("r", this->op_->GetM());
    this->r_.MoveToAcceleratorAsync();

    this->p_.CloneBackend(*this->op_);
    this->p_.Allocate("p", this->op_->GetM());
    this->p_.MoveToAcceleratorAsync();

    this->q_.CloneBackend(*this->op_);
    this->q_.Allocate("q", this->op_->GetM());
    this->q_.MoveToAcceleratorAsync();

    LOG_DEBUG(this, "CG::BuildMoveToAcceleratorAsync()", this->build_ << " #*# end");
}

template <class OperatorType, class VectorType, typename ValueType>
void CG<OperatorType, VectorType, ValueType>::Sync(void)
{
    LOG_DEBUG(this, "CG::Sync()", this->build_ << " #*# begin");

    if(this->precond_ != NULL)
    {
        this->precond_->Sync();
        this->z_.Sync();
    }

    this->r_.Sync();
    this->p_.Sync();
    this->q_.Sync();

    LOG_DEBUG(this, "CG::Sync()", this->build_ << " #*# end");
}
template <class OperatorType, class VectorType, typename ValueType>
void CG<OperatorType, VectorType, ValueType>::Clear(void)
{
    LOG_DEBUG(this, "CG::Clear()", this->build_);

    if(this->build_ == true)
    {
        if(this->precond_ != NULL)
        {
            this->precond_->Clear();
            this->precond_ = NULL;
        }

        this->r_.Clear();
        this->z_.Clear();
        this->p_.Clear();
        this->q_.Clear();

        this->iter_ctrl_.Clear();

        this->build_ = false;
    }
}

template <class OperatorType, class VectorType, typename ValueType>
void CG<OperatorType, VectorType, ValueType>::ReBuildNumeric(void)
{
    LOG_DEBUG(this, "CG::ReBuildNumeric()", this->build_);

    if(this->build_ == true)
    {
        this->r_.Zeros();
        this->z_.Zeros();
        this->p_.Zeros();
        this->q_.Zeros();

        this->iter_ctrl_.Clear();

        if(this->precond_ != NULL)
        {
            this->precond_->ReBuildNumeric();
        }
    }
    else
    {
        this->Build();
    }
}

template <class OperatorType, class VectorType, typename ValueType>
void CG<OperatorType, VectorType, ValueType>::MoveToHostLocalData_(void)
{
    LOG_DEBUG(this, "CG::MoveToHostLocalData_()", this->build_);

    if(this->build_ == true)
    {
        this->r_.MoveToHost();
        this->p_.MoveToHost();
        this->q_.MoveToHost();

        if(this->precond_ != NULL)
        {
            this->z_.MoveToHost();
            this->precond_->MoveToHost();
        }
    }
}

template <class OperatorType, class VectorType, typename ValueType>
void CG<OperatorType, VectorType, ValueType>::MoveToAcceleratorLocalData_(void)
{
    LOG_DEBUG(this, "CG::MoveToAcceleratorLocalData_()", this->build_);

    if(this->build_ == true)
    {
        this->r_.MoveToAccelerator();
        this->p_.MoveToAccelerator();
        this->q_.MoveToAccelerator();

        if(this->precond_ != NULL)
        {
            this->z_.MoveToAccelerator();
            this->precond_->MoveToAccelerator();
        }
    }
}

// TODO
// re-orthogonalization and
// residual - re-computed % iter
template <class OperatorType, class VectorType, typename ValueType>
void CG<OperatorType, VectorType, ValueType>::SolveNonPrecond_(const VectorType& rhs, VectorType* x)
{
    LOG_DEBUG(this, "CG::SolveNonPrecond_()", " #*# begin");

    assert(x != NULL);
    assert(x != &rhs);
    assert(this->op_ != NULL);
    assert(this->precond_ == NULL);
    assert(this->build_ == true);

    const OperatorType* op = this->op_;

    VectorType* r = &this->r_;
    VectorType* p = &this->p_;
    VectorType* q = &this->q_;

    ValueType alpha, beta;
    ValueType rho, rho_old;

    // Initial residual = b - Ax
    op->Apply(*x, r);
    r->ScaleAdd(static_cast<ValueType>(-1), rhs);

    // Initial residual norm |b-Ax0|
    ValueType res_norm = this->Norm(*r);
    // Initial residual norm |b|
    //    ValueType res_norm = this->Norm(rhs);

    if(this->iter_ctrl_.InitResidual(rocalution_abs(res_norm)) == false)
    {
        LOG_DEBUG(this, "CG::SolveNonPrecond_()", " #*# end");
        return;
    }

    // p = r
    p->CopyFrom(*r);

    // rho = (r,r)
    rho = r->DotNonConj(*r);

    while(true)
    {
        // q=Ap
        op->Apply(*p, q);

        // alpha = rho / (p,q)
        alpha = rho / p->DotNonConj(*q);

        // x = x + alpha*p
        x->AddScale(*p, alpha);

        // r = r - alpha*q
        r->AddScale(*q, ValueType(-1.0) * alpha);

        // Check convergence
        res_norm = this->Norm(*r);
        if(this->iter_ctrl_.CheckResidual(rocalution_abs(res_norm), this->index_))
        {
            break;
        }

        // rho = (r,r)
        rho_old = rho;
        rho     = r->DotNonConj(*r);

        // p = beta*p + r
        beta = rho / rho_old;
        p->ScaleAdd(beta, *r);
    }

    LOG_DEBUG(this, "CG::SolveNonPrecond_()", " #*# end");
}

template <class OperatorType, class VectorType, typename ValueType>
void CG<OperatorType, VectorType, ValueType>::SolvePrecond_(const VectorType& rhs, VectorType* x)
{
    LOG_DEBUG(this, "CG::SolvePrecond_()", " #*# begin");

    assert(x != NULL);
    assert(x != &rhs);
    assert(this->op_ != NULL);
    assert(this->precond_ != NULL);
    assert(this->build_ == true);

    const OperatorType* op = this->op_;

    VectorType* r = &this->r_;
    VectorType* z = &this->z_;
    VectorType* p = &this->p_;
    VectorType* q = &this->q_;

    ValueType alpha, beta;
    ValueType rho, rho_old;

    // Initial residual = b - Ax
    op->Apply(*x, r);
    r->ScaleAdd(static_cast<ValueType>(-1), rhs);

    // Initial residual norm |b-Ax0|
    ValueType res_norm = this->Norm(*r);
    // Initial residual norm |b|
    //    ValueType res_norm = this->Norm(rhs);

    // |b - Ax0|
    if(this->iter_ctrl_.InitResidual(rocalution_abs(res_norm)) == false)
    {
        LOG_DEBUG(this, "CG::SolvePrecond_()", " #*# end");
        return;
    }

    // Solve Mz=r
    this->precond_->SolveZeroSol(*r, z);

    // p = z
    p->CopyFrom(*z);

    // rho = (r,z)
    rho = r->DotNonConj(*z);

    while(true)
    {
        // q=Ap
        op->Apply(*p, q);

        // alpha = rho / (p,q)
        alpha = rho / p->DotNonConj(*q);

        // x = x + alpha*p
        x->AddScale(*p, alpha);

        // r = r - alpha*q
        r->AddScale(*q, -alpha);

        // Check convergence
        res_norm = this->Norm(*r);
        if(this->iter_ctrl_.CheckResidual(rocalution_abs(res_norm), this->index_))
        {
            break;
        }

        // Solve Mz=r
        this->precond_->SolveZeroSol(*r, z);

        // rho = (r,z)
        rho_old = rho;
        rho     = r->DotNonConj(*z);

        // p = beta*p + z
        beta = rho / rho_old;
        p->ScaleAdd(beta, *z);
    }

    LOG_DEBUG(this, "CG::SolvePrecond_()", " #*# end");
}

template class CG<LocalMatrix<double>, LocalVector<double>, double>;
template class CG<LocalMatrix<float>, LocalVector<float>, float>;
#ifdef SUPPORT_COMPLEX
template class CG<LocalMatrix<std::complex<double>>,
                  LocalVector<std::complex<double>>,
                  std::complex<double>>;
template class CG<LocalMatrix<std::complex<float>>,
                  LocalVector<std::complex<float>>,
                  std::complex<float>>;
#endif

template class CG<GlobalMatrix<double>, GlobalVector<double>, double>;
template class CG<GlobalMatrix<float>, GlobalVector<float>, float>;
#ifdef SUPPORT_COMPLEX
template class CG<GlobalMatrix<std::complex<double>>,
                  GlobalVector<std::complex<double>>,
                  std::complex<double>>;
template class CG<GlobalMatrix<std::complex<float>>,
                  GlobalVector<std::complex<float>>,
                  std::complex<float>>;
#endif

template class CG<LocalStencil<double>, LocalVector<double>, double>;
template class CG<LocalStencil<float>, LocalVector<float>, float>;
#ifdef SUPPORT_COMPLEX
template class CG<LocalStencil<std::complex<double>>,
                  LocalVector<std::complex<double>>,
                  std::complex<double>>;
template class CG<LocalStencil<std::complex<float>>,
                  LocalVector<std::complex<float>>,
                  std::complex<float>>;
#endif

} // namespace rocalution
