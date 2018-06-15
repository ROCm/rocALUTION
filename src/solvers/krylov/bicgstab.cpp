#include "../../utils/def.hpp"
#include "bicgstab.hpp"
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
#include <limits>

namespace rocalution {

template <class OperatorType, class VectorType, typename ValueType>
BiCGStab<OperatorType, VectorType, ValueType>::BiCGStab()
{
    LOG_DEBUG(this, "BiCGStab::BiCGStab()", "default constructor");
}

template <class OperatorType, class VectorType, typename ValueType>
BiCGStab<OperatorType, VectorType, ValueType>::~BiCGStab()
{
    LOG_DEBUG(this, "BiCGStab::~BiCGStab()", "destructor");

    this->Clear();
}

template <class OperatorType, class VectorType, typename ValueType>
void BiCGStab<OperatorType, VectorType, ValueType>::Print(void) const
{
    if(this->precond_ == NULL)
    {
        LOG_INFO("BiCGStab solver");
    }
    else
    {
        LOG_INFO("PBiCGStab solver, with preconditioner:");
        this->precond_->Print();
    }
}

template <class OperatorType, class VectorType, typename ValueType>
void BiCGStab<OperatorType, VectorType, ValueType>::PrintStart_(void) const
{
    if(this->precond_ == NULL)
    {
        LOG_INFO("BiCGStab (non-precond) linear solver starts");
    }
    else
    {
        LOG_INFO("PBiCGStab solver starts, with preconditioner:");
        this->precond_->Print();
    }
}

template <class OperatorType, class VectorType, typename ValueType>
void BiCGStab<OperatorType, VectorType, ValueType>::PrintEnd_(void) const
{
    if(this->precond_ == NULL)
    {
        LOG_INFO("BiCGStab (non-precond) ends");
    }
    else
    {
        LOG_INFO("PBiCGStab ends");
    }
}

template <class OperatorType, class VectorType, typename ValueType>
void BiCGStab<OperatorType, VectorType, ValueType>::Build(void)
{
    LOG_DEBUG(this, "BiCGStab::Build()", this->build_ << " #*# begin");

    if(this->build_ == true)
    {
        this->Clear();
    }

    assert(this->build_ == false);

    assert(this->op_ != NULL);
    assert(this->op_->GetM() == this->op_->GetN());
    assert(this->op_->GetM() > 0);

    if(this->precond_ != NULL)
    {
        this->precond_->SetOperator(*this->op_);
        this->precond_->Build();

        this->v_.CloneBackend(*this->op_);
        this->z_.CloneBackend(*this->op_);

        this->v_.Allocate("v", this->op_->GetM());
        this->z_.Allocate("z", this->op_->GetM());
    }

    this->r_.CloneBackend(*this->op_);
    this->r0_.CloneBackend(*this->op_);
    this->p_.CloneBackend(*this->op_);
    this->q_.CloneBackend(*this->op_);
    this->t_.CloneBackend(*this->op_);

    this->r_.Allocate("r", this->op_->GetM());
    this->r0_.Allocate("r0", this->op_->GetM());
    this->p_.Allocate("p", this->op_->GetM());
    this->q_.Allocate("q", this->op_->GetM());
    this->t_.Allocate("t", this->op_->GetM());

    this->build_ = true;

    LOG_DEBUG(this, "BiCGStab::Build()", this->build_ << " #*# end");
}

template <class OperatorType, class VectorType, typename ValueType>
void BiCGStab<OperatorType, VectorType, ValueType>::Clear(void)
{
    LOG_DEBUG(this, "BiCGStab::Clear()", this->build_);

    if(this->build_ == true)
    {
        this->r_.Clear();
        this->r0_.Clear();
        this->p_.Clear();
        this->q_.Clear();
        this->t_.Clear();

        if(this->precond_ != NULL)
        {
            this->precond_->Clear();
            this->precond_ = NULL;

            this->v_.Clear();
            this->z_.Clear();
        }

        this->iter_ctrl_.Clear();

        this->build_ = false;
    }
}

template <class OperatorType, class VectorType, typename ValueType>
void BiCGStab<OperatorType, VectorType, ValueType>::ReBuildNumeric(void)
{
    LOG_DEBUG(this, "BiCGStab::ReBuildNumeric()", this->build_);

    if(this->build_ == true)
    {
        this->r_.Zeros();
        this->r0_.Zeros();
        this->p_.Zeros();
        this->q_.Zeros();
        this->t_.Zeros();

        if(this->precond_ != NULL)
        {
            this->precond_->ReBuildNumeric();
            this->precond_ = NULL;

            this->v_.Zeros();
            this->z_.Zeros();
        }

        this->iter_ctrl_.Clear();
    }
    else
    {
        this->Build();
    }
}

template <class OperatorType, class VectorType, typename ValueType>
void BiCGStab<OperatorType, VectorType, ValueType>::MoveToHostLocalData_(void)
{
    LOG_DEBUG(this, "BiCGStab::MoveToHostLocalData_()", this->build_);

    if(this->build_ == true)
    {
        this->r_.MoveToHost();
        this->r0_.MoveToHost();
        this->p_.MoveToHost();
        this->q_.MoveToHost();
        this->t_.MoveToHost();

        if(this->precond_ != NULL)
        {
            this->v_.MoveToHost();
            this->z_.MoveToHost();
        }
    }
}

template <class OperatorType, class VectorType, typename ValueType>
void BiCGStab<OperatorType, VectorType, ValueType>::MoveToAcceleratorLocalData_(void)
{
    LOG_DEBUG(this, "BiCGStab::MoveToAcceleratorLocalData_()", this->build_);

    if(this->build_ == true)
    {
        this->r_.MoveToAccelerator();
        this->r0_.MoveToAccelerator();
        this->p_.MoveToAccelerator();
        this->q_.MoveToAccelerator();
        this->t_.MoveToAccelerator();

        if(this->precond_ != NULL)
        {
            this->v_.MoveToAccelerator();
            this->z_.MoveToAccelerator();
        }
    }
}

template <class OperatorType, class VectorType, typename ValueType>
void BiCGStab<OperatorType, VectorType, ValueType>::SolveNonPrecond_(const VectorType& rhs,
                                                                     VectorType* x)
{
    LOG_DEBUG(this, "BiCGStab::SolveNonPrecond_()", " #*# begin");

    assert(x != NULL);
    assert(x != &rhs);
    assert(this->op_ != NULL);
    assert(this->precond_ == NULL);
    assert(this->build_ == true);

    const OperatorType* op = this->op_;

    VectorType* r  = &this->r_;
    VectorType* r0 = &this->r0_;
    VectorType* p  = &this->p_;
    VectorType* q  = &this->q_;
    VectorType* t  = &this->t_;

    ValueType alpha;
    ValueType beta;
    ValueType omega;
    ValueType rho;
    ValueType rho_old;

    // Inital residual r0 = b - Ax
    op->Apply(*x, r0);
    r0->ScaleAdd(static_cast<ValueType>(-1), rhs);

    // Initial residual norm for |b-Ax0|
    ValueType res_norm = this->Norm(*r0);
    //    ValueType res_norm = this->Norm(rhs);

    if(this->iter_ctrl_.InitResidual(rocalution_abs(res_norm)) == false)
    {
        LOG_DEBUG(this, "BiCGStab::SolveNonPrecond_()", " #*# end");
        return;
    }

    // r = r0
    r->CopyFrom(*r0);

    // rho = <r,r>
    rho = r->Dot(*r);

    // p = r
    p->CopyFrom(*r);

    while(true)
    {
        // q = Ap
        op->Apply(*p, q);

        // alpha = rho / <r0,q>
        alpha = rho / r0->Dot(*q);

        // r = r - alpha * q
        r->AddScale(*q, -alpha);

        // t = Ar
        op->Apply(*r, t);

        // omega = <t,r> / <t,t>
        omega = t->Dot(*r) / t->Dot(*t);

        if((rocalution_abs(omega) == std::numeric_limits<ValueType>::infinity()) ||
           (omega != omega) || (omega == static_cast<ValueType>(0)))
        {
            LOG_INFO("BiCGStab omega == 0 || Nan || Inf !!! Updated solution only in p-direction");

            // Update only for p
            // x = x + alpha*p
            x->AddScale(*p, alpha);

            op->Apply(*x, p);
            p->ScaleAdd(static_cast<ValueType>(-1), rhs);

            res_norm = this->Norm(*p);

            this->iter_ctrl_.CheckResidual(rocalution_abs(res_norm), this->index_);

            break;
        }

        // x = x + alpha * p + omega * r
        x->ScaleAdd2(static_cast<ValueType>(1), *p, alpha, *r, omega);

        // r = r - omega * t
        r->AddScale(*t, -omega);

        // Check convergence
        res_norm = this->Norm(*r);
        if(this->iter_ctrl_.CheckResidual(rocalution_abs(res_norm), this->index_))
        {
            break;
        }

        // rho = <r0,r>
        rho_old = rho;
        rho     = r0->Dot(*r);

        // Check rho for zero
        if(rho == static_cast<ValueType>(0))
        {
            LOG_INFO("BiCGStab rho == 0 !!!");
            break;
        }

        // beta = (rho / rho_old) * (alpha / omega)
        beta = (rho / rho_old) * (alpha / omega);

        // p = beta * p - beta * omega * q + r
        p->ScaleAdd2(beta, *q, -beta * omega, *r, static_cast<ValueType>(1));
    }

    LOG_DEBUG(this, "BiCGStab::SolveNonPrecond_()", " #*# end");
}

template <class OperatorType, class VectorType, typename ValueType>
void BiCGStab<OperatorType, VectorType, ValueType>::SolvePrecond_(const VectorType& rhs,
                                                                  VectorType* x)
{
    LOG_DEBUG(this, "BiCGStab::SolvePrecond_()", " #*# begin");

    assert(x != NULL);
    assert(x != &rhs);
    assert(this->op_ != NULL);
    assert(this->precond_ != NULL);
    assert(this->build_ == true);

    const OperatorType* op = this->op_;

    VectorType* r  = &this->r_;
    VectorType* r0 = &this->r0_;
    VectorType* p  = &this->p_;
    VectorType* q  = &this->q_;
    VectorType* t  = &this->t_;
    VectorType* v  = &this->v_;
    VectorType* z  = &this->z_;

    ValueType alpha;
    ValueType beta;
    ValueType omega;
    ValueType rho;
    ValueType rho_old;

    // Initial residual = b - Ax
    op->Apply(*x, r0);
    r0->ScaleAdd(static_cast<ValueType>(-1), rhs);

    // Initial residual norm for |b-Ax0|
    ValueType res_norm = this->Norm(*r0);
    //    ValueType res_norm = this->Norm(rhs);

    if(this->iter_ctrl_.InitResidual(rocalution_abs(res_norm)) == false)
    {
        LOG_DEBUG(this, "BiCGStab::SolvePrecond_()", " #*# end");
        return;
    }

    // p = r = r0
    r->CopyFrom(*r0);
    p->CopyFrom(*r);

    // rho = <r,r>
    rho = r->Dot(*r);

    // Mz = r
    this->precond_->SolveZeroSol(*r, z);

    while(true)
    {
        // q = Az
        op->Apply(*z, q);

        // alpha = rho / <r0,q>
        alpha = rho / r0->Dot(*q);

        // r = r - alpha * q
        r->AddScale(*q, -alpha);

        // Mv = r
        this->precond_->SolveZeroSol(*r, v);

        // t = Av
        op->Apply(*v, t);

        // omega = (t,r) / (t,t)
        omega = t->Dot(*r) / t->Dot(*t);

        if((rocalution_abs(omega) == std::numeric_limits<ValueType>::infinity()) ||
           (omega != omega) || (omega == static_cast<ValueType>(0)))
        {
            LOG_INFO("BiCGStab omega == 0 || Nan || Inf !!! Updated solution only in p-direction");

            // Update only for p
            // x = x + alpha * p
            x->AddScale(*p, alpha);

            op->Apply(*x, p);
            p->ScaleAdd(static_cast<ValueType>(-1), rhs);

            res_norm = this->Norm(*p);
            this->iter_ctrl_.CheckResidual(rocalution_abs(res_norm), this->index_);

            break;
        }

        // x = x + alpha * z + omega * v
        x->ScaleAdd2(static_cast<ValueType>(1), *z, alpha, *v, omega);

        // r = r - omega * t
        r->AddScale(*t, -omega);

        // Check convergence
        res_norm = this->Norm(*r);
        if(this->iter_ctrl_.CheckResidual(rocalution_abs(res_norm), this->index_))
        {
            break;
        }

        // rho = <r0,r>
        rho_old = rho;
        rho     = r0->Dot(*r);

        // Check rho for zero
        if(rho == static_cast<ValueType>(0))
        {
            LOG_INFO("BiCGStab rho == 0 !!!");
            break;
        }

        // beta = (rho / rho_old) * (alpha / omega)
        beta = (rho / rho_old) * (alpha / omega);

        // p = beta * p - beta * omega * q + r
        p->ScaleAdd2(beta, *q, -beta * omega, *r, static_cast<ValueType>(1));

        // Mz = p
        this->precond_->SolveZeroSol(*p, z);
    }

    LOG_DEBUG(this, "BiCGStab::SolvePrecond_()", " #*# end");
}

template class BiCGStab<LocalMatrix<double>, LocalVector<double>, double>;
template class BiCGStab<LocalMatrix<float>, LocalVector<float>, float>;
#ifdef SUPPORT_COMPLEX
template class BiCGStab<LocalMatrix<std::complex<double>>,
                        LocalVector<std::complex<double>>,
                        std::complex<double>>;
template class BiCGStab<LocalMatrix<std::complex<float>>,
                        LocalVector<std::complex<float>>,
                        std::complex<float>>;
#endif

template class BiCGStab<GlobalMatrix<double>, GlobalVector<double>, double>;
template class BiCGStab<GlobalMatrix<float>, GlobalVector<float>, float>;
#ifdef SUPPORT_COMPLEX
template class BiCGStab<GlobalMatrix<std::complex<double>>,
                        GlobalVector<std::complex<double>>,
                        std::complex<double>>;
template class BiCGStab<GlobalMatrix<std::complex<float>>,
                        GlobalVector<std::complex<float>>,
                        std::complex<float>>;
#endif

template class BiCGStab<LocalStencil<double>, LocalVector<double>, double>;
template class BiCGStab<LocalStencil<float>, LocalVector<float>, float>;
#ifdef SUPPORT_COMPLEX
template class BiCGStab<LocalStencil<std::complex<double>>,
                        LocalVector<std::complex<double>>,
                        std::complex<double>>;
template class BiCGStab<LocalStencil<std::complex<float>>,
                        LocalVector<std::complex<float>>,
                        std::complex<float>>;
#endif

} // namespace rocalution
