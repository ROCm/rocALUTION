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
BiCGStab<OperatorType, VectorType, ValueType>::BiCGStab() {

  LOG_DEBUG(this, "BiCGStab::BiCGStab()",
            "default constructor");

}

template <class OperatorType, class VectorType, typename ValueType>
BiCGStab<OperatorType, VectorType, ValueType>::~BiCGStab() {

  LOG_DEBUG(this, "BiCGStab::~BiCGStab()",
            "destructor");

  this->Clear();

}

template <class OperatorType, class VectorType, typename ValueType>
void BiCGStab<OperatorType, VectorType, ValueType>::Print(void) const {

  if (this->precond_ == NULL) {

    LOG_INFO("BiCGStab solver");

  } else {

    LOG_INFO("PBiCGStab solver, with preconditioner:");
    this->precond_->Print();

  }

}

template <class OperatorType, class VectorType, typename ValueType>
void BiCGStab<OperatorType, VectorType, ValueType>::PrintStart_(void) const {

  if (this->precond_ == NULL) {

    LOG_INFO("BiCGStab (non-precond) linear solver starts");

  } else {

    LOG_INFO("PBiCGStab solver starts, with preconditioner:");
    this->precond_->Print();

  }

}

template <class OperatorType, class VectorType, typename ValueType>
void BiCGStab<OperatorType, VectorType, ValueType>::PrintEnd_(void) const {

  if (this->precond_ == NULL) {

    LOG_INFO("BiCGStab (non-precond) ends");

  } else {

    LOG_INFO("PBiCGStab ends");

  }

}

template <class OperatorType, class VectorType, typename ValueType>
void BiCGStab<OperatorType, VectorType, ValueType>::Build(void) {

  LOG_DEBUG(this, "BiCGStab::Build()",
            this->build_ <<
            " #*# begin");

  if (this->build_ == true)
    this->Clear();

  assert(this->build_ == false);
  this->build_ = true;

  assert(this->op_ != NULL);  
  assert(this->op_->get_nrow() == this->op_->get_ncol());
  assert(this->op_->get_nrow() > 0);


  if (this->precond_ != NULL) {

    this->precond_->SetOperator(*this->op_);

    this->precond_->Build();

    this->z_.CloneBackend(*this->op_);
    this->z_.Allocate("z", this->op_->get_nrow());

    this->q_.CloneBackend(*this->op_);
    this->q_.Allocate("q", this->op_->get_nrow());

  }

  this->r_.CloneBackend(*this->op_);
  this->r_.Allocate("r", this->op_->get_nrow());

  this->p_.CloneBackend(*this->op_);
  this->p_.Allocate("p", this->op_->get_nrow());

  this->v_.CloneBackend(*this->op_);
  this->v_.Allocate("v", this->op_->get_nrow());

  this->r0_.CloneBackend(*this->op_);
  this->r0_.Allocate("r0", this->op_->get_nrow());

  this->t_.CloneBackend(*this->op_);
  this->t_.Allocate("t", this->op_->get_nrow());

  LOG_DEBUG(this, "BiCGStab::Build()",
            this->build_ <<
            " #*# end");

}

template <class OperatorType, class VectorType, typename ValueType>
void BiCGStab<OperatorType, VectorType, ValueType>::Clear(void) {

  LOG_DEBUG(this, "BiCGStab::Clear()",
            this->build_);

  if (this->build_ == true) {

    this->r_ .Clear();
    this->p_ .Clear();
    this->v_ .Clear();
    this->r0_.Clear();
    this->t_ .Clear();

    if (this->precond_ != NULL) {

      this->precond_->Clear();
      this->precond_   = NULL;

      this->q_.Clear();
      this->z_.Clear();

    }

    this->iter_ctrl_.Clear();

    this->build_ = false;

  }

}

template <class OperatorType, class VectorType, typename ValueType>
void BiCGStab<OperatorType, VectorType, ValueType>::ReBuildNumeric(void) {

  LOG_DEBUG(this, "BiCGStab::ReBuildNumeric()",
            this->build_);

  if (this->build_ == true) {

    this->r_ .Zeros();
    this->p_ .Zeros();
    this->v_ .Zeros();
    this->r0_.Zeros();
    this->t_ .Zeros();

    if (this->precond_ != NULL) {

      this->precond_->ReBuildNumeric();
      this->precond_   = NULL;

      this->q_.Zeros();
      this->z_.Zeros();

    }

    this->iter_ctrl_.Clear();

  } else {

    this->Build();

  }

}


template <class OperatorType, class VectorType, typename ValueType>
void BiCGStab<OperatorType, VectorType, ValueType>::MoveToHostLocalData_(void) {

  LOG_DEBUG(this, "BiCGStab::MoveToHostLocalData_()",
            this->build_);

  if (this->build_ == true) {

    this->r_ .MoveToHost();
    this->p_ .MoveToHost();
    this->v_ .MoveToHost();
    this->r0_.MoveToHost();
    this->t_ .MoveToHost();

    if (this->precond_ != NULL) {
      this->z_.MoveToHost();
      this->q_.MoveToHost();
    }

  }

}

template <class OperatorType, class VectorType, typename ValueType>
void BiCGStab<OperatorType, VectorType, ValueType>::MoveToAcceleratorLocalData_(void) {

  LOG_DEBUG(this, "BiCGStab::MoveToAcceleratorLocalData_()",
            this->build_);

  if (this->build_ == true) {

    this->r_ .MoveToAccelerator();
    this->p_ .MoveToAccelerator();
    this->v_ .MoveToAccelerator();
    this->r0_.MoveToAccelerator();
    this->t_ .MoveToAccelerator();

    if (this->precond_ != NULL) {
      this->z_.MoveToAccelerator();
      this->q_.MoveToAccelerator();
    }

  }

}

template <class OperatorType, class VectorType, typename ValueType>
void BiCGStab<OperatorType, VectorType, ValueType>::SolveNonPrecond_(const VectorType &rhs,
                                                                     VectorType *x) {

  LOG_DEBUG(this, "BiCGStab::SolveNonPrecond_()",
            " #*# begin");

  assert(x != NULL);
  assert(x != &rhs);
  assert(this->op_  != NULL);
  assert(this->precond_  == NULL);
  assert(this->build_ == true);

  const OperatorType *op = this->op_;

  VectorType *r     = &this->r_;
  VectorType *p     = &this->p_;
  VectorType *v     = &this->v_;
  VectorType *r0    = &this->r0_;
  VectorType *t     = &this->t_;

  ValueType omega; 
  ValueType rho, rho_old;
  ValueType alpha, beta;

  // inital residual r0 = b - Ax
  op->Apply(*x, r0);
  r0->ScaleAdd(ValueType(-1.0), rhs);

  // r = r0
  r->CopyFrom(*r0);

  // use for |b-Ax0|
  ValueType res_norm = this->Norm(*r0);

  // use for |b|
  //  ValueType init_res = rhs.Norm();

  if (this->iter_ctrl_.InitResidual(rocalution_abs(res_norm)) == false) {

    LOG_DEBUG(this, "BiCGStab::SolveNonPrecond_()",
              " #*# end");
    return;
  }

  rho   = ValueType(1.0);
  alpha = ValueType(1.0);
  omega = ValueType(1.0);

  rho_old   = rho;

  // rho = (r0,r0)
  //  rho = init_res * init_res;
  rho = r0->Dot(*r);

  // p = r 
  p->CopyFrom(*r);

  // v = Ap
  op->Apply(*p, v);

  // alpha = rho / (r0,v)
  alpha = rho / r0->Dot(*v);

  // r = r - alpha*v
  r->AddScale(*v, ValueType(-1.0)*alpha);

  // t = Ar
  op->Apply(*r, t);

  // omega = (t,r) / (t,t)
  omega = t->Dot(*r) / t->Dot(*t);

  if (( rocalution_abs(omega) == std::numeric_limits<double>::infinity()) ||
      ( omega != omega ) ||
      ( omega == ValueType(0.0)) ) {

    LOG_INFO("BiCGStab omega == 0 || Nan || Inf !!! Updated solution only in p-direction");

    // update only for p
    // x = x + alpha*p
    x->AddScale(*p, alpha);

    op->Apply(*x, p);
    p->ScaleAdd(ValueType(-1.0), rhs);

    res_norm = this->Norm(*p);

    this->iter_ctrl_.CheckResidual(rocalution_abs(res_norm), this->index_);

    LOG_DEBUG(this, "BiCGStab::SolveNonPrecond_()",
              " #*# end");

    return;

  }

  // x = x + alpha*p + omega*r
  x->ScaleAdd2(ValueType(1.0), *p, alpha, *r, omega);

  // r = r - omega*t
  r->AddScale(*t, ValueType(-1.0)*omega);

  res_norm = this->Norm(*r);

  while (!this->iter_ctrl_.CheckResidual(rocalution_abs(res_norm), this->index_)) {

    rho_old   = rho;

    // rho = (r0,r)
    rho = r0->Dot(*r);

    if (rho == ValueType(0.0)) {
      LOG_INFO("BiCGStab rho == 0 !!!");
      break;
    }

    beta = (rho/rho_old) * (alpha/omega);

    // p = beta*p - beta*omega*v + r
    p->ScaleAdd2(beta,
                 *v, ValueType(-1.0)*omega*beta,
                 *r, ValueType(1.0));

    // v = Ap
    op->Apply(*p, v);

    // alpha = rho / (r0,v)
    alpha = rho / r0->Dot(*v);

    // r = r - alpha*v
    r->AddScale(*v, ValueType(-1.0)*alpha);

    // t = Ar
    op->Apply(*r, t);

    // omega = (t,r) / (t,t)
    omega = t->Dot(*r) / t->Dot(*t);

    if (( rocalution_abs(omega) == std::numeric_limits<double>::infinity()) ||
        ( omega != omega ) ||
        ( omega == ValueType(0.0)) ) {

      LOG_INFO("BiCGStab omega == 0 || Nan || Inf !!! Updated solution only in p-direction");

      // update only for p
      // x = x + alpha*p
      x->AddScale(*p, alpha);

      op->Apply(*x, p);
      p->ScaleAdd(ValueType(-1.0), rhs);

      res_norm = this->Norm(*p);

      this->iter_ctrl_.CheckResidual(rocalution_abs(res_norm), this->index_);

      break;

    }

    // x = x + alpha*p + omega*r
    x->ScaleAdd2( ValueType(1.0),
                  *p, alpha,
                  *r, omega);

    // r = r - omega*t
    r->AddScale(*t, ValueType(-1.0)*omega);

    res_norm = this->Norm(*r);

  }

  LOG_DEBUG(this, "BiCGStab::SolveNonPrecond_()",
            " #*# end");

}

template <class OperatorType, class VectorType, typename ValueType>
void BiCGStab<OperatorType, VectorType, ValueType>::SolvePrecond_(const VectorType &rhs,
                                                                  VectorType *x) {

  LOG_DEBUG(this, "BiCGStab::SolvePrecond_()",
            " #*# begin");

  assert(x != NULL);
  assert(x != &rhs);
  assert(this->op_  != NULL);
  assert(this->precond_  != NULL);
  assert(this->build_ == true);

  const OperatorType *op = this->op_;

  VectorType *r     = &this->r_;
  VectorType *z     = &this->z_;
  VectorType *q     = &this->q_;
  VectorType *p     = &this->p_;
  VectorType *v     = &this->v_;
  VectorType *r0    = &this->r0_;
  VectorType *t     = &this->t_;

  ValueType omega; 
  ValueType rho, rho_old;
  ValueType alpha, beta;

  // initial residual = b - Ax
  op->Apply(*x, r0);
  r0->ScaleAdd(ValueType(-1.0), rhs);

  // r = r0
  r->CopyFrom(*r0);

  // use for |b-Ax0|
  ValueType res_norm = this->Norm(*r0);

  // use for |b|
  //  ValueType init_res = rhs.Norm();

  if (this->iter_ctrl_.InitResidual(rocalution_abs(res_norm)) == false) {

    LOG_DEBUG(this, "BiCGStab::SolvePrecond_()",
              " #*# end");

    return;

  }

  rho   = ValueType(1.0);
  alpha = ValueType(1.0);
  omega = ValueType(1.0);

  rho_old   = rho;

  // rho = (r0,r0)
  //  rho = init_res * init_res;
  rho = r0->Dot(*r);

  // p = r 
  p->CopyFrom(*r);

  // solve Mz=p
  this->precond_->SolveZeroSol(*p, z);

  // v = Az
  op->Apply(*z, v);

  // alpha = rho / (r0,v)
  alpha = rho / r0->Dot(*v);

  // r = r - alpha*v
  r->AddScale(*v, ValueType(-1.0)*alpha);

  // solve Mr=q
  this->precond_->SolveZeroSol(*r, q);

  // t=Aq
  op->Apply(*q, t);

  // omega = (t,r) / (t,t)
  omega = t->Dot(*r) / t->Dot(*t);

  if (( rocalution_abs(omega) == std::numeric_limits<double>::infinity()) ||
      ( omega != omega ) ||
      ( omega == ValueType(0.0)) ) {

    LOG_INFO("BiCGStab omega == 0 || Nan || Inf !!! Updated solution only in p-direction");

    // update only for p
    // x = x + alpha*p
    x->AddScale(*p, alpha);

    op->Apply(*x, p);
    p->ScaleAdd(ValueType(-1.0), rhs);

    res_norm = this->Norm(*p);

    this->iter_ctrl_.CheckResidual(rocalution_abs(res_norm), this->index_);

    LOG_DEBUG(this, "BiCGStab::SolvePrecond_()",
              " #*# end");

    return;

  }

  // x = x + alpha*z + omega*q
  x->ScaleAdd2( ValueType(1.0),
                *z, alpha,
                *q, omega);

  // r = r - omega*t
  r->AddScale(*t, ValueType(-1.0)*omega);

  res_norm = this->Norm(*r);

  while (!this->iter_ctrl_.CheckResidual(rocalution_abs(res_norm), this->index_)) {

    rho_old   = rho;

    // rho = (r0,r)
    rho = r0->Dot(*r);

    if (rho == ValueType(0.0)) {
      LOG_INFO("BiCGStab rho == 0 !!!");
      break;
    }

    beta = (rho/rho_old) * (alpha/omega);

    // p = beta*p - omega*beta*v + r
    p->ScaleAdd2(beta,
                 *v, ValueType(-1.0)*omega*beta,
                 *r, ValueType(1.0));

    // solve Mz=p
    this->precond_->SolveZeroSol(*p, z);

    // v=Az
    op->Apply(*z, v);

    // alpha = rho/(r0,v)
    alpha = rho / r0->Dot(*v);

    // r = r - alpha*v
    r->AddScale(*v, ValueType(-1.0)*alpha);

    // solve Mr=q
    this->precond_->SolveZeroSol(*r, q);

    // t=Aq
    op->Apply(*q, t);

    // omega = (t,r) / (t,t)
    omega = t->Dot(*r) / t->Dot(*t);

   if (( rocalution_abs(omega) == std::numeric_limits<double>::infinity()) ||
        ( omega != omega ) ||
        ( omega == ValueType(0.0)) ) {

      LOG_INFO("BiCGStab omega == 0 || Nan || Inf !!! Updated solution only in p-direction");

      // update only for p
      // x = x + alpha*p
      x->AddScale(*p, alpha);

      op->Apply(*x, p);
      p->ScaleAdd(ValueType(-1.0), rhs);

      res_norm = this->Norm(*p);

      this->iter_ctrl_.CheckResidual(rocalution_abs(res_norm), this->index_);

      break;

    }

    // x = x + alpha*z + omega*q
    x->ScaleAdd2( ValueType(1.0),
                  *z, alpha,
                  *q, omega);

    // r = r - omega*t
    r->AddScale(*t, ValueType(-1.0)*omega);

    res_norm = this->Norm(*r);

  }

  LOG_DEBUG(this, "BiCGStab::SolvePrecond_()",
            " #*# end");

}


template class BiCGStab< LocalMatrix<double>, LocalVector<double>, double >;
template class BiCGStab< LocalMatrix<float>,  LocalVector<float>, float >;
#ifdef SUPPORT_COMPLEX
template class BiCGStab< LocalMatrix<std::complex<double> >,  LocalVector<std::complex<double> >, std::complex<double> >;
template class BiCGStab< LocalMatrix<std::complex<float> >,  LocalVector<std::complex<float> >, std::complex<float> >;
#endif


template class BiCGStab< GlobalMatrix<double>, GlobalVector<double>, double >;
template class BiCGStab< GlobalMatrix<float>,  GlobalVector<float>, float >;
#ifdef SUPPORT_COMPLEX
template class BiCGStab< GlobalMatrix<std::complex<double> >,  GlobalVector<std::complex<double> >, std::complex<double> >;
template class BiCGStab< GlobalMatrix<std::complex<float> >,  GlobalVector<std::complex<float> >, std::complex<float> >;
#endif

template class BiCGStab< LocalStencil<double>, LocalVector<double>, double >;
template class BiCGStab< LocalStencil<float>,  LocalVector<float>, float >;
#ifdef SUPPORT_COMPLEX
template class BiCGStab< LocalStencil<std::complex<double> >,  LocalVector<std::complex<double> >, std::complex<double> >;
template class BiCGStab< LocalStencil<std::complex<float> >,  LocalVector<std::complex<float> >, std::complex<float> >;
#endif

}
