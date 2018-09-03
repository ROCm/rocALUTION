#include "../../utils/def.hpp"
#include "bicgstabl.hpp"
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
BiCGStabl<OperatorType, VectorType, ValueType>::BiCGStabl() {

  log_debug(this, "BiCGStabl::BiCGStabl()",
            "default constructor");

  this->l_ = 2;

}

template <class OperatorType, class VectorType, typename ValueType>
BiCGStabl<OperatorType, VectorType, ValueType>::~BiCGStabl() {

  log_debug(this, "BiCGStabl::~BiCGStabl()",
            "destructor");

  this->Clear();

}

template <class OperatorType, class VectorType, typename ValueType>
void BiCGStabl<OperatorType, VectorType, ValueType>::Print(void) const {

  if (this->precond_ == NULL) {

    LOG_INFO("BiCGStab(" << this->l_ << ") solver");

  } else {

    LOG_INFO("PBiCGStab(" << this->l_ << ") solver, with preconditioner:");
    this->precond_->Print();

  }

}

template <class OperatorType, class VectorType, typename ValueType>
void BiCGStabl<OperatorType, VectorType, ValueType>::PrintStart_(void) const {

  if (this->precond_ == NULL) {

    LOG_INFO("BiCGStab(" << this->l_ << ") (non-precond) linear solver starts");

  } else {

    LOG_INFO("PBiCGStab(" << this->l_ << ") solver starts, with preconditioner:");
    this->precond_->Print();

  }

}

template <class OperatorType, class VectorType, typename ValueType>
void BiCGStabl<OperatorType, VectorType, ValueType>::PrintEnd_(void) const {

  if (this->precond_ == NULL) {

    LOG_INFO("BiCGStab(" << this->l_ << ") (non-precond) ends");

  } else {

    LOG_INFO("PBiCGStab(" << this->l_ << ") ends");

  }

}

template <class OperatorType, class VectorType, typename ValueType>
void BiCGStabl<OperatorType, VectorType, ValueType>::Build(void) {

  log_debug(this, "BiCGStabl::Build()",
            this->build_,
            " #*# begin");

  if (this->build_ == true)
    this->Clear();

  assert(this->build_ == false);
  this->build_ = true;

  assert(this->op_ != NULL);  
  assert(this->op_->GetM() == this->op_->GetN());
  assert(this->op_->GetM() > 0);


  if (this->precond_ != NULL) {

    this->precond_->SetOperator(*this->op_);

    this->precond_->Build();

    this->z_.CloneBackend(*this->op_);
    this->z_.Allocate("z", this->op_->GetM());

  }

  this->r0_.CloneBackend(*this->op_);
  this->r0_.Allocate("r0", this->op_->GetM());

  this->r_ = new VectorType*[this->l_+1];
  this->u_ = new VectorType*[this->l_+1];

  for (int i=0; i<this->l_+1; ++i) {

    this->r_[i] = new VectorType;
    this->r_[i]->CloneBackend(*this->op_);
    this->r_[i]->Allocate("r", this->op_->GetM());

    this->u_[i] = new VectorType;
    this->u_[i]->CloneBackend(*this->op_);
    this->u_[i]->Allocate("u", this->op_->GetM());

  }

  this->gamma0_ = new ValueType[this->l_];
  this->gamma1_ = new ValueType[this->l_];
  this->gamma2_ = new ValueType[this->l_];
  this->sigma_  = new ValueType[this->l_];

  this->tau_ = new ValueType*[this->l_];
  for (int i=0; i<this->l_; ++i)
    this->tau_[i] = new ValueType[this->l_];

  log_debug(this, "BiCGStabl::Build()",
            this->build_,
            " #*# end");

}

template <class OperatorType, class VectorType, typename ValueType>
void BiCGStabl<OperatorType, VectorType, ValueType>::Clear(void) {

  log_debug(this, "BiCGStabl::Clear()",
            this->build_);

  if (this->build_ == true) {

    this->r0_.Clear();

    for (int i=0; i<this->l_+1; ++i) {

      this->r_[i]->Clear();
      this->u_[i]->Clear();

      delete this->r_[i];
      delete this->u_[i];

    }

    delete [] this->r_;
    delete [] this->u_;

    delete [] this->gamma0_;
    delete [] this->gamma1_;
    delete [] this->gamma2_;
    delete [] this->sigma_;

    for (int i=0; i<this->l_; ++i)
      delete [] this->tau_[i];
    delete [] this->tau_;

    if (this->precond_ != NULL) {

      this->precond_->Clear();
      this->precond_   = NULL;

      this->z_.Clear();

    }

    this->iter_ctrl_.Clear();

    this->build_ = false;

  }

}

template <class OperatorType, class VectorType, typename ValueType>
void BiCGStabl<OperatorType, VectorType, ValueType>::ReBuildNumeric(void) {

  log_debug(this, "BiCGStabl::ReBuildNumeric()",
            this->build_);

  if (this->build_ == true) {

    this->r0_.Zeros();

    for (int i=0; i<this->l_+1; ++i) {

      this->r_[i]->Zeros();
      this->u_[i]->Zeros();

    }

    if (this->precond_ != NULL) {

      this->precond_->ReBuildNumeric();
      this->z_.Zeros();

    }

    this->iter_ctrl_.Clear();

   } else {

    this->Build();

  }


}


template <class OperatorType, class VectorType, typename ValueType>
void BiCGStabl<OperatorType, VectorType, ValueType>::SetOrder(const int l) {

  assert(this->build_ == false);
  assert(l > 0);

  this->l_ = l;

}

template <class OperatorType, class VectorType, typename ValueType>
void BiCGStabl<OperatorType, VectorType, ValueType>::MoveToHostLocalData_(void) {

  log_debug(this, "BiCGStabl::MoveToHostLocalData_()",
            this->build_);

  if (this->build_ == true) {

    this->r0_.MoveToHost();

    for (int i=0; i<this->l_+1; ++i) {

      this->r_[i]->MoveToHost();
      this->u_[i]->MoveToHost();

    }

    if (this->precond_ != NULL) {
      this->z_.MoveToHost();
    }

  }

}

template <class OperatorType, class VectorType, typename ValueType>
void BiCGStabl<OperatorType, VectorType, ValueType>::MoveToAcceleratorLocalData_(void) {

  log_debug(this, "BiCGStabl::MoveToAcceleratorLocalData_()",
            this->build_);

  if (this->build_ == true) {

    this->r0_.MoveToAccelerator();

    for (int i=0; i<this->l_+1; ++i) {

      this->r_[i]->MoveToAccelerator();
      this->u_[i]->MoveToAccelerator();

    }

    if (this->precond_ != NULL) {
      this->z_.MoveToAccelerator();
    }

  }

}

template <class OperatorType, class VectorType, typename ValueType>
void BiCGStabl<OperatorType, VectorType, ValueType>::SolveNonPrecond_(const VectorType &rhs,
                                                                     VectorType *x) {

  log_debug(this, "BiCGStabl::SolveNonPrecond_()",
            " #*# begin",
            (const void*&)rhs,
            x);

  assert(x != NULL);
  assert(x != &rhs);
  assert(this->op_  != NULL);
  assert(this->precond_  == NULL);
  assert(this->build_ == true);

  const OperatorType *op = this->op_;

  VectorType *r0 = &this->r0_;
  VectorType **r = this->r_;
  VectorType **u = this->u_;

  int l = this->l_;

  ValueType alpha   = ValueType(0.0);
  ValueType beta    = ValueType(0.0);
  ValueType omega   = ValueType(1.0);
  ValueType rho_old = ValueType(-1.0);
  ValueType rho;

  ValueType *gamma0 = this->gamma0_;
  ValueType *gamma1 = this->gamma1_;
  ValueType *gamma2 = this->gamma2_;
  ValueType *sigma  = this->sigma_;
  ValueType **tau   = this->tau_;

  // inital residual r0 = b - Ax
  op->Apply(*x, r0);
  r0->ScaleAdd(ValueType(-1.0), rhs);

  ValueType res = this->Norm(*r0);
  this->iter_ctrl_.InitResidual(rocalution_abs(res));

  // r_0 = r0
  r[0]->CopyFrom(*r0);

  // u_0 = 0
  u[0]->Zeros();

  // BiCG part
  for (int j=0; j<l; ++j) {

    // rho = (r_j, r0)
    rho = r[j]->Dot(*r0);

    // beta = alpha * rho / rho_old
    beta = alpha * rho / rho_old;

    // rho_old = rho
    rho_old = rho;

    // u_i = r_i - beta * u_i
    for (int i=0; i<=j; ++i)
      u[i]->ScaleAdd(ValueType(-1.0)*beta, *r[i]);

    // u_j+1 = A u_j
    op->Apply(*u[j], u[j+1]);

    // alpha = rho / (u_j+1, r0)
    alpha = rho / u[j+1]->Dot(*r0);

    // r_i = r_i - alpha * u_i+1
    for (int i=0; i<=j; ++i)
      r[i]->AddScale(*u[i+1], ValueType(-1.0)*alpha);

    // r_j+1 = A r_j
    op->Apply(*r[j], r[j+1]);

    // x = x + alpha * u_0
    x->AddScale(*u[0], alpha);

  }

  // modified Gram Schmidt
  for (int j=0; j<l; ++j) {
    for (int i=0; i<j; ++i) {

      // tau_ij = (r_j+1, r_i+1) / sigma_i
      tau[i][j] = r[j+1]->Dot(*r[i+1]) / sigma[i];

      // r_j+1 = r_j+1 - tau_ij * r_i+1
      r[j+1]->AddScale(*r[i+1], ValueType(-1.0)*tau[i][j]);

    }

    // sigma_j = (r_j+1, r_j+1)
    sigma[j] = r[j+1]->Dot(*r[j+1]);

    // gamma' = (r_0, r_j+1) / sigma_j
    gamma1[j] = r[0]->Dot(*r[j+1]) / sigma[j];

  }

  // omega = gamma'_l-1; gamma_l-1 = gamma'_l-1
  gamma0[l-1] = gamma1[l-1];
  omega = gamma1[l-1];

  // gamma_j = gamma'_j - sum(tau_ji * gamma_i) (i=j+1,...,l-1)
  for (int j=l-2; j>=0; --j) {
    gamma0[j] = gamma1[j];
    for (int i=j+1; i<l; ++i)
      gamma0[j] -= tau[j][i] * gamma0[i];
  }

  // gamma''_j = gamma_j+1 + sum(tau_ji * gamma_i+1) (i=j+1,...,l-2)
  for(int j=0; j<l-1; ++j) {
    gamma2[j] = gamma0[j+1];
    for(int i=j+1; i<l-1; ++i)
      gamma2[j] += tau[j][i] * gamma0[i+1];
  }

  // Update

  // x = x + gamma_0 * r_0
  x->AddScale(*r[0], gamma0[0]);

  // r_0 = r_0 - gamma'_l-1 * r_l
  r[0]->AddScale(*r[l], ValueType(-1.0)*gamma1[l-1]);

  // u_0 = u_0 - gamma_l-1 * u_l
  u[0]->AddScale(*u[l], ValueType(-1.0)*gamma0[l-1]);

  for (int j=1; j<l; ++j) {

    // u_0 = u_0 - gamma_j-1 * u_j
    u[0]->AddScale(*u[j], ValueType(-1.0)*gamma0[j-1]);

    // x = x + gamma''_j-1 * r_j
    x->AddScale(*r[j], gamma2[j-1]);

    // r_0 = r_0 - gamma'_j-1 * r_j
    r[0]->AddScale(*r[j], ValueType(-1.0)*gamma1[j-1]);

  }

  res = this->Norm(*r[0]);
  while (!this->iter_ctrl_.CheckResidual(rocalution_abs(res), this->index_)) {

    rho_old *= ValueType(-1.0) * omega;

    // BiCG part
    for (int j=0; j<l; ++j) {

      // rho = (r_j, r0)
      rho = r[j]->Dot(*r0);

      // beta = alpha * rho / rho_old
      beta = alpha * rho / rho_old;

      // rho_old = rho
      rho_old = rho;

      // u_i = r_i - beta * u_i
      for (int i=0; i<=j; ++i)
        u[i]->ScaleAdd(ValueType(-1.0)*beta, *r[i]);

      // u_j+1 = A u_j
      op->Apply(*u[j], u[j+1]);

      // alpha = rho / (u_j+1, r0)
      alpha = rho / u[j+1]->Dot(*r0);

      // r_i = r_i - alpha * u_i+1
      for (int i=0; i<=j; ++i)
        r[i]->AddScale(*u[i+1], ValueType(-1.0)*alpha);

      // r_j+1 = A r_j
      op->Apply(*r[j], r[j+1]);

      // x = x + alpha * u_0
      x->AddScale(*u[0], alpha);

    }

    // modified Gram Schmidt
    for (int j=0; j<l; ++j) {
      for (int i=0; i<j; ++i) {

        // tau_ij = (r_j+1, r_i+1) / sigma_i
        tau[i][j] = r[j+1]->Dot(*r[i+1]) / sigma[i];

        // r_j+1 = r_j+1 - tau_ij * r_i+1
        r[j+1]->AddScale(*r[i+1], ValueType(-1.0)*tau[i][j]);

      }

      // sigma_j = (r_j+1, r_j+1)
      sigma[j] = r[j+1]->Dot(*r[j+1]);

      // gamma' = (r_0, r_j+1) / sigma_j
      gamma1[j] = r[0]->Dot(*r[j+1]) / sigma[j];

    }

    // omega = gamma'_l-1; gamma_l-1 = gamma'_l-1
    gamma0[l-1] = gamma1[l-1];
    omega = gamma1[l-1];

    // gamma_j = gamma'_j - sum(tau_ji * gamma_i) (i=j+1,...,l-1)
    for (int j=l-2; j>=0; --j) {
      gamma0[j] = gamma1[j];
      for (int i=j+1; i<l; ++i)
        gamma0[j] -= tau[j][i] * gamma0[i];
    }

    // gamma''_j = gamma_j+1 + sum(tau_ji * gamma_i+1) (i=j+1,...,l-2)
    for(int j=0; j<l-1; ++j) {
      gamma2[j] = gamma0[j+1];
      for(int i=j+1; i<l-1; ++i)
        gamma2[j] += tau[j][i] * gamma0[i+1];
    }

    // Update

    // x = x + gamma_0 * r_0
    x->AddScale(*r[0], gamma0[0]);

    // r_0 = r_0 - gamma'_l-1 * r_l
    r[0]->AddScale(*r[l], ValueType(-1.0)*gamma1[l-1]);

    // u_0 = u_0 - gamma_l-1 * u_l
    u[0]->AddScale(*u[l], ValueType(-1.0)*gamma0[l-1]);

    for (int j=1; j<l; ++j) {

      // u_0 = u_0 - gamma_j-1 * u_j
      u[0]->AddScale(*u[j], ValueType(-1.0)*gamma0[j-1]);

      // x = x + gamma''_j-1 * r_j
      x->AddScale(*r[j], gamma2[j-1]);

      // r_0 = r_0 - gamma'_j-1 * r_j
      r[0]->AddScale(*r[j], ValueType(-1.0)*gamma1[j-1]);

    }

    res = this->Norm(*r[0]);

  }

  log_debug(this, "BiCGStabl::SolveNonPrecond_()",
            " #*# end");

}

template <class OperatorType, class VectorType, typename ValueType>
void BiCGStabl<OperatorType, VectorType, ValueType>::SolvePrecond_(const VectorType &rhs,
                                                                  VectorType *x) {

  log_debug(this, "BiCGStabl::SolvePrecond_()",
            " #*# begin",
            (const void*&)rhs,
            x);

  assert(x != NULL);
  assert(x != &rhs);
  assert(this->op_  != NULL);
  assert(this->precond_  != NULL);
  assert(this->build_ == true);

  const OperatorType *op = this->op_;

  VectorType *r0 = &this->r0_;
  VectorType *z  = &this->z_;
  VectorType **r = this->r_;
  VectorType **u = this->u_;

  int l = this->l_;

  ValueType alpha   = ValueType(0.0);
  ValueType beta    = ValueType(0.0);
  ValueType omega   = ValueType(1.0);
  ValueType rho_old = ValueType(-1.0);
  ValueType rho;

  ValueType *gamma0 = this->gamma0_;
  ValueType *gamma1 = this->gamma1_;
  ValueType *gamma2 = this->gamma2_;
  ValueType *sigma  = this->sigma_;
  ValueType **tau   = this->tau_;

  // inital residual z = b - Ax
  op->Apply(*x, z);
  z->ScaleAdd(ValueType(-1.0), rhs);

  ValueType res = this->Norm(*z);

  this->iter_ctrl_.InitResidual(rocalution_abs(res));

  // M r0 = z
  this->precond_->SolveZeroSol(*z, r0);

  // r_0 = r0
  r[0]->CopyFrom(*r0);

  // u_0 = 0
  u[0]->Zeros();

  // BiCG part
  for (int j=0; j<l; ++j) {

    // rho = (r_j, r0)
    rho = r[j]->Dot(*r0);

    // beta = alpha * rho / rho_old
    beta = alpha * rho / rho_old;

    // rho_old = rho
    rho_old = rho;

    // u_i = r_i - beta * u_i
    for (int i=0; i<=j; ++i)
      u[i]->ScaleAdd(ValueType(-1.0)*beta, *r[i]);

    // z = A u_j
    op->Apply(*u[j], z);

    // M u_j+1 = z
    this->precond_->SolveZeroSol(*z, u[j+1]);

    // alpha = rho / (u_j+1, r0)
    alpha = rho / u[j+1]->Dot(*r0);

    // r_i = r_i - alpha * u_i+1
    for (int i=0; i<=j; ++i)
      r[i]->AddScale(*u[i+1], ValueType(-1.0)*alpha);

    // z = A r_j
    op->Apply(*r[j], z);

    // M r_j+1 = z
    this->precond_->SolveZeroSol(*z, r[j+1]);

    // x = x + alpha * u_0
    x->AddScale(*u[0], alpha);

  }

  // modified Gram Schmidt
  for (int j=0; j<l; ++j) {
    for (int i=0; i<j; ++i) {

      // tau_ij = (r_j+1, r_i+1) / sigma_i
      tau[i][j] = r[j+1]->Dot(*r[i+1]) / sigma[i];

      // r_j+1 = r_j+1 - tau_ij * r_i+1
      r[j+1]->AddScale(*r[i+1], ValueType(-1.0)*tau[i][j]);

    }

    // sigma_j = (r_j+1, r_j+1)
    sigma[j] = r[j+1]->Dot(*r[j+1]);

    // gamma' = (r_0, r_j+1) / sigma_j
    gamma1[j] = r[0]->Dot(*r[j+1]) / sigma[j];

  }

  // omega = gamma'_l-1; gamma_l-1 = gamma'_l-1
  gamma0[l-1] = gamma1[l-1];
  omega = gamma1[l-1];

  // gamma_j = gamma'_j - sum(tau_ji * gamma_i) (i=j+1,...,l-1)
  for (int j=l-2; j>=0; --j) {
    gamma0[j] = gamma1[j];
    for (int i=j+1; i<l; ++i)
      gamma0[j] -= tau[j][i] * gamma0[i];
  }

  // gamma''_j = gamma_j+1 + sum(tau_ji * gamma_i+1) (i=j+1,...,l-2)
  for(int j=0; j<l-1; ++j) {
    gamma2[j] = gamma0[j+1];
    for(int i=j+1; i<l-1; ++i)
      gamma2[j] += tau[j][i] * gamma0[i+1];
  }

  // Update

  // x = x + gamma_0 * r_0
  x->AddScale(*r[0], gamma0[0]);

  // r_0 = r_0 - gamma'_l-1 * r_l
  r[0]->AddScale(*r[l], ValueType(-1.0)*gamma1[l-1]);

  // u_0 = u_0 - gamma_l-1 * u_l
  u[0]->AddScale(*u[l], ValueType(-1.0)*gamma0[l-1]);

  for (int j=1; j<l; ++j) {

    // u_0 = u_0 - gamma_j-1 * u_j
    u[0]->AddScale(*u[j], ValueType(-1.0)*gamma0[j-1]);

    // x = x + gamma''_j-1 * r_j
    x->AddScale(*r[j], gamma2[j-1]);

    // r_0 = r_0 - gamma'_j-1 * r_j
    r[0]->AddScale(*r[j], ValueType(-1.0)*gamma1[j-1]);

  }

  res = this->Norm(*r[0]);

  while (!this->iter_ctrl_.CheckResidual(rocalution_abs(res), this->index_)) {

    rho_old *= ValueType(-1.0) * omega;

    // BiCG part
    for (int j=0; j<l; ++j) {

      // rho = (r_j, r0)
      rho = r[j]->Dot(*r0);

      // beta = alpha * rho / rho_old
      beta = alpha * rho / rho_old;

      // rho_old = rho
      rho_old = rho;

      // u_i = r_i - beta * u_i
      for (int i=0; i<=j; ++i)
        u[i]->ScaleAdd(ValueType(-1.0)*beta, *r[i]);

      // z = A u_j
      op->Apply(*u[j], z);

      // M u_j+1 = z
      this->precond_->SolveZeroSol(*z, u[j+1]);

      // alpha = rho / (u_j+1, r0)
      alpha = rho / u[j+1]->Dot(*r0);

      // r_i = r_i - alpha * u_i+1
      for (int i=0; i<=j; ++i)
        r[i]->AddScale(*u[i+1], ValueType(-1.0)*alpha);

      // z = A r_j
      op->Apply(*r[j], z);

      // M r_j+1 = z
      this->precond_->SolveZeroSol(*z, r[j+1]);

      // x = x + alpha * u_0
      x->AddScale(*u[0], alpha);

    }

    // modified Gram Schmidt
    for (int j=0; j<l; ++j) {
      for (int i=0; i<j; ++i) {

        // tau_ij = (r_j+1, r_i+1) / sigma_i
        tau[i][j] = r[j+1]->Dot(*r[i+1]) / sigma[i];

        // r_j+1 = r_j+1 - tau_ij * r_i+1
        r[j+1]->AddScale(*r[i+1], ValueType(-1.0)*tau[i][j]);

      }

      // sigma_j = (r_j+1, r_j+1)
      sigma[j] = r[j+1]->Dot(*r[j+1]);

      // gamma' = (r_0, r_j+1) / sigma_j
      gamma1[j] = r[0]->Dot(*r[j+1]) / sigma[j];

    }

    // omega = gamma'_l-1; gamma_l-1 = gamma'_l-1
    gamma0[l-1] = gamma1[l-1];
    omega = gamma1[l-1];

    // gamma_j = gamma'_j - sum(tau_ji * gamma_i) (i=j+1,...,l-1)
    for (int j=l-2; j>=0; --j) {
      gamma0[j] = gamma1[j];
      for (int i=j+1; i<l; ++i)
        gamma0[j] -= tau[j][i] * gamma0[i];
    }

    // gamma''_j = gamma_j+1 + sum(tau_ji * gamma_i+1) (i=j+1,...,l-2)
    for(int j=0; j<l-1; ++j) {
      gamma2[j] = gamma0[j+1];
      for(int i=j+1; i<l-1; ++i)
        gamma2[j] += tau[j][i] * gamma0[i+1];
    }

    // Update

    // x = x + gamma_0 * r_0
    x->AddScale(*r[0], gamma0[0]);

    // r_0 = r_0 - gamma'_l-1 * r_l
    r[0]->AddScale(*r[l], ValueType(-1.0)*gamma1[l-1]);

    // u_0 = u_0 - gamma_l-1 * u_l
    u[0]->AddScale(*u[l], ValueType(-1.0)*gamma0[l-1]);

    for (int j=1; j<l; ++j) {

      // u_0 = u_0 - gamma_j-1 * u_j
      u[0]->AddScale(*u[j], ValueType(-1.0)*gamma0[j-1]);

      // x = x + gamma''_j-1 * r_j
      x->AddScale(*r[j], gamma2[j-1]);

      // r_0 = r_0 - gamma'_j-1 * r_j
      r[0]->AddScale(*r[j], ValueType(-1.0)*gamma1[j-1]);

    }

    res = this->Norm(*r[0]);

  }

  log_debug(this, "BiCGStabl::SolvePrecond_()",
            " #*# end");

}


template class BiCGStabl< LocalMatrix<double>, LocalVector<double>, double >;
template class BiCGStabl< LocalMatrix<float>,  LocalVector<float>, float >;
#ifdef SUPPORT_COMPLEX
template class BiCGStabl< LocalMatrix<std::complex<double> >,  LocalVector<std::complex<double> >, std::complex<double> >;
template class BiCGStabl< LocalMatrix<std::complex<float> >,  LocalVector<std::complex<float> >, std::complex<float> >;
#endif

template class BiCGStabl< GlobalMatrix<double>, GlobalVector<double>, double >;
template class BiCGStabl< GlobalMatrix<float>,  GlobalVector<float>, float >;
#ifdef SUPPORT_COMPLEX
template class BiCGStabl< GlobalMatrix<std::complex<double> >,  GlobalVector<std::complex<double> >, std::complex<double> >;
template class BiCGStabl< GlobalMatrix<std::complex<float> >,  GlobalVector<std::complex<float> >, std::complex<float> >;
#endif

}
