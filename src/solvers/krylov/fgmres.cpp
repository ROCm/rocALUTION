#include "../../utils/def.hpp"
#include "fgmres.hpp"
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
FGMRES<OperatorType, VectorType, ValueType>::FGMRES() {

  LOG_DEBUG(this, "FGMRES::FGMRES()",
            "default constructor");

  this->size_basis_ = 30;

}

template <class OperatorType, class VectorType, typename ValueType>
FGMRES<OperatorType, VectorType, ValueType>::~FGMRES() {

  LOG_DEBUG(this, "FGMRES::~FGMRES()",
            "destructor");

  this->Clear();

}

template <class OperatorType, class VectorType, typename ValueType>
void FGMRES<OperatorType, VectorType, ValueType>::Print(void) const {

  if (this->precond_ == NULL) { 

    LOG_INFO("FGMRES solver");

  } else {

    LOG_INFO("FGMRES solver, with preconditioner:");
    this->precond_->Print();

  }

}

template <class OperatorType, class VectorType, typename ValueType>
void FGMRES<OperatorType, VectorType, ValueType>::PrintStart_(void) const {

  if (this->precond_ == NULL) { 

    LOG_INFO("FGMRES(" <<this->size_basis_ <<") (non-precond) linear solver starts");

  } else {

    LOG_INFO("FGMRES(" <<this->size_basis_ <<") solver starts, with preconditioner:");
    this->precond_->Print();

  }

}

template <class OperatorType, class VectorType, typename ValueType>
void FGMRES<OperatorType, VectorType, ValueType>::PrintEnd_(void) const {

  if (this->precond_ == NULL) { 

    LOG_INFO("FGMRES(" <<this->size_basis_ <<") (non-precond) ends");

  } else {

    LOG_INFO("FGMRES(" <<this->size_basis_ <<") ends");

  }

}

template <class OperatorType, class VectorType, typename ValueType>
void FGMRES<OperatorType, VectorType, ValueType>::Build(void) {

  LOG_DEBUG(this, "FGMRES::Build()",
            this->build_ <<
            " #*# begin");

  if (this->build_ == true)
    this->Clear();

  assert(this->build_ == false);

  if (this->res_norm_ != 2) {
    LOG_INFO("FGMRES solver supports only L2 residual norm. The solver is switching to L2 norm");
    this->res_norm_ = 2;
  }

  this->build_ = true;

  if (this->precond_ != NULL) {

    this->precond_->SetOperator(*this->op_);

    this->precond_->Build();

    this->z_ = new VectorType*[this->size_basis_+1];
    for (int i=0; i<this->size_basis_+1; ++i) {
      this->z_[i] = new VectorType;
      this->z_[i]->CloneBackend(*this->op_);
      this->z_[i]->Allocate("z", this->op_->get_nrow());
    }

  }

  this->w_.CloneBackend(*this->op_);
  this->w_.Allocate("w", this->op_->get_nrow());

  this->v_ = new VectorType*[this->size_basis_+1];
  for (int i = 0; i < this->size_basis_+1; ++i) {
    this->v_[i] = new VectorType;
    this->v_[i]->CloneBackend(*this->op_);
    this->v_[i]->Allocate("v", this->op_->get_nrow());
  }

  LOG_DEBUG(this, "FGMRES::Build()",
            this->build_ <<
            " #*# end");

}

template <class OperatorType, class VectorType, typename ValueType>
void FGMRES<OperatorType, VectorType, ValueType>::Clear(void) {

  LOG_DEBUG(this, "FGMRES::Clear()",
            this->build_);

  if (this->build_ == true) {

    if (this->precond_ != NULL) {
        this->precond_->Clear();
        this->precond_   = NULL;

        for (int i=0; i<this->size_basis_+1; ++i)
          delete this->z_[i];
        delete[] this->z_;
    }

    this->w_.Clear();

    for (int i = 0; i < this->size_basis_+1; ++i)
      delete this->v_[i];
    delete[] this->v_;

    this->iter_ctrl_.Clear();

    this->build_ = false;

  }

}

template <class OperatorType, class VectorType, typename ValueType>
void FGMRES<OperatorType, VectorType, ValueType>::ReBuildNumeric(void) {

  LOG_DEBUG(this, "FGMRES::ReBuildNumeric()",
            this->build_);

  if (this->build_ == true) {

    this->w_.Zeros();

    for (int i = 0; i < this->size_basis_+1; ++i)
      this->v_[i]->Zeros();

    if (this->precond_ != NULL) {

        this->precond_->ReBuildNumeric();

        for (int i=0; i<this->size_basis_+1; ++i)
          this->z_[i]->Zeros();
    }

    this->iter_ctrl_.Clear();

  } else {

    this->Build();

  }

}


template <class OperatorType, class VectorType, typename ValueType>
void FGMRES<OperatorType, VectorType, ValueType>::MoveToHostLocalData_(void) {

  LOG_DEBUG(this, "FGMRES::MoveToHostLocalData_()",
            this->build_);  

  if (this->build_ == true) {

    this->w_.MoveToHost();

    for (int i = 0; i < this->size_basis_+1; ++i)
      this->v_[i]->MoveToHost();

    if (this->precond_ != NULL) {
      for (int i=0; i<this->size_basis_+1; ++i)
        this->z_[i]->MoveToHost();
      this->precond_->MoveToHost();
    }

  }

}

template <class OperatorType, class VectorType, typename ValueType>
void FGMRES<OperatorType, VectorType, ValueType>::MoveToAcceleratorLocalData_(void) {

  LOG_DEBUG(this, "FGMRES::MoveToAcceleratorLocalData_()",
            this->build_);

  if (this->build_ == true) {

    this->w_.MoveToAccelerator();

    for (int i = 0; i < this->size_basis_+1; ++i)
      this->v_[i]->MoveToAccelerator();


    if (this->precond_ != NULL) {
      for (int i=0; i<this->size_basis_+1; ++i)
        this->z_[i]->MoveToAccelerator();
      this->precond_->MoveToAccelerator();
    }

  }

}

template <class OperatorType, class VectorType, typename ValueType>
void FGMRES<OperatorType, VectorType, ValueType>::SetBasisSize(const int size_basis) {

  LOG_DEBUG(this, "FGMRES:SetBasisSize()",
            size_basis);

  assert(size_basis > 0);
  this->size_basis_ = size_basis;

}

template <class OperatorType, class VectorType, typename ValueType>
void FGMRES<OperatorType, VectorType, ValueType>::SolveNonPrecond_(const VectorType &rhs,
                                                                        VectorType *x) {

  LOG_DEBUG(this, "FGMRES::SolveNonPrecond_()",
            " #*# begin");

  assert(x != NULL);
  assert(x != &rhs);
  assert(this->op_  != NULL);
  assert(this->precond_  == NULL);
  assert(this->build_ == true);
  assert(this->size_basis_ > 0);

  if (this->res_norm_ != 2) {
    LOG_INFO("FGMRES solver supports only L2 residual norm. The solver is switching to L2 norm");
    this->res_norm_ = 2;
  }

  const OperatorType *op = this->op_;

  VectorType *w  = &this->w_;
  VectorType **v = this->v_;

  int size_basis = this->size_basis_;

  std::vector<ValueType> c(size_basis);
  std::vector<ValueType> s(size_basis);
  std::vector<ValueType> H((size_basis+1)*size_basis, ValueType(0.0));
  std::vector<ValueType> sq(size_basis+1, ValueType(0.0));

  // Compute residual V[0] = b - Ax
  op->Apply(*x, v[0]);
  v[0]->ScaleAdd(ValueType(-1.0), rhs);

  // res_norm = (v[0],v[0])
  ValueType res_norm = this->Norm(*v[0]);
  double res = rocalution_abs(res_norm);

  if (this->iter_ctrl_.InitResidual(res) == false) {

      LOG_DEBUG(this, "FGMRES::SolveNonPrecond_()",
            " #*# end");

      return;
  }


  // Residual normalization
  // v = r / ||r||
  v[0]->Scale(ValueType(1.0)/res_norm);
  sq[0] = res_norm;

  for (int i = 0; i < size_basis; ++i) {
    // w = A*v(i)
    op->Apply(*v[i], w);

    // Build Hessenberg matrix H
    for (int j = 0; j <= i; ++j) {
      // H(j,i) = <w,v(j)>
      H[j+i*(size_basis+1)] = w->Dot(*v[j]);

      // w = w - H(j,i) * v(j)
      w->AddScale( *v[j], ValueType(-1.0) * H[j+i*(size_basis+1)] );
    }

    // H(i+1,i) = ||w||
    H[i+1+i*(size_basis+1)] = this->Norm(*w);

    // v(i+1) = w / H(i+1,i)
    w->Scale(ValueType(1.0) / H[i+1+i*(size_basis+1)]);
    v[i+1]->CopyFrom(*w);

    // Apply J(0),...,J(j-1) on ( H(0,i),...,H(i,i) )
    for (int k = 0; k < i; ++k)
      this->ApplyGivensRotation_( c[k], s[k], H[k+i*(size_basis+1)], H[k+1+i*(size_basis+1)] );

    // Construct J(i) (Givens rotation taken from wikipedia pseudo code)
    this->GenerateGivensRotation_(H[i+i*(size_basis+1)], H[i+1+i*(size_basis+1)], c[i], s[i]);

    // Apply J(i) to H(i,i) and H(i,i+1) such that H(i,i+1) = 0
    this->ApplyGivensRotation_(c[i], s[i], H[i+i*(size_basis+1)], H[i+1+i*(size_basis+1)]);

    // Apply J(i) to the norm of the residual sg[i]
    this->ApplyGivensRotation_(c[i], s[i], sq[i], sq[i+1]);

    // Get current residual
    res_norm = rocalution_abs(sq[i+1]);
    res = rocalution_abs(res_norm);

    if (this->iter_ctrl_.CheckResidual(res)) {
      this->BackSubstitute_(sq, H, i);
      // Compute solution of least square problem
      for (int j = 0; j <= i; ++j)
        x->AddScale(*v[j], sq[j]);
      break;
    }

  }

  // If convergence has not been reached, RESTART
  if (!this->iter_ctrl_.CheckResidualNoCount(res)) {
    this->BackSubstitute_(sq, H, size_basis-1);

    // Compute solution of least square problem
    for (int j = 0; j <= size_basis-1; ++j)
      x->AddScale(*v[j], sq[j]);

    // Compute residual with previous solution vector
    op->Apply(*x, v[0]);
    v[0]->ScaleAdd(ValueType(-1.0), rhs);

    res_norm = this->Norm(*v[0]);
    res = rocalution_abs(res_norm);

  }

  while (!this->iter_ctrl_.CheckResidual(res)) {

    sq.assign(sq.size(), ValueType(0.0));

    // Residual normalization
    // v = r / ||r||
    v[0]->Scale(ValueType(1.0)/res_norm);
    sq[0] = res_norm;
    /*
    for (int i = 0; i < (size_basis+1)*size_basis; i++)
      H[i] = ValueType(0.0);

    for (int i = 1; i < size_basis+1; i++)
      sq[i] = ValueType(0.0);
    */
    for (int i = 0; i < size_basis; ++i) {
      // w = A*v(i)
      op->Apply(*v[i], w);

      // Build Hessenberg matrix H
      for (int j = 0; j <= i; ++j) {
        // H(j,i) = <w,v(j)>
        H[j+i*(size_basis+1)] = w->Dot(*v[j]);

        // w = w - H(j,i) * v(j)
        w->AddScale( *v[j], ValueType(-1.0) * H[j+i*(size_basis+1)] );
      }

      // H(i+1,i) = ||w||
      H[i+1+i*(size_basis+1)] = this->Norm(*w);

      // v(i+1) = w / H(i+1,i)
      w->Scale(ValueType(1.0) / H[i+1+i*(size_basis+1)]);
      v[i+1]->CopyFrom(*w);

      // Apply J(0),...,J(j-1) on ( H(0,i),...,H(i,i) )
      for (int k = 0; k < i; ++k)
        this->ApplyGivensRotation_( c[k], s[k], H[k+i*(size_basis+1)], H[k+1+i*(size_basis+1)] );

      // Construct J(i) (Givens rotation taken from wikipedia pseudo code)
      this->GenerateGivensRotation_(H[i+i*(size_basis+1)], H[i+1+i*(size_basis+1)], c[i], s[i]);

      // Apply J(i) to H(i,i) and H(i,i+1) such that H(i,i+1) = 0
      this->ApplyGivensRotation_(c[i], s[i], H[i+i*(size_basis+1)], H[i+1+i*(size_basis+1)]);

      // Apply J(i) to the norm of the residual sg[i]
      this->ApplyGivensRotation_(c[i], s[i], sq[i], sq[i+1]);

      // Get current residual
      res_norm = rocalution_abs(sq[i+1]);
      res = rocalution_abs(res_norm);

      if (this->iter_ctrl_.CheckResidual(res)) {
        this->BackSubstitute_(sq, H, i);
        // Compute solution of least square problem
        for (int j = 0; j <= i; ++j)
          x->AddScale(*v[j], sq[j]);
        break;
      }

    }

    // If convergence has not been reached, RESTART
    if (!this->iter_ctrl_.CheckResidualNoCount(res)) {
      this->BackSubstitute_(sq, H, size_basis-1);

      // Compute solution of least square problem
      for (int j = 0; j <= size_basis-1; ++j)
        x->AddScale(*v[j], sq[j]);

      // Compute residual with previous solution vector
      op->Apply(*x, v[0]);
      v[0]->ScaleAdd(ValueType(-1.0), rhs);

      res_norm = this->Norm(*v[0]);
      res = rocalution_abs(res_norm);

    }
  }

  LOG_DEBUG(this, "FGMRES::SolveNonPrecond_()",
            " #*# end");

}

template <class OperatorType, class VectorType, typename ValueType>
void FGMRES<OperatorType, VectorType, ValueType>::SolvePrecond_(const VectorType &rhs,
                                                                     VectorType *x) {

  LOG_DEBUG(this, "FGMRES::SolvePrecond_()",
            " #*# begin");

  assert(x != NULL);
  assert(x != &rhs);
  assert(this->op_  != NULL);
  assert(this->precond_ != NULL);
  assert(this->build_ == true);
  assert(this->size_basis_ > 0);

  if (this->res_norm_ != 2) {
    LOG_INFO("FGMRES solver supports only L2 residual norm. The solver is switching to L2 norm");
    this->res_norm_ = 2;
  }

  const OperatorType *op = this->op_;

  VectorType **z = this->z_;
  VectorType *w  = &this->w_;
  VectorType **v = this->v_;

  int size_basis = this->size_basis_;

  std::vector<ValueType> c(size_basis);
  std::vector<ValueType> s(size_basis);
  std::vector<ValueType> H((size_basis+1)*size_basis, ValueType(0.0));
  std::vector<ValueType> sq(size_basis+1, ValueType(0.0));

  // Compute residual V[0] = b - Ax
  op->Apply(*x, v[0]);
  v[0]->ScaleAdd(ValueType(-1.0), rhs);

  // res_norm = (v[0],v[0])
  ValueType res_norm = this->Norm(*v[0]);
  double res = rocalution_abs(res_norm);

  if (this->iter_ctrl_.InitResidual(res) == false) {

      LOG_DEBUG(this, "FGMRES::SolvePrecond_()",
            " #*# end");

      return;
  }


  // Residual normalization
  // v = r / ||r||
  v[0]->Scale(ValueType(1.0)/res_norm);
  sq[0] = res_norm;

  for (int i = 0; i < size_basis; ++i) {
    // Mz = v(j)
    this->precond_->SolveZeroSol(*v[i], z[i]);

    // w = A*z
    op->Apply(*z[i], w);

    // Build Hessenberg matrix H
    for (int j = 0; j <= i; ++j) {
      // H(j,i) = <w,v(j)>
      H[j+i*(size_basis+1)] = w->Dot(*v[j]);

      // w = w - H(j,i) * v(j)
      w->AddScale(*v[j], ValueType(-1.0) * H[j+i*(size_basis+1)]);
    }

    // H(i+1,i) = ||w||
    H[i+1+i*(size_basis+1)] = this->Norm(*w);

    // v(i+1) = w / H(i+1,i)
    w->Scale(ValueType(1.0) / H[i+1+i*(size_basis+1)]);
    v[i+1]->CopyFrom(*w);

    // Apply J(0),...,J(j-1) on ( H(0,i),...,H(i,i) )
    for (int k = 0; k < i; ++k)
      this->ApplyGivensRotation_(c[k], s[k], H[k+i*(size_basis+1)], H[k+1+i*(size_basis+1)]);

    // Construct J(i) (Givens rotation taken from wikipedia pseudo code)
    this->GenerateGivensRotation_(H[i+i*(size_basis+1)], H[i+1+i*(size_basis+1)], c[i], s[i]);

    // Apply J(i) to H(i,i) and H(i,i+1) such that H(i,i+1) = 0
    this->ApplyGivensRotation_(c[i], s[i], H[i+i*(size_basis+1)], H[i+1+i*(size_basis+1)]);

    // Apply J(i) to the norm of the residual sg[i]
    this->ApplyGivensRotation_(c[i], s[i], sq[i], sq[i+1]);

    // Get current residual
    res_norm = rocalution_abs(sq[i+1]);
    res = rocalution_abs(res_norm);

    if (this->iter_ctrl_.CheckResidual(res)) {
      this->BackSubstitute_(sq, H, i);

      // Compute solution of least square problem
      for (int j = 0; j <= i; ++j)
        x->AddScale(*z[j], sq[j]);

      break;
    }

  }

  // If convergence has not been reached, RESTART
  if (!this->iter_ctrl_.CheckResidualNoCount(res)) {
    this->BackSubstitute_(sq, H, size_basis-1);

    // Compute solution of least square problem
    for (int j = 0; j <= size_basis-1; ++j)
      x->AddScale(*z[j], sq[j]);

    // Compute residual with previous solution vector
    op->Apply(*x, v[0]);
    v[0]->ScaleAdd(ValueType(-1.0), rhs);

    res_norm = this->Norm(*v[0]);
    res = rocalution_abs(res_norm);

  }

  while (!this->iter_ctrl_.CheckResidual(res)) {

    sq.assign(sq.size(), ValueType(0.0));

    // Residual normalization
    // v = r / ||r||
    v[0]->Scale(ValueType(1.0)/res_norm);
    sq[0] = res_norm;

    for (int i = 0; i < size_basis; ++i) {
      // Mz = v(j)
      this->precond_->SolveZeroSol(*v[i], z[i]);

      // w = A*z
      op->Apply(*z[i], w);

      // Build Hessenberg matrix H
      for (int j = 0; j <= i; ++j) {
        // H(j,i) = <w,v(j)>
        H[j+i*(size_basis+1)] = w->Dot(*v[j]);

        // w = w - H(j,i) * v(j)
        w->AddScale(*v[j], ValueType(-1.0) * H[j+i*(size_basis+1)]);
      }

      // H(i+1,i) = ||w||
      H[i+1+i*(size_basis+1)] = this->Norm(*w);

      // v(i+1) = w / H(i+1,i)
      w->Scale(ValueType(1.0) / H[i+1+i*(size_basis+1)]);
      v[i+1]->CopyFrom(*w);

      // Apply J(0),...,J(j-1) on ( H(0,i),...,H(i,i) )
      for (int k = 0; k < i; ++k)
        this->ApplyGivensRotation_(c[k], s[k], H[k+i*(size_basis+1)], H[k+1+i*(size_basis+1)]);

      // Construct J(i) (Givens rotation taken from wikipedia pseudo code)
      this->GenerateGivensRotation_(H[i+i*(size_basis+1)], H[i+1+i*(size_basis+1)], c[i], s[i]);

      // Apply J(i) to H(i,i) and H(i,i+1) such that H(i,i+1) = 0
      this->ApplyGivensRotation_(c[i], s[i], H[i+i*(size_basis+1)], H[i+1+i*(size_basis+1)]);

      // Apply J(i) to the norm of the residual sg[i]
      this->ApplyGivensRotation_(c[i], s[i], sq[i], sq[i+1]);

      // Get current residual
      res_norm = rocalution_abs(sq[i+1]);
      res = rocalution_abs(res_norm);

      if (this->iter_ctrl_.CheckResidual(res)) {
        this->BackSubstitute_(sq, H, i);

        // Compute solution of least square problem
        for (int j = 0; j <= i; ++j)
          x->AddScale(*z[j], sq[j]);

        break;
      }

    }

    // If convergence has not been reached, RESTART
    if (!this->iter_ctrl_.CheckResidualNoCount(res)) {
      this->BackSubstitute_(sq, H, size_basis-1);

      // Compute solution of least square problem
      for (int j = 0; j <= size_basis-1; ++j)
        x->AddScale(*z[j], sq[j]);

      // Compute residual with previous solution vector
      op->Apply(*x, v[0]);
      v[0]->ScaleAdd(ValueType(-1.0), rhs);

      res_norm = this->Norm(*v[0]);
      res = rocalution_abs(res_norm);

    }

  }

  LOG_DEBUG(this, "FGMRES::SolvePrecond_()",
            " #*# end");

}

template <class OperatorType, class VectorType, typename ValueType>
void FGMRES<OperatorType, VectorType, ValueType>::GenerateGivensRotation_(const ValueType &x, const ValueType &y,
                                                                                ValueType &c,       ValueType &s) const {

  if (y == ValueType(0.0)) {

  	c = ValueType(1.0);
  	s = ValueType(0.0);

  } else if (rocalution_abs(y) > rocalution_abs(x)) {

  	ValueType tmp = x / y;
  	s = ValueType(1.0) / sqrt(ValueType(1.0) + tmp * tmp);
  	c = tmp * s;

  } else {

  	ValueType tmp = y / x;
  	c = ValueType(1.0) / sqrt(ValueType(1.0) + tmp * tmp);
  	s = tmp * c;

  }

}

template <class OperatorType, class VectorType, typename ValueType>
void FGMRES<OperatorType, VectorType, ValueType>::ApplyGivensRotation_(const ValueType &c, const ValueType &s,
                                                                            ValueType &x,       ValueType &y) const {

  ValueType temp = x;
  x =  c * x + s * y;
  y = -s * temp + c * y;

}

template <class OperatorType, class VectorType, typename ValueType>
void FGMRES<OperatorType, VectorType, ValueType>::BackSubstitute_(std::vector<ValueType> &sq,
                                                                  const std::vector<ValueType> &H,
                                                                  int k) const {

  for (int i = k; i >= 0; --i) {
    sq[i] = sq[i] / H[i+i*(this->size_basis_+1)];
    for (int j = i-1; j >= 0; --j)
      sq[j] = sq[j] - H[j+i*(this->size_basis_+1)] * sq[i];

  }

}


template class FGMRES< LocalMatrix<double>, LocalVector<double>, double >;
template class FGMRES< LocalMatrix<float>,  LocalVector<float>, float >;
#ifdef SUPPORT_COMPLEX
template class FGMRES< LocalMatrix<std::complex<double> >,  LocalVector<std::complex<double> >, std::complex<double> >;
template class FGMRES< LocalMatrix<std::complex<float> >,  LocalVector<std::complex<float> >, std::complex<float> >;
#endif

template class FGMRES< GlobalMatrix<double>, GlobalVector<double>, double >;
template class FGMRES< GlobalMatrix<float>,  GlobalVector<float>, float >;
#ifdef SUPPORT_COMPLEX
template class FGMRES< GlobalMatrix<std::complex<double> >,  GlobalVector<std::complex<double> >, std::complex<double> >;
template class FGMRES< GlobalMatrix<std::complex<float> >,  GlobalVector<std::complex<float> >, std::complex<float> >;
#endif

template class FGMRES< LocalStencil<double>, LocalVector<double>, double >;
template class FGMRES< LocalStencil<float>,  LocalVector<float>, float >;
#ifdef SUPPORT_COMPLEX
template class FGMRES< LocalStencil<std::complex<double> >,  LocalVector<std::complex<double> >, std::complex<double> >;
template class FGMRES< LocalStencil<std::complex<float> >,  LocalVector<std::complex<float> >, std::complex<float> >;
#endif

}
