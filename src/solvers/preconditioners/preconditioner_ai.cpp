#include "../../utils/def.hpp"
#include "preconditioner_ai.hpp"
#include "../solver.hpp"

#include "../../base/local_matrix.hpp"

#include "../../base/local_vector.hpp"

#include "../../utils/log.hpp"

#include <math.h>
#include <complex>

namespace paralution {

template <class OperatorType, class VectorType, typename ValueType>
AIChebyshev<OperatorType, VectorType, ValueType>::AIChebyshev() {

  LOG_DEBUG(this, "AIChebyshev::AIChebyshev()",
            "default constructor");

  this->p_ = 0;
  this->lambda_min_ = ValueType(0.0);
  this->lambda_max_ = ValueType(0.0);

}

template <class OperatorType, class VectorType, typename ValueType>
AIChebyshev<OperatorType, VectorType, ValueType>::~AIChebyshev() {

  LOG_DEBUG(this, "AIChebyshev::~AIChebyshev()",
            "destructor");

  this->Clear();

}

template <class OperatorType, class VectorType, typename ValueType>
void AIChebyshev<OperatorType, VectorType, ValueType>::Print(void) const {

  LOG_INFO("Approximate Inverse Chebyshev(" << this->p_ <<") preconditioner");

  if (this->build_ == true) {
    LOG_INFO("AI matrix nnz = " << this->AIChebyshev_.get_nnz());
  }

}

template <class OperatorType, class VectorType, typename ValueType>
void AIChebyshev<OperatorType, VectorType, ValueType>::Set(const int p, const ValueType lambda_min, const ValueType lambda_max) {

  LOG_DEBUG(this, "AIChebyshev::Set()",
            "p=" << p <<
            " lambda_min=" << lambda_min <<
            " lambda_max=" << lambda_max);

  assert(p > 0);
  assert(lambda_min != ValueType(0.0));
  assert(lambda_max != ValueType(0.0));
  assert(this->build_ == false);

  this->p_ = p;
  this->lambda_min_ = lambda_min;
  this->lambda_max_ = lambda_max;

}

template <class OperatorType, class VectorType, typename ValueType>
void AIChebyshev<OperatorType, VectorType, ValueType>::Build(void) {

  LOG_DEBUG(this, "AIChebyshev::Build()",
            this->build_ <<
            " #*# begin");

  if (this->build_ == true)
    this->Clear();

  assert(this->build_ == false);
  this->build_ = true;

  assert(this->op_ != NULL);

  this->AIChebyshev_.CloneFrom(*this->op_);

  const ValueType q = (ValueType(1.0) - sqrt(this->lambda_min_/this->lambda_max_)) /
                      (ValueType(1.0) + sqrt(this->lambda_min_/this->lambda_max_));
  ValueType c = ValueType(1.0) / sqrt(this->lambda_min_ * this->lambda_max_);

  // Shifting
  // Z = 2/(beta-alpha) [A-(beta+alpha)/2]
  OperatorType Z;
  Z.CloneFrom(*this->op_);

  Z.AddScalarDiagonal(ValueType(-1.0)*(this->lambda_max_ + this->lambda_min_)/(ValueType(2.0)));
  Z.ScaleDiagonal(ValueType(2.0)/(this->lambda_max_ - this->lambda_min_));

  // Chebyshev formula/series
  // ai = I c_0 / 2 + sum c_k T_k
  // Tk = 2 Z T_k-1 - T_k-2

  // 1st term
  // T_0 = I
  // ai = I c_0 / 2
  this->AIChebyshev_.AddScalarDiagonal(c/ValueType(2.0));

  OperatorType Tkm2;
  Tkm2.CloneFrom(Z);
  // 2nd term
  // T_1 = Z
  // ai = ai + c_1 Z
  c = c * ValueType(-1.0)*q;
  this->AIChebyshev_.MatrixAdd(Tkm2, ValueType(1.0),
                               c, true);

  // T_2 = 2*Z*Z - I
  // + c (2*Z*Z - I)
  OperatorType Tkm1;
  Tkm1.CloneBackend(*this->op_);
  Tkm1.MatrixMult(Z, Z);
  Tkm1.Scale(ValueType(2.0));
  Tkm1.AddScalarDiagonal(ValueType(-1.0));

  c = c * ValueType(-1.0)*q;
  this->AIChebyshev_.MatrixAdd(Tkm1, ValueType(1.0),
                               c, true);


  // T_k = 2 Z T_k-1 - T_k-2     
  OperatorType Tk;
  Tk.CloneBackend(*this->op_);

  for (int i=2; i<=this->p_; ++i){

    Tk.MatrixMult(Z, Tkm1);
    Tk.MatrixAdd(Tkm2, ValueType(2.0),
                 ValueType(-1.0), true);
    
    c = c * ValueType(-1.0)*q;
    this->AIChebyshev_.MatrixAdd(Tk, ValueType(1.0),
                                 c, true);
    
    if (i+1 <= this->p_) {
      Tkm2.CloneFrom(Tkm1);
      Tkm1.CloneFrom(Tk);
    }

  }

  LOG_DEBUG(this, "AIChebyshev::Build()",
            this->build_ <<
            " #*# end");

}

template <class OperatorType, class VectorType, typename ValueType>
void AIChebyshev<OperatorType, VectorType, ValueType>::Clear(void) {

  LOG_DEBUG(this, "AIChebyshev::Clear()",
            this->build_);

  this->AIChebyshev_.Clear();
  this->build_ = false;

}

template <class OperatorType, class VectorType, typename ValueType>
void AIChebyshev<OperatorType, VectorType, ValueType>::MoveToHostLocalData_(void) {

  LOG_DEBUG(this, "AIChebyshev::MoveToHostLocalData_()",
            this->build_);  


  this->AIChebyshev_.MoveToHost();

}

template <class OperatorType, class VectorType, typename ValueType>
void AIChebyshev<OperatorType, VectorType, ValueType>::MoveToAcceleratorLocalData_(void) {

  LOG_DEBUG(this, "AIChebyshev::MoveToAcceleratorLocalData_()",
            this->build_);


  this->AIChebyshev_.MoveToAccelerator();

}

template <class OperatorType, class VectorType, typename ValueType>
void AIChebyshev<OperatorType, VectorType, ValueType>::Solve(const VectorType &rhs,
                                                    VectorType *x) {


  LOG_DEBUG(this, "AIChebyshev::Solve()",
            " #*# begin");

  assert(this->build_ == true);
  assert(x != NULL);
  assert(x != &rhs);


  this->AIChebyshev_.Apply(rhs, x);

  LOG_DEBUG(this, "AIChebyshev::Solve()",
            " #*# end");

}



template <class OperatorType, class VectorType, typename ValueType>
FSAI<OperatorType, VectorType, ValueType>::FSAI() {

  LOG_DEBUG(this, "FSAI::FSAI()",
            "default constructor");


  this->op_mat_format_ = false;
  this->precond_mat_format_ = CSR;

  this->matrix_power_ = 1;
  this->external_pattern_ = false;
  this->matrix_pattern_ = NULL;

}

template <class OperatorType, class VectorType, typename ValueType>
FSAI<OperatorType, VectorType, ValueType>::~FSAI() {
  
  LOG_DEBUG(this, "FSAI::~FSAI()",
            "destructor");

  this->Clear();
  this->matrix_pattern_ = NULL;

}

template <class OperatorType, class VectorType, typename ValueType>
void FSAI<OperatorType, VectorType, ValueType>::Print(void) const {

  LOG_INFO("Factorized Sparse Approximate Inverse preconditioner");

  if (this->build_ == true) {
    LOG_INFO("FSAI matrix nnz = " << this->FSAI_L_.get_nnz()
                                   + this->FSAI_LT_.get_nnz()
                                   - this->FSAI_L_.get_nrow());
  }

}

template <class OperatorType, class VectorType, typename ValueType>
void FSAI<OperatorType, VectorType, ValueType>::Set(const int power) {

  LOG_DEBUG(this, "FSAI::Set()",
            power);


  assert(this->build_ == false);
  assert(power > 0);

  this->matrix_power_ = power;

}

template <class OperatorType, class VectorType, typename ValueType>
void FSAI<OperatorType, VectorType, ValueType>::Set(const OperatorType &pattern) {

  LOG_DEBUG(this, "FSAI::Set()",
            "");

  assert(this->build_ == false);

  this->matrix_pattern_ = &pattern;

}

template <class OperatorType, class VectorType, typename ValueType>
void FSAI<OperatorType, VectorType, ValueType>::Build(void) {

  LOG_DEBUG(this, "FSAI::Build()",
            this->build_ <<
            " #*# begin");


  if (this->build_ == true)
    this->Clear();

  assert(this->build_ == false);
  this->build_ = true;

  assert(this->op_ != NULL);

  this->FSAI_L_.CloneFrom(*this->op_);
  this->FSAI_L_.FSAI(this->matrix_power_, this->matrix_pattern_);

  this->FSAI_LT_.CloneFrom(this->FSAI_L_);
  this->FSAI_LT_.Transpose();

  this->t_.CloneBackend(*this->op_);
  this->t_.Allocate("temporary", this->op_->get_nrow());

  if (this->op_mat_format_ == true) {
    this->FSAI_L_.ConvertTo(this->precond_mat_format_);
    this->FSAI_LT_.ConvertTo(this->precond_mat_format_);
  }

}

template <class OperatorType, class VectorType, typename ValueType>
void FSAI<OperatorType, VectorType, ValueType>::Clear(void) {

  LOG_DEBUG(this, "FSAI::Clear()",
            this->build_);


  if (this->build_ == true) {

    this->FSAI_L_.Clear();
    this->FSAI_LT_.Clear();

    this->t_.Clear();

    this->op_mat_format_ = false;
    this->precond_mat_format_ = CSR;

    this->build_ = false;

  }

  LOG_DEBUG(this, "FSAI::Build()",
            this->build_ <<
            " #*# end");

}

template <class OperatorType, class VectorType, typename ValueType>
void FSAI<OperatorType, VectorType, ValueType>::SetPrecondMatrixFormat(const unsigned int mat_format) {

  LOG_DEBUG(this, "FSAI::SetPrecondMatrixFormat()",
            mat_format);


  this->op_mat_format_ = true;
  this->precond_mat_format_ = mat_format;

}

template <class OperatorType, class VectorType, typename ValueType>
void FSAI<OperatorType, VectorType, ValueType>::MoveToHostLocalData_(void) {

  LOG_DEBUG(this, "FSAI::MoveToHostLocalData_()",
            this->build_);  


  this->FSAI_L_.MoveToHost();
  this->FSAI_LT_.MoveToHost();

  this->t_.MoveToHost();

}

template <class OperatorType, class VectorType, typename ValueType>
void FSAI<OperatorType, VectorType, ValueType>::MoveToAcceleratorLocalData_(void) {

  LOG_DEBUG(this, "FSAI::MoveToAcceleratorLocalData_()",
            this->build_);

  this->FSAI_L_.MoveToAccelerator();
  this->FSAI_LT_.MoveToAccelerator();

  this->t_.MoveToAccelerator();

}


template <class OperatorType, class VectorType, typename ValueType>
void FSAI<OperatorType, VectorType, ValueType>::Solve(const VectorType &rhs, VectorType *x) {

  LOG_DEBUG(this, "FSAI::Solve()",
            " #*# begin");

  assert(this->build_ == true);
  assert(x != NULL);
  assert(x != &rhs);

  this->FSAI_L_.Apply(rhs, &this->t_);
  this->FSAI_LT_.Apply(this->t_, x);

  LOG_DEBUG(this, "FSAI::Solve()",
            " #*# end");

}



template <class OperatorType, class VectorType, typename ValueType>
SPAI<OperatorType, VectorType, ValueType>::SPAI() {

  LOG_DEBUG(this, "SPAI::SPAI()",
            "default constructor");

  this->op_mat_format_ = false;
  this->precond_mat_format_ = CSR;

}

template <class OperatorType, class VectorType, typename ValueType>
SPAI<OperatorType, VectorType, ValueType>::~SPAI() {

  LOG_DEBUG(this, "SPAI::~SPAI()",
            "destructor");


  this->Clear();

}

template <class OperatorType, class VectorType, typename ValueType>
void SPAI<OperatorType, VectorType, ValueType>::Print(void) const {

  LOG_INFO("SParse Approximate Inverse preconditioner");

  if (this->build_ == true) {
    LOG_INFO("SPAI matrix nnz = " << this->SPAI_.get_nnz());
  }

}

template <class OperatorType, class VectorType, typename ValueType>
void SPAI<OperatorType, VectorType, ValueType>::Build(void) {

  LOG_DEBUG(this, "SPAI::Build()",
            this->build_ <<
            " #*# begin");


  if (this->build_ == true)
    this->Clear();

  assert(this->build_ == false);
  this->build_ = true;

  assert(this->op_ != NULL);

  this->SPAI_.CloneFrom(*this->op_);
  this->SPAI_.SPAI();

  if (this->op_mat_format_ == true)
    this->SPAI_.ConvertTo(this->precond_mat_format_);

  LOG_DEBUG(this, "SPAI::Build()",
            this->build_ <<
            " #*# end");

}

template <class OperatorType, class VectorType, typename ValueType>
void SPAI<OperatorType, VectorType, ValueType>::Clear(void) {

  LOG_DEBUG(this, "SPAI::Clear()",
            this->build_);

  if (this->build_ == true) {

    this->SPAI_.Clear();

    this->op_mat_format_ = false;
    this->precond_mat_format_ = CSR;

    this->build_ = false;

  }

}

template <class OperatorType, class VectorType, typename ValueType>
void SPAI<OperatorType, VectorType, ValueType>::SetPrecondMatrixFormat(const unsigned int mat_format) {

  LOG_DEBUG(this, "SPAI::SetPrecondMatrixFormat()",
            mat_format);

  this->op_mat_format_ = true;
  this->precond_mat_format_ = mat_format;

}

template <class OperatorType, class VectorType, typename ValueType>
void SPAI<OperatorType, VectorType, ValueType>::MoveToHostLocalData_(void) {

  LOG_DEBUG(this, "SPAI::MoveToHostLocalData_()",
            this->build_);  

  this->SPAI_.MoveToHost();

}

template <class OperatorType, class VectorType, typename ValueType>
void SPAI<OperatorType, VectorType, ValueType>::MoveToAcceleratorLocalData_(void) {

  LOG_DEBUG(this, "SPAI::MoveToAcceleratorLocalData_()",
            this->build_);

  this->SPAI_.MoveToAccelerator();

}


template <class OperatorType, class VectorType, typename ValueType>
void SPAI<OperatorType, VectorType, ValueType>::Solve(const VectorType &rhs, VectorType *x) {

  LOG_DEBUG(this, "SPAI::Solve()",
            " #*# begin");

  assert(this->build_ == true);
  assert(x != NULL);
  assert(x != &rhs);

  this->SPAI_.Apply(rhs, x);

  LOG_DEBUG(this, "SPAI::Solve()",
            " #*# end");

}



template <class OperatorType, class VectorType, typename ValueType>
TNS<OperatorType, VectorType, ValueType>::TNS() {

  LOG_DEBUG(this, "TNS::TNS()",
            "default constructor");

  this->op_mat_format_ = false;
  this->precond_mat_format_ = CSR;

  this->impl_ = true;

}

template <class OperatorType, class VectorType, typename ValueType>
TNS<OperatorType, VectorType, ValueType>::~TNS() {

  LOG_DEBUG(this, "TNS::~TNS()",
            "destructor");


  this->Clear();

}

template <class OperatorType, class VectorType, typename ValueType>
void TNS<OperatorType, VectorType, ValueType>::Print(void) const {

  LOG_INFO("Truncated Neumann Series (TNS) Preconditioner");

  if (this->build_ == true) {

    if (this->impl_ == true) {
      LOG_INFO("Implicit TNS L matrix nnz = " << this->L_.get_nnz());
    } else {
      LOG_INFO("Explicit TNS matrix nnz = " << this->TNS_.get_nnz());
    }
  }

}

template <class OperatorType, class VectorType, typename ValueType>
void TNS<OperatorType, VectorType, ValueType>::Set(const bool imp) {

  assert(this->build_ != true);

  this->impl_ = imp;

}

template <class OperatorType, class VectorType, typename ValueType>
void TNS<OperatorType, VectorType, ValueType>::Build(void) {

  LOG_DEBUG(this, "TNS::Build()",
            this->build_ <<
            " #*# begin");


  if (this->build_ == true)
    this->Clear();

  assert(this->build_ == false);
  this->build_ = true;

  assert(this->op_ != NULL);

  if (this->impl_ == true) {
    // implicit computation

    this->L_.CloneBackend(*this->op_);
    this->LT_.CloneBackend(*this->op_);
    
    this->tmp1_.CloneBackend(*this->op_);
    this->tmp2_.CloneBackend(*this->op_);
    this->Dinv_.CloneBackend(*this->op_);
    
    this->op_->ExtractInverseDiagonal(&this->Dinv_);
    
    this->op_->ExtractL(&this->L_, false);
    this->L_.DiagonalMatrixMultR(this->Dinv_);
    
    this->LT_.CloneFrom(this->L_);
    this->LT_.Transpose();
    
    this->tmp1_.Allocate("tmp1 vec for TNS", this->op_->get_nrow());
    this->tmp2_.Allocate("tmp2 vec for TNS", this->op_->get_nrow());
  
  } else {
    // explicit computation

    OperatorType K, KT;
    
    this->L_.CloneBackend(*this->op_);
    this->Dinv_.CloneBackend(*this->op_);
    this->TNS_.CloneBackend(*this->op_);

    K.CloneBackend(*this->op_);
    KT.CloneBackend(*this->op_);
    
    this->op_->ExtractInverseDiagonal(&this->Dinv_);

    // get the diagonal but flash them to zero
    // keep the structure
    this->op_->ExtractL(&this->L_, true);
    this->L_.ScaleDiagonal(ValueType(0.0));
    this->L_.DiagonalMatrixMultR(this->Dinv_);

    K.MatrixMult(this->L_, this->L_);

    // add -I
    this->L_.AddScalarDiagonal(ValueType(-1.0));
    
    K.MatrixAdd(this->L_, 
                ValueType(1.0), // for L^2
                ValueType(-1.0),  // for (-I+L)
                true);

    KT.CloneFrom(K);
    KT.Transpose();
    
    KT.DiagonalMatrixMultR(this->Dinv_);
        
    this->TNS_.MatrixMult(KT, K);

    K.Clear();
    KT.Clear();
    
    this->L_.Clear();
    this->Dinv_.Clear();
  }


  if (this->op_mat_format_ == true) {
    this->TNS_.ConvertTo(this->precond_mat_format_);
    this->L_.ConvertTo(this->precond_mat_format_);
    this->LT_.ConvertTo(this->precond_mat_format_);
  }

  LOG_DEBUG(this, "TNS::Build()",
            this->build_ <<
            " #*# end");

}

template <class OperatorType, class VectorType, typename ValueType>
void TNS<OperatorType, VectorType, ValueType>::Clear(void) {

  LOG_DEBUG(this, "TNS::Clear()",
            this->build_);

  if (this->build_ == true) {

    this->TNS_.Clear();

    this->L_.Clear();
    this->LT_.Clear();
    this->Dinv_.Clear();

    this->tmp1_.Clear();
    this->tmp2_.Clear();

    this->op_mat_format_ = false;
    this->precond_mat_format_ = CSR;

    this->build_ = false;

  }

}

template <class OperatorType, class VectorType, typename ValueType>
void TNS<OperatorType, VectorType, ValueType>::SetPrecondMatrixFormat(const unsigned int mat_format) {

  LOG_DEBUG(this, "TNS::SetPrecondMatrixFormat()",
            mat_format);

  this->op_mat_format_ = true;
  this->precond_mat_format_ = mat_format;

}

template <class OperatorType, class VectorType, typename ValueType>
void TNS<OperatorType, VectorType, ValueType>::MoveToHostLocalData_(void) {

  LOG_DEBUG(this, "TNS::MoveToHostLocalData_()",
            this->build_);  

  this->TNS_.MoveToHost();
  this->L_.MoveToHost();
  this->LT_.MoveToHost();
  this->Dinv_.MoveToHost();
  this->tmp1_.MoveToHost();
  this->tmp2_.MoveToHost();

}

template <class OperatorType, class VectorType, typename ValueType>
void TNS<OperatorType, VectorType, ValueType>::MoveToAcceleratorLocalData_(void) {

  LOG_DEBUG(this, "TNS::MoveToAcceleratorLocalData_()",
            this->build_);

  this->TNS_.MoveToHost();
  this->L_.MoveToAccelerator();
  this->LT_.MoveToAccelerator();
  this->Dinv_.MoveToAccelerator();
  this->tmp1_.MoveToAccelerator();
  this->tmp2_.MoveToAccelerator();

}


template <class OperatorType, class VectorType, typename ValueType>
void TNS<OperatorType, VectorType, ValueType>::Solve(const VectorType &rhs, VectorType *x) {

  LOG_DEBUG(this, "TNS::Solve()",
            " #*# begin");

  assert(this->build_ == true);
  assert(x != NULL);
  assert(x != &rhs);

  if (this->impl_ == true) {
    // implicit

    this->L_.Apply(rhs, &this->tmp1_);
    this->L_.Apply(this->tmp1_, &this->tmp2_);
    this->tmp1_.AddScale(this->tmp2_, ValueType(-1.0));

    x->CopyFrom(rhs);
    x->AddScale(this->tmp1_, ValueType(-1.0));


    x->PointWiseMult(this->Dinv_);
    
    this->LT_.Apply(*x, &this->tmp1_);
    this->LT_.Apply(this->tmp1_, &this->tmp2_);
    
    x->ScaleAdd2(ValueType(1.0),
                 this->tmp1_, ValueType(-1.0),
                 this->tmp2_, ValueType(1.0));

  } else {
    // explicit

    this->TNS_.Apply(rhs, x);
  
  }

  //  LOG_INFO(x->Norm());

  LOG_DEBUG(this, "TNS::Solve()",
            " #*# end");

}


template class AIChebyshev< LocalMatrix<double>, LocalVector<double>, double >;
template class AIChebyshev< LocalMatrix<float>,  LocalVector<float>, float >;
#ifdef SUPPORT_COMPLEX
template class AIChebyshev< LocalMatrix<std::complex<double> >,  LocalVector<std::complex<double> >, std::complex<double> >;
template class AIChebyshev< LocalMatrix<std::complex<float> >,  LocalVector<std::complex<float> >, std::complex<float> >;
#endif

template class FSAI< LocalMatrix<double>, LocalVector<double>, double >;
template class FSAI< LocalMatrix<float>,  LocalVector<float>, float >;
#ifdef SUPPORT_COMPLEX
template class FSAI< LocalMatrix<std::complex<double> >,  LocalVector<std::complex<double> >, std::complex<double> >;
template class FSAI< LocalMatrix<std::complex<float> >,  LocalVector<std::complex<float> >, std::complex<float> >;
#endif

template class SPAI< LocalMatrix<double>, LocalVector<double>, double >;
template class SPAI< LocalMatrix<float>,  LocalVector<float>, float >;
#ifdef SUPPORT_COMPLEX
template class SPAI< LocalMatrix<std::complex<double> >,  LocalVector<std::complex<double> >, std::complex<double> >;
template class SPAI< LocalMatrix<std::complex<float> >,  LocalVector<std::complex<float> >, std::complex<float> >;
#endif

template class TNS< LocalMatrix<double>, LocalVector<double>, double >;
template class TNS< LocalMatrix<float>,  LocalVector<float>, float >;
#ifdef SUPPORT_COMPLEX
template class TNS< LocalMatrix<std::complex<double> >,  LocalVector<std::complex<double> >, std::complex<double> >;
template class TNS< LocalMatrix<std::complex<float> >,  LocalVector<std::complex<float> >, std::complex<float> >;
#endif

}
