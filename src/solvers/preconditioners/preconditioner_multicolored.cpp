#include "../../utils/def.hpp"
#include "preconditioner_multicolored.hpp"
#include "preconditioner.hpp"
#include "../solver.hpp"

#include "../../base/local_matrix.hpp"

#include "../../base/local_vector.hpp"

#include "../../utils/log.hpp"
#include "../../utils/allocate_free.hpp"

#include <complex>

namespace paralution {

template <class OperatorType, class VectorType, typename ValueType>
MultiColored<OperatorType, VectorType, ValueType>::MultiColored() {

  LOG_DEBUG(this, "MultiColored::MultiColored()",
            "default constructor");

  this->num_blocks_ = 0 ;
  this->block_sizes_ = NULL ; 

  this->preconditioner_ = NULL;
  this->analyzer_op_ = NULL ;

  this->op_mat_format_ = false;
  this->precond_mat_format_ = CSR;

  this->decomp_ = true;

}

template <class OperatorType, class VectorType, typename ValueType>
MultiColored<OperatorType, VectorType, ValueType>::~MultiColored() {

  LOG_DEBUG(this, "MultiColored::~MultiColored()",
            "destructor");


  this->Clear();

}

template <class OperatorType, class VectorType, typename ValueType>
void MultiColored<OperatorType, VectorType, ValueType>::Clear(void) {

  LOG_DEBUG(this, "MultiColored::Clear()",
            this->build_);

  if (this->build_ == true) {

    delete this->preconditioner_ ;
    this->preconditioner_ = NULL;

    if (this->decomp_ == true) {

      for(int i=0; i<this->num_blocks_; ++i) {

        this->x_block_[i]->Clear();
        delete this->x_block_[i];

        this->diag_block_[i]->Clear();
        delete this->diag_block_[i];
        this->diag_solver_[i]->Clear();
        delete this->diag_solver_[i];

        for(int j=0; j<this->num_blocks_; ++j)
          delete this->preconditioner_block_[i][j];

        delete[] this->preconditioner_block_[i];
      }

      delete[] this->x_block_;
      delete[] this->diag_block_;
      delete[] this->diag_solver_;
      delete[] this->preconditioner_block_;
    }

    if (this->analyzer_op_ != this->op_)
      delete this->analyzer_op_;

    this->analyzer_op_ = NULL ;

    this->x_.Clear();

    this->permutation_.Clear();
    free_host(&this->block_sizes_);      
    this->num_blocks_ = 0 ;

    this->diag_.Clear();

    this->op_mat_format_ = false;
    this->precond_mat_format_ = CSR;

    this->decomp_ = true;

    this->build_ = false;

  }

}

template <class OperatorType, class VectorType, typename ValueType>
void MultiColored<OperatorType, VectorType, ValueType>::SetPrecondMatrixFormat(const unsigned int mat_format) {

  LOG_DEBUG(this, "MultiColored::SetPrecondMatrixFormat()",
            mat_format);

  this->op_mat_format_ = true;
  this->precond_mat_format_ = mat_format;

}

template <class OperatorType, class VectorType, typename ValueType>
void MultiColored<OperatorType, VectorType, ValueType>::SetDecomposition(const bool decomp) {

  LOG_DEBUG(this, "MultiColored::SetDecomposition()",
            decomp);

  this->decomp_ = decomp;

}


template <class OperatorType, class VectorType, typename ValueType>
void MultiColored<OperatorType, VectorType, ValueType>::Build_Analyser_(void) {

  LOG_DEBUG(this, "MultiColored::Build_Analyser_()",
            "");
  
  assert(this->op_ != NULL);
  this->analyzer_op_ = NULL;

  this->preconditioner_ = new OperatorType ;
  this->preconditioner_->CloneFrom(*this->op_);
  
  this->permutation_.CloneBackend(*this->op_);

}


template <class OperatorType, class VectorType, typename ValueType>
void MultiColored<OperatorType, VectorType, ValueType>::Analyse_(void) {

  LOG_DEBUG(this, "MultiColored::Analyse_()",
            "");

  
  if (this->analyzer_op_ != NULL) {

    // use extra matrix
    this->analyzer_op_->MultiColoring(this->num_blocks_, 
                                      &this->block_sizes_, 
                                      &this->permutation_);
  } else {

    // op_ matrix
    this->op_->MultiColoring(this->num_blocks_, 
                             &this->block_sizes_, 
                             &this->permutation_);   
  }

}

template <class OperatorType, class VectorType, typename ValueType>
void MultiColored<OperatorType, VectorType, ValueType>::Permute_(void) {

  LOG_DEBUG(this, "MultiColored::Permute_()",
            "");

  assert(this->permutation_.get_size() > 0);

  this->preconditioner_->Permute(this->permutation_);

}


template <class OperatorType, class VectorType, typename ValueType>
void MultiColored<OperatorType, VectorType, ValueType>::Factorize_(void) {
}

template <class OperatorType, class VectorType, typename ValueType>
void MultiColored<OperatorType, VectorType, ValueType>::Decompose_(void) {

  LOG_DEBUG(this, "MultiColored::Decompose_()",
            " * beging");

  if (this->decomp_ == true) {

    assert(this->num_blocks_ > 0);
    assert(this->block_sizes_ != NULL);

    int *offsets = NULL;
    allocate_host(this->num_blocks_+1, &offsets);    

    offsets[0] = 0 ;
    for(int i=0; i<this->num_blocks_; ++i)
      offsets[i+1] = this->block_sizes_[i];

    // sum up
    for(int i=0; i<this->num_blocks_; ++i)
      offsets[i+1] += offsets[i];

    this->diag_solver_ = new Solver<OperatorType, VectorType, ValueType>* [this->num_blocks_];

    this->preconditioner_block_ = new OperatorType** [this->num_blocks_]; 
    for (int i=0; i<this->num_blocks_; ++i) 
      this->preconditioner_block_[i] = new OperatorType* [this->num_blocks_];

    this->x_block_    = new VectorType* [this->num_blocks_];
    this->diag_block_ = new VectorType* [this->num_blocks_];

    for(int i=0; i<this->num_blocks_; ++i)
      for(int j=0; j<this->num_blocks_; ++j) {
        this->preconditioner_block_[i][j] = new LocalMatrix<ValueType>;
        this->preconditioner_block_[i][j]->CloneBackend( *this->op_ );
        
      }

    this->preconditioner_->ExtractSubMatrices(this->num_blocks_,
                                              this->num_blocks_,
                                              offsets,
                                              offsets,
                                              this->preconditioner_block_);

    free_host(&offsets);

    int x_offset = 0;
    for (int i=0; i<this->num_blocks_; ++i) {
      this->diag_block_[i] = new VectorType;
      this->diag_block_[i]->CloneBackend(*this->op_); // clone backend
      this->diag_block_[i]->Allocate("Diagonal preconditioners blocks",
                                     this->block_sizes_[i]);
      
      this->preconditioner_block_[i][i]->ExtractDiagonal(this->diag_block_[i]);

      
      this->x_block_[i] = new VectorType; // empty vector
      
      this->x_block_[i]->CloneBackend(*this->op_); // clone backend
      this->x_block_[i]->Allocate("MultiColored Preconditioner x_block_",
                                  this->block_sizes_[i]);
      
      
      x_offset += this->block_sizes_[i];

      Jacobi<OperatorType, VectorType, ValueType> *jacobi = new Jacobi<OperatorType, VectorType, ValueType>;
      jacobi->SetOperator(*this->preconditioner_block_[i][i]);
      jacobi->Build();
      
      this->diag_solver_[i] = jacobi;
      
      this->preconditioner_block_[i][i]->Clear();
    }
    
    // Clone the format 
    // e.g. the preconditioner block matrices will have the same format as this->op_
    if (this->op_mat_format_ == true)
      for(int i=0; i<this->num_blocks_; ++i)
        for(int j=0; j<this->num_blocks_; ++j)
          this->preconditioner_block_[i][j]->ConvertTo(this->precond_mat_format_);
  
  } else {

    this->diag_.CloneBackend(*this->op_);
    this->preconditioner_->ExtractDiagonal(&this->diag_);

  }

  this->x_.CloneBackend(*this->op_); 
  this->x_.Allocate("Permuted solution vector",
                    this->op_->get_nrow());

  LOG_DEBUG(this, "MultiColored::Decompose_()",
            " * end");

}

template <class OperatorType, class VectorType, typename ValueType>
void MultiColored<OperatorType, VectorType, ValueType>::Build(void) {

  LOG_DEBUG(this, "MultiColored::Build()",
            this->build_ <<
            " #*# begin");

  assert(this->build_ == false);

  assert(this->op_ != NULL);

  this->Build_Analyser_();
  this->Analyse_();

  if ((this->analyzer_op_ != this->op_) &&
      (this->analyzer_op_ != NULL))
    this->analyzer_op_->Clear();

  this->Permute_();
  this->Factorize_();
  this->Decompose_();

  // TODO check for correctness

  //  this->op_->WriteFileMTX("op.mtx");
  //  this->preconditioner_->WriteFileMTX("precond.mtx");

  this->build_ = true ;

  if (this->decomp_ == true) {

    this->preconditioner_->Clear();

  } else {

    this->PostAnalyse_();
  }

  LOG_DEBUG(this, "MultiColored::Build()",
            this->build_ <<
            " #*# end");

}

template <class OperatorType, class VectorType, typename ValueType>
void MultiColored<OperatorType, VectorType, ValueType>::PostAnalyse_(void) {
}

template <class OperatorType, class VectorType, typename ValueType>
void MultiColored<OperatorType, VectorType, ValueType>::Solve(const VectorType &rhs,
                                                              VectorType *x) {

  LOG_DEBUG(this, "MultiColored::Solve()",
            " #*# begin");

  assert(x != NULL);
  assert(x != &rhs);
  assert(this->build_ == true);

  if (this->decomp_ == true) {
    
    // Solve via decomposition

    this->ExtractRHSinX_(rhs, x);
    
    this->SolveL_();
    this->SolveD_();
    this->SolveR_();
    
    this->InsertSolution_(x);  

  } else {

    // Solve directly

    this->Solve_(rhs, x);

  }

  LOG_DEBUG(this, "MultiColored::Solve()",
            " #*# end");

}

template <class OperatorType, class VectorType, typename ValueType>
void MultiColored<OperatorType, VectorType, ValueType>::ExtractRHSinX_(const VectorType &rhs,
                                                                       VectorType *x) {

  LOG_DEBUG(this, "MultiColored::ExtractRHSinX_()",
            "");

  assert(this->build_ == true);

  x->CopyFromPermute(rhs,
                     this->permutation_);

  int x_offset = 0;
  for (int i=0; i<this->num_blocks_; ++i) {

    this->x_block_[i]->CopyFrom(*x,
                                x_offset,
                                0,
                                this->block_sizes_[i]);

    x_offset += this->block_sizes_[i];

  }

}

template <class OperatorType, class VectorType, typename ValueType>
void MultiColored<OperatorType, VectorType, ValueType>::InsertSolution_(VectorType *x) {

  LOG_DEBUG(this, "MultiColored::InsertSolution_()",
            "");

  assert(this->build_ == true);

  int x_offset = 0;
  for (int i=0; i<this->num_blocks_; ++i) {


    this->x_.CopyFrom(*this->x_block_[i],
                      0,
                      x_offset,
                      this->block_sizes_[i]);
    
    x_offset += this->block_sizes_[i];
    
  }

  x->CopyFromPermuteBackward(this->x_,
                             this->permutation_);
  

}

template <class OperatorType, class VectorType, typename ValueType>
void MultiColored<OperatorType, VectorType, ValueType>::MoveToHostLocalData_(void) {

  LOG_DEBUG(this, "MultiColored::MoveToHostLocalData_()",
            this->build_);  

  if (this->build_ == true) {

    this->preconditioner_->MoveToHost();
    
    if (this->decomp_ == true) 
      for(int i=0; i<this->num_blocks_; ++i) {
        
        this->x_block_[i]->MoveToHost();
        this->diag_block_[i]->MoveToHost();
        this->diag_solver_[i]->MoveToHost();
        
        for(int j=0; j<this->num_blocks_; ++j) {
          this->preconditioner_block_[i][j]->MoveToHost();
        }
      }
    
    if ((this->analyzer_op_ != this->op_) &&
        (this->analyzer_op_ != NULL))
      
      this->analyzer_op_->MoveToHost();
  }
    
  this->permutation_.MoveToHost();
  this->x_.MoveToHost();
  

}

template <class OperatorType, class VectorType, typename ValueType>
void MultiColored<OperatorType, VectorType, ValueType>::MoveToAcceleratorLocalData_(void) {

  LOG_DEBUG(this, "MultiColored::MoveToAcceleratorLocalData_()",
            this->build_);

  if (this->build_ == true) {
    
    this->preconditioner_->MoveToAccelerator();
    
    if (this->decomp_ == true) 
      for(int i=0; i<this->num_blocks_; ++i) {
        
        this->x_block_[i]->MoveToAccelerator();
        this->diag_block_[i]->MoveToAccelerator();
        this->diag_solver_[i]->MoveToAccelerator();
        
        for(int j=0; j<this->num_blocks_; ++j) 
          this->preconditioner_block_[i][j]->MoveToAccelerator();
        
      }
    
    if ((this->analyzer_op_ != this->op_) &&
        (this->analyzer_op_ != NULL))
      this->analyzer_op_->MoveToAccelerator();
  }  
  
  this->permutation_.MoveToAccelerator();
  this->x_.MoveToAccelerator();
  
}


template class MultiColored< LocalMatrix<double>, LocalVector<double>, double >;
template class MultiColored< LocalMatrix<float>,  LocalVector<float>, float >;
#ifdef SUPPORT_COMPLEX
template class MultiColored< LocalMatrix<std::complex<double> >,  LocalVector<std::complex<double> >, std::complex<double> >;
template class MultiColored< LocalMatrix<std::complex<float> >,  LocalVector<std::complex<float> >, std::complex<float> >;
#endif

}
