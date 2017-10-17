#include "../../utils/def.hpp"
#include "preconditioner_blockprecond.hpp"
#include "preconditioner.hpp"
#include "../solver.hpp"

#include "../../base/local_matrix.hpp"

#include "../../base/local_vector.hpp"

#include "../../utils/allocate_free.hpp"
#include "../../utils/log.hpp"

#include <complex>

namespace paralution {

template <class OperatorType, class VectorType, typename ValueType>
BlockPreconditioner<OperatorType, VectorType, ValueType>::BlockPreconditioner() {

  LOG_DEBUG(this, "BlockPreconditioner::BlockPreconditioner()",
            "default constructor");

  this->num_blocks_ = 0 ;
  this->block_sizes_ = NULL ; 

  this->op_mat_format_ = false;
  this->precond_mat_format_ = CSR;

  this->diag_solve_ = false;
  this->A_last_ = NULL;

}

template <class OperatorType, class VectorType, typename ValueType>
BlockPreconditioner<OperatorType, VectorType, ValueType>::~BlockPreconditioner() {

  LOG_DEBUG(this, "BlockPreconditioner::~BlockPreconditioner()",
            "destructor");

  this->Clear();

}

template <class OperatorType, class VectorType, typename ValueType>
void BlockPreconditioner<OperatorType, VectorType, ValueType>::Clear(void) {

  LOG_DEBUG(this, "BlockPreconditioner::Clear()",
            this->build_);

  if (this->build_ == true) {


    for(int i=0; i<this->num_blocks_; ++i) {

      this->x_block_[i]->Clear();
      this->tmp_block_[i]->Clear();
      delete this->x_block_[i];
      delete this->tmp_block_[i];

      if (this->D_solver_[i] != NULL) {
        this->D_solver_[i]->Clear();
        this->D_solver_[i] = NULL;
      }

        for(int j=0; j<this->num_blocks_; ++j)
          delete this->A_block_[i][j];

        delete[] this->A_block_[i];     
        this->A_block_[i] = NULL;

    }

    delete[] this->x_block_;
    delete[] this->tmp_block_;
    delete[] this->A_block_;
    delete[] this->D_solver_;
    

    free_host(&this->block_sizes_);      
    this->num_blocks_ = 0 ;

    this->op_mat_format_ = false;
    this->precond_mat_format_ = CSR;

    this->permutation_.Clear();
    this->x_.Clear();

    this->build_ = false ;

  }

}

template <class OperatorType, class VectorType, typename ValueType>
void BlockPreconditioner<OperatorType, VectorType, ValueType>::Print(void) const {

  if (this->build_ == true) {

    LOG_INFO("BlockPreconditioner with " << this->num_blocks_ << " blocks:");
    
    for (int i=0; i<this->num_blocks_; ++i)
      this->D_solver_[i]->Print();

    
  } else {

    LOG_INFO("BlockPreconditioner (I)LU preconditioner");
    
  }
}

template <class OperatorType, class VectorType, typename ValueType>
void BlockPreconditioner<OperatorType, VectorType, ValueType>::Set(const int n,
                                                                   const int *size,
                                                                   Solver<OperatorType, VectorType, ValueType> **D_Solver) {

  LOG_DEBUG(this, "BlockPreconditioner::Set()",
            n);

  assert(this->build_ == false);

  assert(n > 0);
  this->num_blocks_ = n;

  this->block_sizes_ = new int[n];
  this->D_solver_ = new Solver<OperatorType, VectorType, ValueType>*[n];

  for (int i=0; i<this->num_blocks_; ++i) {

    this->block_sizes_[i] = size[i];
    this->D_solver_[i] = D_Solver[i];

  }

}

template <class OperatorType, class VectorType, typename ValueType>
void BlockPreconditioner<OperatorType, VectorType, ValueType>::SetDiagonalSolver(void) {

  LOG_DEBUG(this, "BlockPreconditioner::SetDiagonalSolver()",
            "");

  this->diag_solve_ = true;

}

template <class OperatorType, class VectorType, typename ValueType>
void BlockPreconditioner<OperatorType, VectorType, ValueType>::SetLSolver(void) {

  LOG_DEBUG(this, "BlockPreconditioner::SetLSolver()",
            "");

  this->diag_solve_ = false;

}

template <class OperatorType, class VectorType, typename ValueType>
void BlockPreconditioner<OperatorType, VectorType, ValueType>::SetExternalLastMatrix(const OperatorType &mat) {

  LOG_DEBUG(this, "BlockPreconditioner::SetExternalLastMatrix()",
            "");


  assert(this->build_ == false);
  assert(this->A_last_ == NULL);

  this->A_last_ = new OperatorType;
  this->A_last_->CloneBackend(mat);
  this->A_last_->CopyFrom(mat);

}

template <class OperatorType, class VectorType, typename ValueType>
void BlockPreconditioner<OperatorType, VectorType, ValueType>::SetPermutation(const LocalVector<int> &perm) {

  LOG_DEBUG(this, "BlockPreconditioner::SetPermutation()",
            "");

  assert(perm.get_size() > 0);

  this->permutation_.CopyFrom(perm);

}


template <class OperatorType, class VectorType, typename ValueType>
void BlockPreconditioner<OperatorType, VectorType, ValueType>::Build(void) {

  LOG_DEBUG(this, "BlockPreconditioner::Build()",
            this->build_ <<
            " #*# begin");

  assert(this->build_ == false);
  this->build_ = true ;

  assert(this->op_ != NULL);

  this->x_block_ = new VectorType* [this->num_blocks_];
  this->tmp_block_ = new VectorType* [this->num_blocks_];

  for (int i=0; i<this->num_blocks_; ++i) {
    
    this->x_block_[i] = new VectorType;
    this->x_block_[i]->CloneBackend(*this->op_); // clone backend
    this->x_block_[i]->Allocate("Diagonal preconditioners",
                                this->block_sizes_[i]);

    this->tmp_block_[i] = new VectorType;
    this->tmp_block_[i]->CloneBackend(*this->op_); // clone backend
    this->tmp_block_[i]->Allocate("Diagonal preconditioners",
                                  this->block_sizes_[i]);
    
    //    LOG_INFO("Block=" << i << " size=" << this->block_sizes_[i]);

  }

  int *offsets = NULL;
  allocate_host(this->num_blocks_+1, &offsets);    
  
  offsets[0] = 0 ;
  for(int k=0; k<this->num_blocks_; ++k)
    offsets[k+1] = this->block_sizes_[k];
  
  // sum up
  for(int i=0; i<this->num_blocks_; ++i)
    offsets[i+1] += offsets[i];
  
  this->A_block_ = new OperatorType** [this->num_blocks_]; 
  for (int k=0; k<this->num_blocks_; ++k) 
    this->A_block_[k] = new OperatorType* [this->num_blocks_];
  
  
  for(int k=0; k<this->num_blocks_; ++k)
    for(int j=0; j<this->num_blocks_; ++j) {
      this->A_block_[k][j] = new OperatorType;
      this->A_block_[k][j]->CloneBackend( *this->op_ );
      
    }


  if (this->permutation_.get_size() > 0) {

    // with permutation

    assert(this->permutation_.get_size() == this->op_->get_nrow());
    assert(this->permutation_.get_size() == this->op_->get_ncol());

    this->permutation_.CloneBackend(*this->op_);

    LocalMatrix<ValueType> perm_op_;
    perm_op_.CloneFrom(*this->op_);
    perm_op_.Permute(this->permutation_);

    perm_op_.ExtractSubMatrices(this->num_blocks_,
                                this->num_blocks_,
                                offsets,
                                offsets,
                                this->A_block_);

    this->x_.CloneBackend(*this->op_);
    this->x_.Allocate("x (not permuted)",
                      this->op_->get_nrow());


  } else {

    // without permutation

    this->op_->ExtractSubMatrices(this->num_blocks_,
                                  this->num_blocks_,
                                  offsets,
                                  offsets,
                                  this->A_block_);

  }

  free_host(&offsets);

  if (this->A_last_ != NULL) {

    assert(this->A_block_[this->num_blocks_-1][this->num_blocks_-1]->get_nrow() ==
           this->A_last_->get_nrow());
    assert(this->A_block_[this->num_blocks_-1][this->num_blocks_-1]->get_ncol() ==
           this->A_last_->get_ncol());

    //    this->A_block_[this->num_blocks_-1][this->num_blocks_-1]->info();
    this->A_block_[this->num_blocks_-1][this->num_blocks_-1]->Clear();
    delete this->A_block_[this->num_blocks_-1][this->num_blocks_-1];

    this->A_block_[this->num_blocks_-1][this->num_blocks_-1] = this->A_last_;
    //    this->A_block_[this->num_blocks_-1][this->num_blocks_-1]->info();
    this->A_last_ = NULL;
  }

  for(int i=0; i<this->num_blocks_; ++i) {

    this->D_solver_[i]->SetOperator(*this->A_block_[i][i]);
    this->D_solver_[i]->Build();

  }

  for (int i=0; i<this->num_blocks_; ++i) {

    // Clean U part
    for (int j=i+1; j<this->num_blocks_; ++j)
      this->A_block_[i][j]->Clear();

    // Clean L part if not needed
    if (this->diag_solve_ == true)
      for (int j=0; j<i; ++j)
        this->A_block_[i][j]->Clear();

  }

  LOG_DEBUG(this, "BlockPreconditioner::Build()",
            this->build_ <<
            " #*# end");

}


template <class OperatorType, class VectorType, typename ValueType>
void BlockPreconditioner<OperatorType, VectorType, ValueType>::Solve(const VectorType &rhs,
                                                                  VectorType *x) {

  LOG_DEBUG(this, ":BlockPreconditioner:Solve()",
            " #*# begin");
  
  assert(this->build_ == true);

  // Extract RHS into the solution

  if (this->permutation_.get_size() > 0) {

    assert(this->permutation_.get_size() == this->x_.get_size());
    assert(this->op_->get_nrow() == this->x_.get_size());
    assert(this->x_.get_size() == x->get_size());
    assert(this->x_.get_size() == rhs.get_size());

    // with permutation
    this->x_.CopyFromPermute(rhs,
                             this->permutation_);


    int x_offset = 0;
    for (int i=0; i<this->num_blocks_; ++i) {
      
      this->x_block_[i]->CopyFrom(this->x_,
                                  x_offset,
                                  0,
                                  this->block_sizes_[i]);
      
      x_offset += this->block_sizes_[i];

    }



  } else {

    // without permutaion
    x->CopyFrom(rhs);

    int x_offset = 0;
    for (int i=0; i<this->num_blocks_; ++i) {

      this->x_block_[i]->CopyFrom(*x,
                                  x_offset,
                                  0,
                                  this->block_sizes_[i]);
      
      x_offset += this->block_sizes_[i];
      
    }
  }


  
  // Solve L
  for (int i=0; i<this->num_blocks_; ++i){

    if (this->diag_solve_ == false) {
      for (int j=0; j<i; ++j)
        this->A_block_[i][j]->ApplyAdd(*this->x_block_[j],
                                       ValueType(-1.0),   
                                       this->x_block_[i]);
    }

    this->D_solver_[i]->SolveZeroSol(*this->x_block_[i],
                                     this->tmp_block_[i]);
    
    this->x_block_[i]->CopyFrom(*this->tmp_block_[i]);
    
    
  }


  // Insert Solution
  
  if (this->permutation_.get_size() > 0) {
    // with permutation


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
    
  } else {

    // without permutaion
    
    int x_offset = 0;
    for (int i=0; i<this->num_blocks_; ++i) {
      
      
      x->CopyFrom(*this->x_block_[i],
                  0,
                  x_offset,
                  this->block_sizes_[i]);
      
      x_offset += this->block_sizes_[i];
      
    }    
    

  }
  

  LOG_DEBUG(this, "BlockPreconditioner::Solve()",
            " #*# end");

}

template <class OperatorType, class VectorType, typename ValueType>
void BlockPreconditioner<OperatorType, VectorType, ValueType>::MoveToHostLocalData_(void) {

  LOG_DEBUG(this, "BlockPreconditioner::MoveToHostLocalData_()",
            this->build_);  

  if (this->build_ == true) {

    for(int i=0; i<this->num_blocks_; ++i) {
        
      this->x_block_[i]->MoveToHost();
      this->tmp_block_[i]->MoveToHost();
      this->D_solver_[i]->MoveToHost();
        
      for(int j=0; j<this->num_blocks_; ++j) 
        this->A_block_[i][j]->MoveToHost();
        
      }

    this->permutation_.MoveToHost();
    this->x_.MoveToHost();
  }

}

template <class OperatorType, class VectorType, typename ValueType>
void BlockPreconditioner<OperatorType, VectorType, ValueType>::MoveToAcceleratorLocalData_(void) {

  LOG_DEBUG(this, "BlockPreconditioner::MoveToAcceleratorLocalData_()",
            this->build_);

  if (this->build_ == true) {

    for(int i=0; i<this->num_blocks_; ++i) {
        
      this->x_block_[i]->MoveToAccelerator();
      this->tmp_block_[i]->MoveToAccelerator();
      this->D_solver_[i]->MoveToAccelerator();
        
      for(int j=0; j<this->num_blocks_; ++j) 
        this->A_block_[i][j]->MoveToAccelerator();
        
      }

    this->permutation_.MoveToAccelerator();
    this->x_.MoveToAccelerator();
  }

}


template class BlockPreconditioner< LocalMatrix<double>, LocalVector<double>, double >;
template class BlockPreconditioner< LocalMatrix<float>,  LocalVector<float>, float >;
#ifdef SUPPORT_COMPLEX
template class BlockPreconditioner< LocalMatrix<std::complex<double> >,  LocalVector<std::complex<double> >, std::complex<double> >;
template class BlockPreconditioner< LocalMatrix<std::complex<float> >,  LocalVector<std::complex<float> >, std::complex<float> >;
#endif

}
