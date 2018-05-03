// #######################################################################################################
// ###                                                                                                 ###
// ###                               rocALUTION FORTRAN PLUG-IN                                        ###
// ###                                                                                                 ###
// ###                                                                                                 ###
// ###     Each function listed here can be executed from Fortran code by simply calling it:           ###
// ###                                                                                                 ###
// ###        C function name: "void rocalution_fortran_function_();"                                  ###
// ###        Fortran syntax: "call rocalution_fortran_function()"                                     ###
// ###                                                                                                 ###
// #######################################################################################################
// ###                                                                                                 ###
// ###     rocalution_init                                                                             ###
// ###        Initalize the rocALUTION backend                                                         ###
// ###                                                                                                 ###
// #######################################################################################################
// ###                                                                                                 ###
// ###     rocalution_stop                                                                             ###
// ###        Stops the rocALUTION backend                                                             ###
// ###                                                                                                 ###
// #######################################################################################################
// ###                                                                                                 ###
// ###     rocalution_fortran_solve_coo_                                                               ###
// ###        Solves a linear system for given COO matrix, rhs, solution vector,                       ###
// ###        solver and preconditioner.                                                               ###
// ###                                                                                                 ###
// ###        input parameters:                                                                        ###
// ###                                                                                                 ###
// ###          n          -  number of matrix rows                                                    ###
// ###          m          -  number of matrix cols                                                    ###
// ###          nnz        -  number of non-zero matrix elements                                       ###
// ###          solver     -  solver string, can be CG,BiCGStab,FixedPoint,GMRES,FGMRES                ###
// ###          mformat    -  matrix format for the solving procedure, can be                          ###
// ###                        CSR,MCSR,BCSR,COO,DIA,ELL,HYB,DENSE                                      ###
// ###          precond    -  preconditioner string, can be None, Jacobi, MultiColoredGS,              ###
// ###                        MultiColoredSGS, ILU, MultiColoredILU, FSAI                              ###
// ###          pformat    -  preconditioner format for MultiColored preconditioners, can be           ###
// ###                        CSR,MCSR,BCSR,COO,DIA,ELL,HYB,DENSE                                      ###
// ###          row        -  matrix row index - see COO format                                        ###
// ###          col        -  matrix col index - see COO format                                        ###
// ###          val        -  matrix values    - see COO format                                        ###
// ###          rhs        -  right hand side vector                                                   ###
// ###          atol       -  absolute tolerance                                                       ###
// ###          rtol       -  relative tolerance                                                       ###
// ###          maxiter    -  maximum iterations allowed                                               ###
// ###          basis      -  Basis size when using GMRES                                              ###
// ###          p          -  ILU(p) factorization based on power                                      ###
// ###                        p needs to be specified when ILU preconditioner is chosen.               ###
// ###          q          -  ILU(p,q)                                                                 ###
// ###                        p and q need to be specified when MultiColoredILU preconditioner         ###
// ###                        is chosen.                                                               ###
// ###                                                                                                 ###
// ###        output parameters:                                                                       ###
// ###                                                                                                 ###
// ###          x          -  solution vector                                                          ###
// ###          iter       -  iteration count                                                          ###
// ###          resnorm    -  residual norm                                                            ###
// ###          err        -  error code                                                               ###
// ###                                                                                                 ###
// #######################################################################################################
// ###                                                                                                 ###
// ###     rocalution_fortran_solve_csr_                                                               ###
// ###        Solves a linear system for given CSR matrix, rhs, solution vector,                       ###
// ###        solver and preconditioner.                                                               ###
// ###                                                                                                 ###
// ###        input parameters:                                                                        ###
// ###                                                                                                 ###
// ###          n          -  number of matrix rows                                                    ###
// ###          m          -  number of matrix cols                                                    ###
// ###          nnz        -  number of non-zero matrix elements                                       ###
// ###          solver     -  solver string, can be CG,BiCGStab,FixedPoint,GMRES                       ###
// ###          mformat    -  matrix format for the solving procedure, can be                          ###
// ###                        CSR,MCSR,BCSR,COO,DIA,ELL,HYB,DENSE                                      ###
// ###          precond    -  preconditioner string, can be None, Jacobi, MultiColoredGS,              ###
// ###                        MultiColoredSGS, ILU, MultiColoredILU                                    ###
// ###          pformat    -  preconditioner format for MultiColored preconditioners, can be           ###
// ###                        CSR,MCSR,BCSR,COO,DIA,ELL,HYB,DENSE                                      ###
// ###          row_offset -  matrix row offset - see CSR format                                       ###
// ###          col        -  matrix col index  - see CSR format                                       ###
// ###          val        -  matrix values     - see CSR format                                       ###
// ###          rhs        -  right hand side vector                                                   ###
// ###          atol       -  absolute tolerance                                                       ###
// ###          rtol       -  relative tolerance                                                       ###
// ###          maxiter    -  maximum iterations allowed                                               ###
// ###          basis      -  Basis size when using GMRES                                              ###
// ###          p          -  ILU(p) factorization based on power                                      ###
// ###                        p needs to be specified when ILU preconditioner is chosen.               ###
// ###          q          -  ILU(p,q)                                                                 ###
// ###                        p and q need to be specified when MultiColoredILU preconditioner         ###
// ###                        is chosen.                                                               ###
// ###                                                                                                 ###
// ###        output parameters:                                                                       ###
// ###                                                                                                 ###
// ###          x          -  solution vector                                                          ###
// ###          iter       -  iteration count                                                          ###
// ###          resnorm    -  residual norm                                                            ###
// ###          err        -  error code                                                               ###
// ###                                                                                                 ###
// #######################################################################################################
// ###                                                                                                 ###
// ###        error codes:                                                                             ###
// ###                                                                                                 ###
// ###           0  -  no error                                                                        ###
// ###           1  -  absolute tolerance is reached                                                   ###
// ###           2  -  relative tolerance is reached                                                   ###
// ###           3  -  divergence tolerance is reached                                                 ###
// ###           4  -  max iter is reached                                                             ###
// ###           5  -  invalid solver                                                                  ###
// ###           6  -  invalid preconditioner                                                          ###
// ###           7  -  invalid matrix format                                                           ###
// ###           8  -  invalid preconditioner format                                                   ###
// ###                                                                                                 ###
// #######################################################################################################
#include <rocalution.hpp>

extern "C" 
{
  void rocalution_init(void);
  void rocalution_stop(void);
  void rocalution_fortran_solve_coo(int, int, int, char*, char*, char*, char*, const int*, const int*,
                                    const double*, const double*, double, double, double, int, int,
                                    int, int, double*, int&, double&, int&);
  void rocalution_fortran_solve_csr(int, int, int, char*, char*, char*, char*, const int*, const int*,
                                    const double*, const double*, double, double, double, int, int,
                                    int, int, double*, int&, double&, int&);
}

enum _solver_type{
  BiCGStab,
  CG,
  FixedPoint,
  GMRES,
  FGMRES
};

enum _precond_type{
  None,
  Jacobi,
  MultiColoredGS,
  MultiColoredSGS,
  ILU,
  MultiColoredILU,
  FSAI
};

void rocalution_fortran_solve(char*, char*, char*, char*, double, double, double, int, int, int, int,
                              rocalution::LocalMatrix<double>*, rocalution::LocalVector<double>*,
                              rocalution::LocalVector<double>*, int&, double&, int&);

/// Initializes the rocALUTION backend
void rocalution_init(void) {

  rocalution::init_rocalution();
  rocalution::info_rocalution();

}

/// Stops the rocALUTION backend
void rocalution_stop(void) {

  rocalution::stop_rocalution();

}

/// Solves a linear system for given COO matrix, rhs, solution vector, solver and preconditioner.
void rocalution_fortran_solve_coo(int n, int m, int nnz, char *solver, char *mformat, char *precond, char *pformat,
                                  const int *fortran_row, const int *fortran_col, const double *fortran_val,
                                  const double *fortran_rhs, double atol, double rtol, double div, int maxiter,
                                  int basis, int p, int q, double *fortran_x, int &iter, double &resnorm, int &err) {

  rocalution::LocalVector<double> rocalution_x;
  rocalution::LocalVector<double> rocalution_rhs;
  rocalution::LocalMatrix<double> rocalution_mat;

  int *row = NULL;
  int *col = NULL;
  double *val = NULL;

  rocalution::allocate_host(nnz, &row);
  rocalution::allocate_host(nnz, &col);
  rocalution::allocate_host(nnz, &val);

  double *in_rhs = NULL;
  double *in_x = NULL;
  rocalution::allocate_host(m, &in_rhs);
  rocalution::allocate_host(n, &in_x);

  for (int i=0; i<m; ++i)
    in_rhs[i] = fortran_rhs[i];
  for (int i=0; i<n; ++i)
    in_x[i] = fortran_x[i];

  rocalution_rhs.SetDataPtr(&in_rhs, "Imported Fortran rhs", m);
  rocalution_x.SetDataPtr(&in_x, "Imported Fortran x", n);

  // Copy matrix so we can convert it to any other format without breaking the fortran code
  for (int i=0; i<nnz; ++i) {
    // Shift row and col index since Fortran arrays start at 1
    row[i] = fortran_row[i] - 1;
    col[i] = fortran_col[i] - 1;
    val[i] = fortran_val[i];
  }

  // Allocate rocalution data structures
  rocalution_mat.SetDataPtrCOO(&row, &col, &val, "Imported Fortran COO Matrix", nnz, n, m);
  rocalution_mat.ConvertToCSR();
  rocalution_mat.info();

  rocalution_fortran_solve(solver, mformat, precond, pformat, atol, rtol, div, maxiter, basis, p, q,
                           &rocalution_mat, &rocalution_rhs, &rocalution_x, iter, resnorm, err);

  rocalution_x.MoveToHost();
  rocalution_x.LeaveDataPtr(&in_x);

  for (int i=0; i<n; ++i)
    fortran_x[i] = in_x[i];

  delete [] in_x;

}


/// Solves a linear system for given CSR matrix, rhs, solution vector, solver and preconditioner.
void rocalution_fortran_solve_csr(int n, int m, int nnz, char *solver, char *mformat, char *precond, char *pformat,
                                  const int *fortran_row_offset, const int *fortran_col, const double *fortran_val,
                                  const double *fortran_rhs, double atol, double rtol, double div, int maxiter,
                                  int basis, int p, int q, double *fortran_x, int &iter, double &resnorm, int &err) {

  rocalution::LocalVector<double> rocalution_x;
  rocalution::LocalVector<double> rocalution_rhs;
  rocalution::LocalMatrix<double> rocalution_mat;

  int *row_offset = NULL;
  int *col        = NULL;
  double *val     = NULL;

  rocalution::allocate_host(n+1, &row_offset);
  rocalution::allocate_host(nnz, &col);
  rocalution::allocate_host(nnz, &val);

  double *in_rhs = NULL;
  double *in_x = NULL;
  rocalution::allocate_host(m, &in_rhs);
  rocalution::allocate_host(n, &in_x);

  for (int i=0; i<m; ++i)
    in_rhs[i] = fortran_rhs[i];
  for (int i=0; i<n; ++i)
    in_x[i] = fortran_x[i];

  rocalution_rhs.SetDataPtr(&in_rhs, "Imported Fortran rhs", m);
  rocalution_x.SetDataPtr(&in_x, "Imported Fortran x", n);

  // Copy matrix so we can convert it to any other format without breaking the fortran code
  // Shift since Fortran arrays start at 1
  for (int i=0; i<n+1; ++i)
    row_offset[i] = fortran_row_offset[i] - 1;

  for (int i=0; i<nnz; ++i) {
    col[i] = fortran_col[i] - 1;
    val[i] = fortran_val[i];
  }

  // Allocate rocalution data structures
  rocalution_mat.SetDataPtrCSR(&row_offset, &col, &val, "Imported Fortran CSR Matrix", nnz, n, m);
  rocalution_mat.info();

  rocalution_fortran_solve(solver, mformat, precond, pformat, atol, rtol, div, maxiter, basis, p, q,
                           &rocalution_mat, &rocalution_rhs, &rocalution_x, iter, resnorm, err);

  rocalution_x.MoveToHost();
  rocalution_x.LeaveDataPtr(&in_x);

  for (int i=0; i<n; ++i)
    fortran_x[i] = in_x[i];

  delete [] in_x;

}


void rocalution_fortran_solve(char *solver, char *mformat, char *precond, char *pformat, double atol, double rtol,
                              double div, int maxiter, int basis, int p, int q, rocalution::LocalMatrix<double> *mat,
                              rocalution::LocalVector<double> *rhs, rocalution::LocalVector<double> *x,
                              int &iter, double &resnorm, int &err) {

  // Iterative Linear Solver and Preconditioner
  rocalution::IterativeLinearSolver<rocalution::LocalMatrix<double>, rocalution::LocalVector<double>, double > *ls = NULL;
  rocalution::Preconditioner<rocalution::LocalMatrix<double>, rocalution::LocalVector<double>, double >        *pr = NULL;

  _solver_type psolver;
  _precond_type pprecond;
  rocalution::_matrix_format matformat;
  rocalution::_matrix_format preformat;

  // Prepare solver type
  if      (std::string(solver) == "BiCGStab")   psolver = BiCGStab;
  else if (std::string(solver) == "CG")         psolver = CG;
  else if (std::string(solver) == "FixedPoint") psolver = FixedPoint;
  else if (std::string(solver) == "GMRES")      psolver = GMRES;
  else if (std::string(solver) == "FGMRES")     psolver = FGMRES;
  else {
    err = 5;
    return;
  }

  // Prepare preconditioner type
  if      (std::string(precond) == "None" ||
            std::string(precond) == "")               pprecond = None;
  else if (std::string(precond) == "Jacobi")          pprecond = Jacobi;
  else if (std::string(precond) == "MultiColoredGS")  pprecond = MultiColoredGS;
  else if (std::string(precond) == "MultiColoredSGS") pprecond = MultiColoredSGS;
  else if (std::string(precond) == "ILU")             pprecond = ILU;
  else if (std::string(precond) == "MultiColoredILU") pprecond = MultiColoredILU;
  else if (std::string(precond) == "FSAI")            pprecond = FSAI;
  else {
    err = 6;
    return;
  }

  // Prepare matrix format for solving
  if      (std::string(mformat) == "CSR")   matformat = rocalution::CSR;
  else if (std::string(mformat) == "MCSR")  matformat = rocalution::MCSR;
  else if (std::string(mformat) == "BCSR")  matformat = rocalution::BCSR;
  else if (std::string(mformat) == "COO")   matformat = rocalution::COO;
  else if (std::string(mformat) == "DIA")   matformat = rocalution::DIA;
  else if (std::string(mformat) == "ELL")   matformat = rocalution::ELL;
  else if (std::string(mformat) == "HYB")   matformat = rocalution::HYB;
  else if (std::string(mformat) == "DENSE") matformat = rocalution::DENSE;
  else {
    err = 7;
    return;
  }

  // Prepare preconditioner format
  if      (std::string(pformat) == "CSR")   preformat = rocalution::CSR;
  else if (std::string(pformat) == "MCSR")  preformat = rocalution::MCSR;
  else if (std::string(pformat) == "BCSR")  preformat = rocalution::BCSR;
  else if (std::string(pformat) == "COO")   preformat = rocalution::COO;
  else if (std::string(pformat) == "DIA")   preformat = rocalution::DIA;
  else if (std::string(pformat) == "ELL")   preformat = rocalution::ELL;
  else if (std::string(pformat) == "HYB")   preformat = rocalution::HYB;
  else if (std::string(pformat) == "DENSE") preformat = rocalution::DENSE;
  else if (pprecond != None) {
    err = 8;
    return;
  }

  // Switch for solver selection
  switch(psolver) {
    case BiCGStab:
      ls = new rocalution::BiCGStab<rocalution::LocalMatrix<double>, rocalution::LocalVector<double>, double >;
      break;
    case CG:
      ls = new rocalution::CG<rocalution::LocalMatrix<double>, rocalution::LocalVector<double>, double >;
      break;
    case FixedPoint:
      ls = new rocalution::FixedPoint<rocalution::LocalMatrix<double>, rocalution::LocalVector<double>, double >;
      break;
    case GMRES:
      rocalution::GMRES<rocalution::LocalMatrix<double>, rocalution::LocalVector<double>, double > *ls_gmres;
      ls_gmres = new rocalution::GMRES<rocalution::LocalMatrix<double>, rocalution::LocalVector<double>, double >;
      ls_gmres->SetBasisSize(basis);
      ls = ls_gmres;
      break;
    case FGMRES:
      rocalution::FGMRES<rocalution::LocalMatrix<double>, rocalution::LocalVector<double>, double > *ls_fgmres;
      ls_fgmres = new rocalution::FGMRES<rocalution::LocalMatrix<double>, rocalution::LocalVector<double>, double >;
      ls_fgmres->SetBasisSize(basis);
      ls = ls_fgmres;
      break;
  }

  // Switch for preconditioner selection
  switch(pprecond) {
    case None:
      break;
    case Jacobi:
      pr = new rocalution::Jacobi<rocalution::LocalMatrix<double>, rocalution::LocalVector<double>, double >;
      ls->SetPreconditioner(*pr);
      break;
    case MultiColoredGS:
      rocalution::MultiColoredGS<rocalution::LocalMatrix<double>, rocalution::LocalVector<double>, double > *p_mcgs;
      p_mcgs = new rocalution::MultiColoredGS<rocalution::LocalMatrix<double>, rocalution::LocalVector<double>, double >;
      p_mcgs->SetPrecondMatrixFormat(preformat);
      pr = p_mcgs;
      ls->SetPreconditioner(*pr);
      break;
    case MultiColoredSGS:
      rocalution::MultiColoredSGS<rocalution::LocalMatrix<double>, rocalution::LocalVector<double>, double > *p_mcsgs;
      p_mcsgs = new rocalution::MultiColoredSGS<rocalution::LocalMatrix<double>, rocalution::LocalVector<double>, double >;
      p_mcsgs->SetPrecondMatrixFormat(preformat);
      pr = p_mcsgs;
      ls->SetPreconditioner(*pr);
      break;
    case ILU:
      rocalution::ILU<rocalution::LocalMatrix<double>, rocalution::LocalVector<double>, double > *p_ilu;
      p_ilu = new rocalution::ILU<rocalution::LocalMatrix<double>, rocalution::LocalVector<double>, double >;
      p_ilu->Set(p);
      pr = p_ilu;
      ls->SetPreconditioner(*pr);
      break;
    case MultiColoredILU:
      rocalution::MultiColoredILU<rocalution::LocalMatrix<double>, rocalution::LocalVector<double>, double > *p_mcilu;
      p_mcilu = new rocalution::MultiColoredILU<rocalution::LocalMatrix<double>, rocalution::LocalVector<double>, double >;
      p_mcilu->Set(p,q);
      p_mcilu->SetPrecondMatrixFormat(preformat);
      pr = p_mcilu;
      ls->SetPreconditioner(*pr);
      break;
    case FSAI:
      rocalution::FSAI<rocalution::LocalMatrix<double>, rocalution::LocalVector<double>, double > *p_fsai;
      p_fsai = new rocalution::FSAI<rocalution::LocalMatrix<double>, rocalution::LocalVector<double>, double >;
      p_fsai->Set(q);
      p_fsai->SetPrecondMatrixFormat(preformat);
      pr = p_fsai;
      ls->SetPreconditioner(*pr);
      break;
  }

  ls->SetOperator(*mat);
  ls->Init(atol, rtol, div, maxiter);

  ls->MoveToAccelerator();
  mat->MoveToAccelerator();
  x->MoveToAccelerator();
  rhs->MoveToAccelerator();

  ls->Build();

  switch(matformat) {
    case rocalution::CSR:
      mat->ConvertToCSR();
      break;
    case rocalution::MCSR:
      mat->ConvertToMCSR();
      break;
    case rocalution::BCSR:
      mat->ConvertToBCSR();
      break;
    case rocalution::COO:
      mat->ConvertToCOO();
      break;
    case rocalution::DIA:
      mat->ConvertToDIA();
      break;
    case rocalution::ELL:
      mat->ConvertToELL();
      break;
    case rocalution::HYB:
      mat->ConvertToHYB();
      break;
    case rocalution::DENSE:
      mat->ConvertToDENSE();
      break;
  }

  x->info();
  rhs->info();
  mat->info();

  ls->Solve(*rhs, x);

  iter = ls->GetIterationCount();
  resnorm = ls->GetCurrentResidual();
  err = ls->GetSolverStatus();

  ls->Clear();
  delete ls;
  if ( pr != NULL ) {
    pr->Clear();
    delete pr;
  }

}
