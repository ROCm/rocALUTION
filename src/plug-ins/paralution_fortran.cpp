// #######################################################################################################
// ###                                                                                                 ###
// ###                               PARALUTION FORTRAN PLUG-IN                                        ###
// ###                                                                                                 ###
// ###                                                                                                 ###
// ###     Each function listed here can be executed from Fortran code by simply calling it:           ###
// ###                                                                                                 ###
// ###        C function name: "void paralution_fortran_function_();"                                  ###
// ###        Fortran syntax: "call paralution_fortran_function()"                                     ###
// ###                                                                                                 ###
// #######################################################################################################
// ###                                                                                                 ###
// ###     paralution_init                                                                             ###
// ###        Initalize the PARALUTION backend                                                         ###
// ###                                                                                                 ###
// #######################################################################################################
// ###                                                                                                 ###
// ###     paralution_stop                                                                             ###
// ###        Stops the PARALUTION backend                                                             ###
// ###                                                                                                 ###
// #######################################################################################################
// ###                                                                                                 ###
// ###     paralution_fortran_solve_coo_                                                               ###
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
// ###     paralution_fortran_solve_csr_                                                               ###
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
#include <paralution.hpp>

extern "C" 
{
  void paralution_init(void);
  void paralution_stop(void);
  void paralution_fortran_solve_coo(int, int, int, char*, char*, char*, char*, const int*, const int*,
                                    const double*, const double*, double, double, double, int, int,
                                    int, int, double*, int&, double&, int&);
  void paralution_fortran_solve_csr(int, int, int, char*, char*, char*, char*, const int*, const int*,
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

void paralution_fortran_solve(char*, char*, char*, char*, double, double, double, int, int, int, int,
                              paralution::LocalMatrix<double>*, paralution::LocalVector<double>*,
                              paralution::LocalVector<double>*, int&, double&, int&);

/// Initializes the PARALUTION backend
void paralution_init(void) {

  paralution::init_paralution();
  paralution::info_paralution();

}

/// Stops the PARALUTION backend
void paralution_stop(void) {

  paralution::stop_paralution();

}

/// Solves a linear system for given COO matrix, rhs, solution vector, solver and preconditioner.
void paralution_fortran_solve_coo(int n, int m, int nnz, char *solver, char *mformat, char *precond, char *pformat,
                                  const int *fortran_row, const int *fortran_col, const double *fortran_val,
                                  const double *fortran_rhs, double atol, double rtol, double div, int maxiter,
                                  int basis, int p, int q, double *fortran_x, int &iter, double &resnorm, int &err) {

  paralution::LocalVector<double> paralution_x;
  paralution::LocalVector<double> paralution_rhs;
  paralution::LocalMatrix<double> paralution_mat;

  int *row = NULL;
  int *col = NULL;
  double *val = NULL;

  paralution::allocate_host(nnz, &row);
  paralution::allocate_host(nnz, &col);
  paralution::allocate_host(nnz, &val);

  double *in_rhs = NULL;
  double *in_x = NULL;
  paralution::allocate_host(m, &in_rhs);
  paralution::allocate_host(n, &in_x);

  for (int i=0; i<m; ++i)
    in_rhs[i] = fortran_rhs[i];
  for (int i=0; i<n; ++i)
    in_x[i] = fortran_x[i];

  paralution_rhs.SetDataPtr(&in_rhs, "Imported Fortran rhs", m);
  paralution_x.SetDataPtr(&in_x, "Imported Fortran x", n);

  // Copy matrix so we can convert it to any other format without breaking the fortran code
  for (int i=0; i<nnz; ++i) {
    // Shift row and col index since Fortran arrays start at 1
    row[i] = fortran_row[i] - 1;
    col[i] = fortran_col[i] - 1;
    val[i] = fortran_val[i];
  }

  // Allocate paralution data structures
  paralution_mat.SetDataPtrCOO(&row, &col, &val, "Imported Fortran COO Matrix", nnz, n, m);
  paralution_mat.ConvertToCSR();
  paralution_mat.info();

  paralution_fortran_solve(solver, mformat, precond, pformat, atol, rtol, div, maxiter, basis, p, q,
                           &paralution_mat, &paralution_rhs, &paralution_x, iter, resnorm, err);

  paralution_x.MoveToHost();
  paralution_x.LeaveDataPtr(&in_x);

  for (int i=0; i<n; ++i)
    fortran_x[i] = in_x[i];

  delete [] in_x;

}


/// Solves a linear system for given CSR matrix, rhs, solution vector, solver and preconditioner.
void paralution_fortran_solve_csr(int n, int m, int nnz, char *solver, char *mformat, char *precond, char *pformat,
                                  const int *fortran_row_offset, const int *fortran_col, const double *fortran_val,
                                  const double *fortran_rhs, double atol, double rtol, double div, int maxiter,
                                  int basis, int p, int q, double *fortran_x, int &iter, double &resnorm, int &err) {

  paralution::LocalVector<double> paralution_x;
  paralution::LocalVector<double> paralution_rhs;
  paralution::LocalMatrix<double> paralution_mat;

  int *row_offset = NULL;
  int *col        = NULL;
  double *val     = NULL;

  paralution::allocate_host(n+1, &row_offset);
  paralution::allocate_host(nnz, &col);
  paralution::allocate_host(nnz, &val);

  double *in_rhs = NULL;
  double *in_x = NULL;
  paralution::allocate_host(m, &in_rhs);
  paralution::allocate_host(n, &in_x);

  for (int i=0; i<m; ++i)
    in_rhs[i] = fortran_rhs[i];
  for (int i=0; i<n; ++i)
    in_x[i] = fortran_x[i];

  paralution_rhs.SetDataPtr(&in_rhs, "Imported Fortran rhs", m);
  paralution_x.SetDataPtr(&in_x, "Imported Fortran x", n);

  // Copy matrix so we can convert it to any other format without breaking the fortran code
  // Shift since Fortran arrays start at 1
  for (int i=0; i<n+1; ++i)
    row_offset[i] = fortran_row_offset[i] - 1;

  for (int i=0; i<nnz; ++i) {
    col[i] = fortran_col[i] - 1;
    val[i] = fortran_val[i];
  }

  // Allocate paralution data structures
  paralution_mat.SetDataPtrCSR(&row_offset, &col, &val, "Imported Fortran CSR Matrix", nnz, n, m);
  paralution_mat.info();

  paralution_fortran_solve(solver, mformat, precond, pformat, atol, rtol, div, maxiter, basis, p, q,
                           &paralution_mat, &paralution_rhs, &paralution_x, iter, resnorm, err);

  paralution_x.MoveToHost();
  paralution_x.LeaveDataPtr(&in_x);

  for (int i=0; i<n; ++i)
    fortran_x[i] = in_x[i];

  delete [] in_x;

}


void paralution_fortran_solve(char *solver, char *mformat, char *precond, char *pformat, double atol, double rtol,
                              double div, int maxiter, int basis, int p, int q, paralution::LocalMatrix<double> *mat,
                              paralution::LocalVector<double> *rhs, paralution::LocalVector<double> *x,
                              int &iter, double &resnorm, int &err) {

  // Iterative Linear Solver and Preconditioner
  paralution::IterativeLinearSolver<paralution::LocalMatrix<double>, paralution::LocalVector<double>, double > *ls = NULL;
  paralution::Preconditioner<paralution::LocalMatrix<double>, paralution::LocalVector<double>, double >        *pr = NULL;

  _solver_type psolver;
  _precond_type pprecond;
  paralution::_matrix_format matformat;
  paralution::_matrix_format preformat;

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
  if      (std::string(mformat) == "CSR")   matformat = paralution::CSR;
  else if (std::string(mformat) == "MCSR")  matformat = paralution::MCSR;
  else if (std::string(mformat) == "BCSR")  matformat = paralution::BCSR;
  else if (std::string(mformat) == "COO")   matformat = paralution::COO;
  else if (std::string(mformat) == "DIA")   matformat = paralution::DIA;
  else if (std::string(mformat) == "ELL")   matformat = paralution::ELL;
  else if (std::string(mformat) == "HYB")   matformat = paralution::HYB;
  else if (std::string(mformat) == "DENSE") matformat = paralution::DENSE;
  else {
    err = 7;
    return;
  }

  // Prepare preconditioner format
  if      (std::string(pformat) == "CSR")   preformat = paralution::CSR;
  else if (std::string(pformat) == "MCSR")  preformat = paralution::MCSR;
  else if (std::string(pformat) == "BCSR")  preformat = paralution::BCSR;
  else if (std::string(pformat) == "COO")   preformat = paralution::COO;
  else if (std::string(pformat) == "DIA")   preformat = paralution::DIA;
  else if (std::string(pformat) == "ELL")   preformat = paralution::ELL;
  else if (std::string(pformat) == "HYB")   preformat = paralution::HYB;
  else if (std::string(pformat) == "DENSE") preformat = paralution::DENSE;
  else if (pprecond != None) {
    err = 8;
    return;
  }

  // Switch for solver selection
  switch(psolver) {
    case BiCGStab:
      ls = new paralution::BiCGStab<paralution::LocalMatrix<double>, paralution::LocalVector<double>, double >;
      break;
    case CG:
      ls = new paralution::CG<paralution::LocalMatrix<double>, paralution::LocalVector<double>, double >;
      break;
    case FixedPoint:
      ls = new paralution::FixedPoint<paralution::LocalMatrix<double>, paralution::LocalVector<double>, double >;
      break;
    case GMRES:
      paralution::GMRES<paralution::LocalMatrix<double>, paralution::LocalVector<double>, double > *ls_gmres;
      ls_gmres = new paralution::GMRES<paralution::LocalMatrix<double>, paralution::LocalVector<double>, double >;
      ls_gmres->SetBasisSize(basis);
      ls = ls_gmres;
      break;
    case FGMRES:
      paralution::FGMRES<paralution::LocalMatrix<double>, paralution::LocalVector<double>, double > *ls_fgmres;
      ls_fgmres = new paralution::FGMRES<paralution::LocalMatrix<double>, paralution::LocalVector<double>, double >;
      ls_fgmres->SetBasisSize(basis);
      ls = ls_fgmres;
      break;
  }

  // Switch for preconditioner selection
  switch(pprecond) {
    case None:
      break;
    case Jacobi:
      pr = new paralution::Jacobi<paralution::LocalMatrix<double>, paralution::LocalVector<double>, double >;
      ls->SetPreconditioner(*pr);
      break;
    case MultiColoredGS:
      paralution::MultiColoredGS<paralution::LocalMatrix<double>, paralution::LocalVector<double>, double > *p_mcgs;
      p_mcgs = new paralution::MultiColoredGS<paralution::LocalMatrix<double>, paralution::LocalVector<double>, double >;
      p_mcgs->SetPrecondMatrixFormat(preformat);
      pr = p_mcgs;
      ls->SetPreconditioner(*pr);
      break;
    case MultiColoredSGS:
      paralution::MultiColoredSGS<paralution::LocalMatrix<double>, paralution::LocalVector<double>, double > *p_mcsgs;
      p_mcsgs = new paralution::MultiColoredSGS<paralution::LocalMatrix<double>, paralution::LocalVector<double>, double >;
      p_mcsgs->SetPrecondMatrixFormat(preformat);
      pr = p_mcsgs;
      ls->SetPreconditioner(*pr);
      break;
    case ILU:
      paralution::ILU<paralution::LocalMatrix<double>, paralution::LocalVector<double>, double > *p_ilu;
      p_ilu = new paralution::ILU<paralution::LocalMatrix<double>, paralution::LocalVector<double>, double >;
      p_ilu->Set(p);
      pr = p_ilu;
      ls->SetPreconditioner(*pr);
      break;
    case MultiColoredILU:
      paralution::MultiColoredILU<paralution::LocalMatrix<double>, paralution::LocalVector<double>, double > *p_mcilu;
      p_mcilu = new paralution::MultiColoredILU<paralution::LocalMatrix<double>, paralution::LocalVector<double>, double >;
      p_mcilu->Set(p,q);
      p_mcilu->SetPrecondMatrixFormat(preformat);
      pr = p_mcilu;
      ls->SetPreconditioner(*pr);
      break;
    case FSAI:
      paralution::FSAI<paralution::LocalMatrix<double>, paralution::LocalVector<double>, double > *p_fsai;
      p_fsai = new paralution::FSAI<paralution::LocalMatrix<double>, paralution::LocalVector<double>, double >;
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
    case paralution::CSR:
      mat->ConvertToCSR();
      break;
    case paralution::MCSR:
      mat->ConvertToMCSR();
      break;
    case paralution::BCSR:
      mat->ConvertToBCSR();
      break;
    case paralution::COO:
      mat->ConvertToCOO();
      break;
    case paralution::DIA:
      mat->ConvertToDIA();
      break;
    case paralution::ELL:
      mat->ConvertToELL();
      break;
    case paralution::HYB:
      mat->ConvertToHYB();
      break;
    case paralution::DENSE:
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
