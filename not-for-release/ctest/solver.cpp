#include <iostream>
#include <rocalution.hpp>

#define ValueType double

using namespace rocalution;

int main(int argc, char* argv[]) {

  init_rocalution();

  LocalVector<ValueType> x;
  LocalVector<ValueType> rhs;
  LocalVector<ValueType> sol;
  LocalMatrix<ValueType> mat;

  mat.ReadFileMTX(std::string(argv[1]));

  mat.MoveToAccelerator();
  x.MoveToAccelerator();
  rhs.MoveToAccelerator();
  sol.MoveToAccelerator();

  x.Allocate("x", mat.get_nrow());
  rhs.Allocate("rhs", mat.get_nrow());
  sol.Allocate("sol", mat.get_nrow());

  // b = A*1
  sol.Ones();
  x.SetRandom((ValueType) 0, (ValueType) 2, time(NULL));
  mat.Apply(sol, &rhs);

  // Iterative Linear Solver
  IterativeLinearSolver<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType > *ils = NULL;

  std::string sName = argv[2];
  if (sName == "CG")        ils = new CG<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType>;
  if (sName == "FCG")       ils = new FCG<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType>;
  if (sName == "CR")        ils = new CR<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType>;
  if (sName == "BiCGStab")  ils = new BiCGStab<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType>;
  if (sName == "BiCGStabl") ils = new BiCGStabl<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType>;
  if (sName == "QMRCGStab") ils = new QMRCGStab<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType>;
  if (sName == "IDR")       ils = new IDR<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType>;
  if (sName == "GMRES")     ils = new GMRES<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType>;
  if (sName == "FGMRES")    ils = new FGMRES<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType>;

  // Preconditioner
  Preconditioner<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType > *p = NULL;

  ils->SetOperator(mat);
  ils->Init(1e-8, 0.0, 1e+8, 10000);

  std::string pName = argv[3];

  if (pName == "None")   p = NULL;
  if (pName == "Jacobi") p = new Jacobi<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType>;
  if (pName == "ILU")    p = new ILU<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType>;
  if (pName == "ILUT")   p = new ILUT<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType>;
  if (pName == "IC")     p = new IC<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType>;
  if (pName == "FSAI")   p = new FSAI<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType>;
  if (pName == "SPAI")   p = new SPAI<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType>;
  if (pName == "TNS")    p = new TNS<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType>;
  if (pName == "MCILU")  p = new MultiColoredILU<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType>;

  if (p != NULL)
    ils->SetPreconditioner(*p);

  ils->Build();

  std::string mFormat = argv[4];

  if (mFormat == "CSR") mat.ConvertTo(CSR);
  if (mFormat == "MCSR") mat.ConvertTo(MCSR);
  if (mFormat == "COO") mat.ConvertTo(COO);
  if (mFormat == "ELL") mat.ConvertTo(ELL);
  if (mFormat == "DIA") mat.ConvertTo(DIA);
  if (mFormat == "HYB") mat.ConvertTo(HYB);

  ils->Solve(rhs, &x);

  x.ScaleAdd((ValueType) -1, sol);
  std::cout << "Error Norm = " << x.Norm() << std::endl;

  if (x.Norm() > 1e-5) {
    std::cout << "Test failed." << std::endl;
    exit(-1);
  }

  stop_rocalution();

  return 0;

}
