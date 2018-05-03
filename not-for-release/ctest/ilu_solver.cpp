#include <iostream>
#include <paralution.hpp>

#define ValueType double

using namespace paralution;

int main(int argc, char* argv[]) {

  init_paralution();

  LocalVector<ValueType> x;
  LocalVector<ValueType> rhs;
  LocalVector<ValueType> sol;
  LocalMatrix<ValueType> mat;

  mat.MoveToAccelerator();
  x.MoveToAccelerator();
  rhs.MoveToAccelerator();
  sol.MoveToAccelerator();

  mat.ReadFileMTX(std::string(argv[1]));

  x.Allocate("x", mat.get_nrow());
  rhs.Allocate("rhs", mat.get_nrow());
  sol.Allocate("sol", mat.get_nrow());

  // b = A*1
  sol.Ones();
  x.SetRandom((ValueType) 0, (ValueType) 2, time(NULL));
  mat.Apply(sol, &rhs);

  // Solver
  GMRES<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType> ls;
  ILU<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType> p;

  ls.SetOperator(mat);
  ls.Init(1e-8, 0.0, 1e+8, 10000);

  int ilu_pq = atoi(argv[3]);

  p.Set(ilu_pq);

  ls.SetPreconditioner(p);

  ls.Build();

  std::string mFormat = argv[2];

  if (mFormat == "CSR")  mat.ConvertTo(CSR); 
  if (mFormat == "MCSR") mat.ConvertTo(MCSR);
  if (mFormat == "COO")  mat.ConvertTo(COO);
  if (mFormat == "ELL")  mat.ConvertTo(ELL);
  if (mFormat == "DIA")  mat.ConvertTo(DIA);
  if (mFormat == "HYB")  mat.ConvertTo(HYB);

  mat.info();

  ls.Solve(rhs, &x);

  x.ScaleAdd((ValueType) -1, sol);
  std::cout << "Error Norm = " << x.Norm() << std::endl;

  if (x.Norm() > 1e-4) {
    std::cout << "Test failed." << std::endl;
    exit(-1);
  }

  stop_paralution();

  return 0;

}