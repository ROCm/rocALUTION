#include <iostream>
#include <cstdlib>

#include <paralution.hpp>

using namespace paralution;

int main(int argc, char* argv[]) {

  if (argc == 1) { 
    std::cerr << argv[0] << " <matrix> [Num threads]" << std::endl;
    exit(1);
  }

  init_paralution();

  if (argc > 2) {
    set_omp_threads_paralution(atoi(argv[2]));
  } 

  info_paralution();

  LocalVector<double> x;
  LocalVector<double> rhs;

  LocalMatrix<double> mat;


  mat.ReadFileMTX(std::string(argv[1]));
  mat.info();

  x.Allocate("x", mat.get_nrow());
  rhs.Allocate("rhs", mat.get_nrow());

  x.info();
  rhs.info();

  rhs.Ones();
  
  mat.Apply(rhs, &x);

  std::cout << "dot=" << x.Dot(rhs) << std::endl;

  mat.ConvertToELL();
  mat.info();

  mat.MoveToAccelerator();
  x.MoveToAccelerator();
  rhs.MoveToAccelerator();
  mat.info();

  rhs.Ones();
  
  mat.Apply(rhs, &x);

  std::cout << "dot=" << x.Dot(rhs) << std::endl;

  stop_paralution();

  return 0;
}
