#include <iostream>
#include <cstdlib>

#include <rocalution.hpp>

using namespace rocalution;

int main(int argc, char* argv[]) {

  if (argc == 1) { 
    std::cerr << argv[0] << " <matrix> [Num threads]" << std::endl;
    exit(1);
  }

  init_rocalution();

  if (argc > 2) {
    set_omp_threads_rocalution(atoi(argv[2]));
  } 

  info_rocalution();

  LocalVector<double> x;
  LocalVector<double> rhs;

  LocalMatrix<double> mat;


  mat.ReadFileMTX(std::string(argv[1]));
  mat.Info();

  x.Allocate("x", mat.get_nrow());
  rhs.Allocate("rhs", mat.get_nrow());

  x.Info();
  rhs.Info();

  rhs.Ones();
  
  mat.Apply(rhs, &x);

  std::cout << "dot=" << x.Dot(rhs) << std::endl;

  mat.ConvertToELL();
  mat.Info();

  mat.MoveToAccelerator();
  x.MoveToAccelerator();
  rhs.MoveToAccelerator();
  mat.Info();

  rhs.Ones();
  
  mat.Apply(rhs, &x);

  std::cout << "dot=" << x.Dot(rhs) << std::endl;

  stop_rocalution();

  return 0;
}
