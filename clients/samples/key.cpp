#include <iostream>
#include <rocalution.hpp>

using namespace rocalution;

int main(int argc, char* argv[]) {

  if (argc == 1) { 
    std::cerr << argv[0] << " <matrix>" << std::endl;
    exit(1);
  }

  init_rocalution();

  LocalMatrix<double> mat;

  mat.ReadFileMTX(std::string(argv[1]));

  mat.Info();

  long int row_key;
  long int col_key;
  long int val_key;
  
  mat.Key(row_key,
          col_key,
          val_key);

  std::cout << "Row key = " << row_key << std::endl
            << "Col key = " << col_key << std::endl
            << "Val key = " << val_key << std::endl;

  stop_rocalution();

  return 0;
}
