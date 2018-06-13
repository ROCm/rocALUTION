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

  LocalVector<double> vec1;
  LocalVector<double> vec2;

  LocalMatrix<double> mat;

  double tick, tack;

  const int max_tests = 200;

  mat.ReadFileCSR(std::string(argv[1]));

  vec1.Allocate("x", mat.GetM());
  vec2.Allocate("rhs", mat.GetM());

  int size = mat.GetM();
  int nnz  = mat.GetNnz();

  vec1.Ones();
  vec2.Zeros();
  mat.Apply(vec1, &vec2);

  mat.MoveToAccelerator();
  vec1.MoveToAccelerator();  
  vec2.MoveToAccelerator();

  mat.Info();    
  vec1.Info();
  vec1.Info();


  std::cout << "----------------------------------------------------" << std::endl;
  std::cout << "Number of tests = " << max_tests << std::endl;


  std::cout << "----------------------------------------------------" << std::endl;
  std::cout << "Stand alone micro benchmarks" << std::endl;

  // Dot product
  // Size = 2*size
  // Flop = 2 per element
  vec1.Dot(vec2);

  _rocalution_sync();
  tick = rocalution_time();

  for (int i=0; i<max_tests; ++i) {
    vec1.Dot(vec2);
    _rocalution_sync();
  }

  _rocalution_sync();
  tack = rocalution_time();
  std::cout << "Dot execution: " << (tack-tick)/max_tests/1e3 << " msec" << "; "
            << max_tests*double(sizeof(double)*(size+size))/(tack-tick)/1000 << " Gbyte/sec; "
            << max_tests*double((2*size))/(tack-tick)/1000 << " GFlop/sec" << std::endl;

  // Reduce
  // Size = size
  // Flop = 1 per element
  vec1.Reduce();

  _rocalution_sync();
  tick = rocalution_time();

  for (int i=0; i<max_tests; ++i) {
    vec1.Reduce();
    _rocalution_sync();
  }

  _rocalution_sync();
  tack = rocalution_time();
  std::cout << "Reduce execution: " << (tack-tick)/max_tests/1e3 << " msec" << "; "
            << max_tests*double(sizeof(double)*(size))/(tack-tick)/1000 << " Gbyte/sec; "
            << max_tests*double((size-1))/(tack-tick)/1000 << " GFlop/sec" << std::endl;

  // Norm
  // Size = size
  // Flop = 2 per element
  vec1.Norm();

  _rocalution_sync();
  tick = rocalution_time();

  for (int i=0; i<max_tests; ++i) {
    vec1.Norm();
    _rocalution_sync();
  }

  _rocalution_sync();
  tack = rocalution_time();
  std::cout << "Norm execution: " << (tack-tick)/max_tests/1e3 << " msec" << "; "
            << max_tests*double(sizeof(double)*(size))/(tack-tick)/1000 << " Gbyte/sec; "
            << max_tests*double((2*size))/(tack-tick)/1000 << " GFlop/sec" << std::endl;

  // Vector Update 1
  // Size = 3xsize
  // Flop = 2 per element
  vec1.ScaleAdd(double(5.5), vec2);

  _rocalution_sync();
  tick = rocalution_time();

  for (int i=0; i<max_tests; ++i) {
    vec1.ScaleAdd(double(5.5), vec2);
    _rocalution_sync();
  }

  _rocalution_sync();
  tack = rocalution_time();
  std::cout << "Vector update (scaleadd) execution: " << (tack-tick)/max_tests/1e3 << " msec" << "; "
            << max_tests*double(sizeof(double)*(size+size+size))/(tack-tick)/1000 << " Gbyte/sec; "
            << max_tests*double((2*size))/(tack-tick)/1000 << " GFlop/sec" << std::endl;



  // Vector Update 2
  // Size = 3*size
  // Flop = 2 per element
  vec1.AddScale(vec2, double(5.5));

  _rocalution_sync();
  tick = rocalution_time();

  for (int i=0; i<max_tests; ++i) {
    vec1.AddScale(vec2, double(5.5));
    _rocalution_sync();
  }

  _rocalution_sync();
  tack = rocalution_time();
  std::cout << "Vector update (addscale) execution: " << (tack-tick)/max_tests/1e3 << " msec" << "; "
            << max_tests*double(sizeof(double)*(size+size+size))/(tack-tick)/1000 << " Gbyte/sec; "
            << max_tests*double((2*size))/(tack-tick)/1000 << " GFlop/sec" << std::endl;



  mat.ConvertToCSR();
  nnz = mat.GetNnz();

  mat.Info();
  // Matrix-Vector Multiplication
  // Size = int(size+nnz) [row_offset + col] + valuetype(2*size+nnz) [in + out + nnz]
  // Flop = 2 per entry (nnz)
  mat.Apply(vec1, &vec2);

  _rocalution_sync();
  tick = rocalution_time();

  for (int i=0; i<max_tests; ++i) {
    mat.Apply(vec1, &vec2);
    _rocalution_sync();
  }

  _rocalution_sync();
  tack = rocalution_time();
  std::cout << "CSR SpMV execution: " << (tack-tick)/max_tests/1e3 << " msec" << "; "
            << max_tests*double((sizeof(double)*(size+size+nnz)+sizeof(int)*(size+nnz)))/(tack-tick)/1000 << " Gbyte/sec; "
            << max_tests*double((2*nnz))/(tack-tick)/1000 << " GFlop/sec" << std::endl;

  mat.ConvertToMCSR();
  nnz = mat.GetNnz();

  mat.Info();
  // Matrix-Vector Multiplication
  // Size = int(size+(nnz-size)) [row_offset + col] + valuetype(2*size+nnz) [in + out + nnz]
  // Flop = 2 per entry (nnz)
  mat.Apply(vec1, &vec2);

  _rocalution_sync();
  tick = rocalution_time();

  for (int i=0; i<max_tests; ++i) {
    mat.Apply(vec1, &vec2);
    _rocalution_sync();
  }

  _rocalution_sync();
  tack = rocalution_time();
  std::cout << "MCSR SpMV execution: " << (tack-tick)/max_tests/1e3 << " msec" << "; "
            << max_tests*double((sizeof(double)*(size+size+nnz-size)+sizeof(int)*(size+nnz)))/(tack-tick)/1000 << " Gbyte/sec; "
            << max_tests*double((2*nnz))/(tack-tick)/1000 << " GFlop/sec" << std::endl;


  mat.ConvertToELL();
  nnz = mat.GetNnz();
 
  mat.Info();
  // Matrix-Vector Multiplication  
  // Size = int(nnz) [col] + valuetype(2*size+nnz) [in + out + nnz]  
  // Flop = 2 per entry (nnz)
  mat.Apply(vec1, &vec2);

  _rocalution_sync();
  tick = rocalution_time();

  for (int i=0; i<max_tests; ++i) {
    mat.Apply(vec1, &vec2);
    _rocalution_sync();
  }

  _rocalution_sync();
  tack = rocalution_time();
  std::cout << "ELL SpMV execution: " << (tack-tick)/max_tests/1e3 << " msec" << "; "
            << max_tests*double((sizeof(double)*(size+size+nnz)+sizeof(int)*(nnz)))/(tack-tick)/1000 << " Gbyte/sec; "
            << max_tests*double((2*nnz))/(tack-tick)/1000 << " GFlop/sec" << std::endl;


  mat.ConvertToCOO();
  nnz = mat.GetNnz();

  mat.Info();
  // Matrix-Vector Multiplication
  // Size = int(2*nnz) + valuetype(2*size+nnz)
  // Flop = 2 per entry (nnz)
  mat.Apply(vec1, &vec2);

  _rocalution_sync();
  tick = rocalution_time();

  for (int i=0; i<max_tests; ++i) {
    mat.Apply(vec1, &vec2);
    _rocalution_sync();
  }

  _rocalution_sync();
  tack = rocalution_time();
  std::cout << "COO SpMV execution: " << (tack-tick)/max_tests/1e3 << " msec" << "; "
            << max_tests*double((sizeof(double)*(size+size+nnz)+sizeof(int)*(2*nnz)))/(tack-tick)/1000 << " Gbyte/sec; "
            << max_tests*double((2*nnz))/(tack-tick)/1000 << " GFlop/sec" << std::endl;


  mat.ConvertToHYB();
  nnz = mat.GetNnz();

  mat.Info();
  // Matrix-Vector Multiplication
  // Size = int(nnz) [col] + valuetype(2*size+nnz) [in + out + nnz]  
  // Flop = 2 per entry (nnz)
  mat.Apply(vec1, &vec2);

  _rocalution_sync();
  tick = rocalution_time();

  for (int i=0; i<max_tests; ++i) {
    mat.Apply(vec1, &vec2);
    _rocalution_sync();
  }

  _rocalution_sync();
  tack = rocalution_time();
  std::cout << "HYB SpMV execution: " << (tack-tick)/max_tests/1e3 << " msec" << "; "
    // like O(ELL)
            << max_tests*double((sizeof(double)*(size+size+nnz)+sizeof(int)*(nnz)))/(tack-tick)/1000 << " Gbyte/sec; "
            << max_tests*double((2*nnz))/(tack-tick)/1000 << " GFlop/sec" << std::endl;



  mat.ConvertToDIA();
  nnz = mat.GetNnz();

  mat.Info();
  // Matrix-Vector Multiplication
  // Size = int(size+nnz) + valuetype(2*size+nnz)
  // Flop = 2 per entry (nnz)
  mat.Apply(vec1, &vec2);

  _rocalution_sync();
  tick = rocalution_time();

  for (int i=0; i<max_tests; ++i) {
    mat.Apply(vec1, &vec2);
    _rocalution_sync();
  }

  _rocalution_sync();
  tack = rocalution_time();
  std::cout << "DIA SpMV execution: " << (tack-tick)/max_tests/1e3 << " msec" << "; "
    // assuming ndiag << size
            << max_tests*double((sizeof(double)*(nnz)))/(tack-tick)/1000 << " Gbyte/sec; "
            << max_tests*double((2*nnz))/(tack-tick)/1000 << " GFlop/sec" << std::endl;




  mat.ConvertToCSR();

  std::cout << "----------------------------------------------------" << std::endl;
  std::cout << "Combined micro benchmarks" << std::endl;

  double dot_tick=0, dot_tack=0;
  double norm_tick=0, norm_tack=0;
  double updatev1_tick=0, updatev1_tack=0;
  double updatev2_tick=0, updatev2_tack=0;
  double spmv_tick=0, spmv_tack=0;

  for (int i=0; i<max_tests; ++i) {

    vec1.Ones();
    vec2.Zeros();
    mat.Apply(vec1, &vec2);


    // Dot product
    // Size = 2*size
    // Flop = 2 per element
    vec1.Dot(vec2);

    _rocalution_sync();
    dot_tick += rocalution_time();

      vec1.Dot(vec2);

    _rocalution_sync();    
    dot_tack += rocalution_time();
    

    vec1.Ones();
    vec2.Zeros();
    mat.Apply(vec1, &vec2);
    
    // Norm
    // Size = size
    // Flop = 2 per element
    vec1.Norm();
    
    _rocalution_sync();
    norm_tick += rocalution_time();

      vec1.Norm();

    _rocalution_sync();
    norm_tack += rocalution_time();
    
    
    vec1.Ones();
    vec2.Zeros();
    mat.Apply(vec1, &vec2);

    // Vector Update 1
    // Size = 3xsize
    // Flop = 2 per element
    vec1.ScaleAdd(double(5.5), vec2);

    _rocalution_sync();
    updatev1_tick += rocalution_time();
    
      vec1.ScaleAdd(double(5.5), vec2);

    _rocalution_sync();
    updatev1_tack += rocalution_time(); 
    
    
    vec1.Ones();
    vec2.Zeros();
    mat.Apply(vec1, &vec2);
  
    // Vector Update 2
    // Size = 3*size
    // Flop = 2 per element
    vec1.AddScale(vec2, double(5.5));

    _rocalution_sync();
    updatev2_tick += rocalution_time();
    
      vec1.AddScale(vec2, double(5.5));

    _rocalution_sync();
    updatev2_tack += rocalution_time();
   
    vec1.Ones();
    vec2.Zeros();
    mat.Apply(vec1, &vec2);
    
    // Matrix-Vector Multiplication
    // Size = int(size+nnz) + valuetype(2*size+nnz)
    // Flop = 2 per entry (nnz)
    mat.Apply(vec1, &vec2);

    _rocalution_sync();
    spmv_tick += rocalution_time();
    
      mat.Apply(vec1, &vec2);    

    _rocalution_sync();
    spmv_tack += rocalution_time();

  }

    std::cout << "Dot execution: " << (dot_tack-dot_tick)/max_tests/1e3 << " msec" << "; "
	      << max_tests*double(sizeof(double)*(size+size))/(dot_tack-dot_tick)/1000 << " Gbyte/sec; "
	      << max_tests*double((2*size))/(dot_tack-dot_tick)/1000 << " GFlop/sec" << std::endl;

    std::cout << "Norm execution: " << (norm_tack-norm_tick)/max_tests/1e3 << " msec" << "; "
	      << max_tests*double(sizeof(double)*(size))/(norm_tack-norm_tick)/1000 << " Gbyte/sec; "
	      << max_tests*double((2*size))/(norm_tack-norm_tick)/1000 << " GFlop/sec" << std::endl;

    std::cout << "Vector update (scaleadd) execution: " << (updatev1_tack-updatev1_tick)/max_tests/1e3 << " msec" << "; "
	      << max_tests*double(sizeof(double)*(size+size+size))/(updatev1_tack-updatev1_tick)/1000 << " Gbyte/sec; "
	      << max_tests*double((2*size))/(updatev1_tack-updatev1_tick)/1000 << " GFlop/sec" << std::endl;

    std::cout << "Vector update (addscale) execution: " << (updatev2_tack-updatev2_tick)/max_tests/1e3 << " msec" << "; "
	      << max_tests*double(sizeof(double)*(size+size+size))/(updatev2_tack-updatev2_tick)/1000 << " Gbyte/sec; "
	      << max_tests*double((2*size))/(updatev2_tack-updatev2_tick)/1000 << " GFlop/sec" << std::endl;

    std::cout << "SpMV execution: " << (spmv_tack-spmv_tick)/max_tests/1e3 << " msec" << "; "
	      << max_tests*double((sizeof(double)*(size+size+nnz)+sizeof(int)*(size+nnz)))/(spmv_tack-spmv_tick)/1000 << " Gbyte/sec; "
	      << max_tests*double((2*nnz)/(spmv_tack-spmv_tick))/1000 << " GFlop/sec" << std::endl;


  stop_rocalution();

  return 0;
}
