  LocalMatrix<double> mat;

  mat.ReadFileMTX(std::string(argv[1]));

  long int row_key;
  long int col_key;
  long int val_key;
  
  mat.Key(row_key,
          col_key,
          val_key);

  std::cout << "Row key = " << row_key << std::endl
            << "Col key = " << col_key << std::endl
            << "Val key = " << val_key << std::endl;
