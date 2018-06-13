GMRES<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType > ls;

BlockPreconditioner<LocalMatrix<ValueType>, LocalVector<ValueType>, 
                    ValueType > p;
Solver<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType > **p2;

int n = 2;
int *size;
size = new int[n];

p2 = new Solver<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType >*[n];

  for (int i=0; i<n; ++i) {
    size[i] = mat.GetM() / n ;   
    
    MultiColoredILU<LocalMatrix<ValueType>, LocalVector<ValueType>, 
                    ValueType > *mc;
    mc = new MultiColoredILU<LocalMatrix<ValueType>, LocalVector<ValueType>, 
                             ValueType >;
    p2[i] = mc;
    
  }

p.Set(n, size, p2);

ls.SetOperator(mat);
ls.SetPreconditioner(p);

ls.Build();

ls.Solve(rhs, &x);
