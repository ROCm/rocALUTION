// Linear Solver
GMRES<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType > ls;

// Preconditioner
//  AS<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType > p;
RAS<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType > p; 

// Second level preconditioners
Solver<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType > **p2;

int n = 3;
int *size;
size = new int[n];

p2 = new Solver<LocalMatrix<ValueType>, 
                LocalVector<ValueType>, ValueType >*[n];

for (int i=0; i<n; ++i) {
  size[i] = mat.get_nrow() / n ;   

  MultiColoredILU<LocalMatrix<ValueType>, 
                  LocalVector<ValueType>, ValueType > *mc;
  mc = new MultiColoredILU<LocalMatrix<ValueType>, 
                           LocalVector<ValueType>, ValueType >;
  p2[i] = mc;

 }

 p.Set(n, 
       20,
       p2);
  
ls.SetOperator(mat);
ls.SetPreconditioner(p);

ls.Build();

ls.Solve(rhs, &x);
