GMRES<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType > ls;

BlockPreconditioner<LocalMatrix<ValueType>, LocalVector<ValueType>, 
                    ValueType > p;
Solver<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType > **p2;

int n = 2;
int *size;
size = new int[n];

p2 = new Solver<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType >*[n];


// Block 0
size[0] = mat.get_nrow() / n ;       
MultiColoredILU<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType > *mc;
mc = new MultiColoredILU<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType >;
p2[0] = mc;

// Block 1
size[1] = mat.get_nrow() / n ;       
AMG<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType > *amg;
amg = new AMG<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType >;
amg->InitMaxIter(2);
amg->Verbose(0);
p2[1] = amg;


p.Set(n, size, p2);
p.SetDiagonalSolver();

ls.SetOperator(mat);
ls.SetPreconditioner(p);

ls.Build();

ls.Solve(rhs, &x);
