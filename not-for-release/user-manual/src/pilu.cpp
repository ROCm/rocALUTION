ILU<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType > p;

p.Set(1);
ls.SetPreconditioner(p);
ls.Build();
