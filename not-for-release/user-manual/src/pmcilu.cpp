MultiColoredILU<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType > p;

p.Set(1, 2);
ls.SetPreconditioner(p);
ls.Build();
