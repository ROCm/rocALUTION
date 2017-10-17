FSAI<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType > p;

p.Set(2);
ls.SetPreconditioner(p);
ls.Build();
