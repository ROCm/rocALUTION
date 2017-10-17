MultiColoredGS<LocalMatrix<ValueType>, LocalVector<ValueType>, ValueType > p;

p.SetRelaxation(1.6);

ls.SetPreconditioner(p);
ls.Build();
