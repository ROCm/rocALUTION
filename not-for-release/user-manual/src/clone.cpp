LocalVector<ValueType> vec;

// allocate and init vec (host or accelerator)
// ...

LocalVector<ValueType> tmp;

// tmp will have the same values 
// and it will be on the same backend as vec
tmp.CloneFrom(vec);


