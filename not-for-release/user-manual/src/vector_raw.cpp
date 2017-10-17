LocalVector<ValueType> vec;

ValueType *ptr_vec = NULL;

vec.Allocate("vector", 100);

vec.LeaveDataPtr(&ptr_vec);
