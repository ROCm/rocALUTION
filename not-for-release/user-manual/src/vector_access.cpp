LocalVector<ValueType> vec;

vec.Allocate("vector", 100);

vec.Ones();

for (int i=0; i<100; i=i+2)
  vec[i] = -1;
