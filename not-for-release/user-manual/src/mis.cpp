LocalVector<int> mis;
int size;

mat.MaximalIndependentSet(size,
                         &mis);
mat.Permute(mis);

