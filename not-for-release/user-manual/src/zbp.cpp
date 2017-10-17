LocalVector<int> zbp;
int size;

mat.ZeroBlockPermutation(size,
                         &zbp);
mat.Permute(zbp);

