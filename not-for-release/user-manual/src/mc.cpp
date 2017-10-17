LocalVector<int> mc;
int num_colors;
int *block_colors = NULL;

mat.MultiColoring(num_colors,
                  &block_colors,
                  &mc);

mat.Permute(mc);
