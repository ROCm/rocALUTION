out.write((char*) &nrow, sizeof(IndexType));
out.write((char*) &ncol, sizeof(IndexType));
out.write((char*) &nnz,  sizeof(IndexType));
out.write((char*) row_offset, (nrow+1)*sizeof(IndexType));
out.write((char*) col, nnz*sizeof(IndexType));
out.write((char*) val, nnz*sizeof(ValueType));
