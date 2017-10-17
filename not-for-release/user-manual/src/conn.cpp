LocalVector<int> conn;

mat.ConnectivityOrder(&conn);
mat.Permute(conn);
