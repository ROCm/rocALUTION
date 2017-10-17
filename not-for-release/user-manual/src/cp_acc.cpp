LocalVector<ValueType> vec1, vec2;

// vec1 is on the host
vec1.MoveToHost();

// vec2 is on the accelrator
vec2.MoveToAcclerator();

// copy vec2 to vec1
vec1.CopyFrom(vec2);

// copy vec1 to vec2
vec2.CopyFrom(vec1);
