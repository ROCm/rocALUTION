LocalVector<ValueType> vec1, vec2;

// Allocate and init vec1 and vec2
// ...

vec1.MoveToAccelerator();

// now vec1 is on the accelerator (if any)
// and vec2 is on the host

// we can copy vec1 to vec2 and vice versa

vec1.CopyFrom(vec2);

