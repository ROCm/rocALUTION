#ifndef PARALUTION_OCL_MATH_REAL_HPP_
#define PARALUTION_OCL_MATH_REAL_HPP_

namespace paralution {

const char *ocl_kernels_math_real = CL_KERNEL(

// Assigns scalar
inline static ValueType ocl_set(const ValueType val) { return val; }

// Absolute value
inline static ValueType ocl_abs(const ValueType val) { return fabs(val); }

// Norm
inline static ValueType ocl_norm(const ValueType val) { return val * val; }

// Replaces operator*
inline static ValueType ocl_mult(const ValueType lhs, const ValueType rhs) { return lhs * rhs; }
inline static ValueType ocl_multc(const ValueType lhs, const ValueType rhs) { return lhs * rhs; }

// Replaces operator/
inline static ValueType ocl_div(const ValueType lhs, const ValueType rhs) { return lhs / rhs; }

// Replaces power
inline static ValueType ocl_pow(const ValueType x, const ValueType y) { return pow(x, y); }

// Replaces operator ==
inline static bool ocl_equal(const ValueType lhs, const ValueType rhs) { return (lhs == rhs); }

// Replaces operator !=
inline static bool ocl_nequal(const ValueType lhs, const ValueType rhs) { return (lhs != rhs); }

// Atomic add
inline static void ocl_atomic_add(__global ValueType *address, ValueType val) {

  union {
    ValueType f;
    AtomicType i;
  } old, new;

  do {
    old.f = *address; new.f = old.f + val;
  } while (atom_cmpxchg((volatile __global AtomicType*) address, old.i, new.i) != old.i);

}

);

}

#endif // PARALUTION_OCL_MATH_REAL_HPP_
