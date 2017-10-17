#ifndef PARALUTION_OCL_MATH_COMPLEX_HPP_
#define PARALUTION_OCL_MATH_COMPLEX_HPP_

namespace paralution {

const char *ocl_kernels_math_complex = CL_KERNEL(

// Assigns scalar
inline static ValueType ocl_set(const ValueType val) { ValueType res; res.x = val.x; res.y = (RealType) 0; return res; }

// Absolute value
inline static RealType ocl_abs(const ValueType val) { return sqrt(val.x*val.x + val.y*val.y); }

// Norm
inline static ValueType ocl_norm(const ValueType val) { ValueType res; res.x = val.x*val.x+val.y*val.y; res.y = 0.0; return res; }

// Replaces operator*
inline static ValueType ocl_mult(const ValueType lhs, const ValueType rhs) { ValueType res; res.x = lhs.x*rhs.x - lhs.y*rhs.y; res.y = lhs.x*rhs.y+lhs.y*rhs.x; return res; }
inline static ValueType ocl_multc(const ValueType lhs, const ValueType rhs) { ValueType res; res.x = lhs.x*rhs.x + lhs.y*rhs.y; res.y = lhs.x*rhs.y - lhs.y*rhs.x; return res; }

// Replaces operator/
inline static ValueType ocl_div(const ValueType lhs, const ValueType rhs) { ValueType res; RealType div = rhs.x*rhs.x+rhs.y*rhs.y; res.x = (lhs.x*rhs.x+lhs.y*rhs.y)/div; res.y = (lhs.y*rhs.x-lhs.x*rhs.y)/div; return res; }

// Complex power is currently unsupported
inline static ValueType ocl_pow(const ValueType x, const ValueType y) { return pow(x, y); }

// Replaces operator==
inline static bool ocl_equal(const ValueType lhs, const ValueType rhs) { bool cmp = false; if (lhs.x == rhs.x && lhs.y == rhs.y) cmp = true; return cmp; }

// Replaces operator!=
inline static bool ocl_nequal(const ValueType lhs, const ValueType rhs) { bool cmp = true; if (lhs.x == rhs.x && lhs.y == rhs.y) cmp = false; return cmp; }

// Complex atomic_add is currently unsupported
inline static void ocl_atomic_add(__global ValueType *address, ValueType val) { return; }

);

}

#endif // PARALUTION_OCL_MATH_COMPLEX_HPP_
