#ifndef PARALUTION_HIP_COMPLEX_HPP_
#define PARALUTION_HIP_COMPLEX_HPP_

#include <cuComplex.h>

namespace paralution {

__device__ static __inline__ cuDoubleComplex operator+(const cuDoubleComplex &lhs, const cuDoubleComplex &rhs) { return cuCadd(lhs, rhs); }
__device__ static __inline__ cuFloatComplex operator+(const cuFloatComplex &lhs, const cuFloatComplex &rhs) { return cuCaddf(lhs, rhs); }

__device__ static __inline__ cuDoubleComplex operator*(const cuDoubleComplex &lhs, const cuDoubleComplex &rhs) { return cuCmul(lhs, rhs); }
__device__ static __inline__ cuFloatComplex operator*(const cuFloatComplex &lhs, const cuFloatComplex &rhs) { return cuCmulf(lhs, rhs); }

__device__ static __inline__ cuDoubleComplex operator/(const cuDoubleComplex &lhs, const cuDoubleComplex &rhs) { return cuCdiv(lhs, rhs); }
__device__ static __inline__ cuFloatComplex operator/(const cuFloatComplex &lhs, const cuFloatComplex &rhs) { return cuCdivf(lhs, rhs); }

__device__ static __inline__ double hip_abs(const double val) { return abs(val); }
__device__ static __inline__ float hip_abs(const float val) { return abs(val); }
__device__ static __inline__ double hip_abs(const cuDoubleComplex &val) { return cuCabs(val); }
__device__ static __inline__ float hip_abs(const cuFloatComplex &val) { return cuCabsf(val); }

__device__ static __inline__ void make_ValueType(float &val, const float &scalar) { val = (float) scalar; }
__device__ static __inline__ void make_ValueType(double &val, const double &scalar) { val = (double) scalar; }
__device__ static __inline__ void make_ValueType(cuFloatComplex &val, const float &scalar) { val = make_cuFloatComplex((float) scalar, (float) 0); }
__device__ static __inline__ void make_ValueType(cuDoubleComplex &val, const double &scalar) { val = make_cuDoubleComplex((double) scalar, (double) 0); }
__device__ static __inline__ void make_ValueType(int &val, const int &scalar) { val = (int) scalar; }

__device__ static __inline__ void assign_volatile_ValueType(const float *x, volatile float *y) { *y = *x; }
__device__ static __inline__ void assign_volatile_ValueType(const double *x, volatile double *y) { *y = *x; }
__device__ static __inline__ void assign_volatile_ValueType(const cuFloatComplex *x, volatile cuFloatComplex *y) { y->x = x->x; y->y = x->y; }
__device__ static __inline__ void assign_volatile_ValueType(const cuDoubleComplex *x, volatile cuDoubleComplex *y) { y->x = x->x; y->y = x->y; }

__device__ static __inline__ void assign_volatile_ValueType(const volatile float *x, float *y) { *y = *x; }
__device__ static __inline__ void assign_volatile_ValueType(const volatile double *x, double *y) { *y = *x; }
__device__ static __inline__ void assign_volatile_ValueType(const volatile cuFloatComplex *x, cuFloatComplex *y) { *y = make_cuFloatComplex(x->x, x->y); }
__device__ static __inline__ void assign_volatile_ValueType(const volatile cuDoubleComplex *x, cuDoubleComplex *y) { *y = make_cuDoubleComplex(x->x, x->y); }

__device__ static __inline__ float add_volatile_ValueType(const volatile float *x, float *y) { return *x + *y; }
__device__ static __inline__ double add_volatile_ValueType(const volatile double *x, double *y) { return *x + *y; }
__device__ static __inline__ cuFloatComplex add_volatile_ValueType(const volatile cuFloatComplex *x, cuFloatComplex *y) { cuFloatComplex volx; assign_volatile_ValueType(x, &volx);  return cuCaddf(volx, *y); }
__device__ static __inline__ cuDoubleComplex add_volatile_ValueType(const volatile cuDoubleComplex *x, cuDoubleComplex *y) { cuDoubleComplex volx; assign_volatile_ValueType(x, &volx); return cuCadd(volx, *y); }

__device__ static __inline__ void atomic_add_hip(float *address, float val) { atomicAdd(address, val); }
__device__ static __inline__ void atomic_add_hip(double *address, double val) { unsigned long long int *address_as_ull = (unsigned long long int*) address; unsigned long long int old = *address_as_ull, assumed; do { assumed = old; old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed))); }while (assumed != old); }
__device__ static __inline__ void atomic_add_hip(cuFloatComplex *address, cuFloatComplex val) { atomic_add_hip((float*) address, val.x); atomic_add_hip((float*) address+1, val.y); }
__device__ static __inline__ void atomic_add_hip(cuDoubleComplex *address, cuDoubleComplex val) { atomic_add_hip((double*) address, val.x); atomic_add_hip((double*) address+1, val.y); }

}

#endif // PARALUTION_HIP_COMPLEX_HPP_
