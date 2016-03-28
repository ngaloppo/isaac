#include <cmath>
#include <iostream>
#include <type_traits>
#include "common.hpp"
#include "isaac/array.h"
#include "isaac/driver/common.h"
#include "clBLAS.h"
#include "cublas.h"

namespace sc = isaac;
typedef isaac::int_t int_t;

template<typename T>
void test_impl(std::string const & SLICE, T epsilon, simple_vector_base<T> & cy, simple_matrix_base<T> const & cA, simple_vector_base<T> & cx,
                                        sc::array_base & y, sc::array_base const & A, sc::array_base & x, int& nfail, int& npass)
{
  std::string TYPE = std::is_same<T, float>::value?"S":"D";

  sc::int_t M = A.shape()[0];
  sc::int_t N = A.shape()[1];
  simple_vector<T> bufy(M);
  simple_vector<T> bufx(N);
  T a = 4.2, b = 5.6;
  sc::driver::CommandQueue queue = sc::driver::backend::queues::get(y.context(),0);
  T yi = 0, xi = 0;
  if(y.context().backend()==sc::driver::OPENCL)
  {
      cl_command_queue clqueue = queue.handle().cl();
      ADD_TEST_2D_RD("GEMV-ROW-N", M, N, 0, yi+=cA(i,j)*cx[j], cy[i] = a*yi + b*cy[i],
               BLAS<T>::F(clblasSgemv, clblasDgemv)(clblasRowMajor, clblasTrans, N, M, a, cl(A), off(A), ld(A),  cl(x), off(x), inc(x), b, cl(y), off(y), inc(y), 1, &clqueue, 0, NULL, NULL), y, bufy, cy);
      ADD_TEST_2D_RD("GEMV-ROW-T", N, M, 0, xi+=cA(j,i)*cy[j], cx[i] = a*xi + b*cx[i],
              BLAS<T>::F(clblasSgemv, clblasDgemv)(clblasRowMajor, clblasNoTrans, N, M, a, cl(A), off(A), ld(A), cl(y), off(y), inc(y), b, cl(x), off(x), inc(x), 1, &clqueue, 0, NULL, NULL), x, bufx, cx);
      ADD_TEST_2D_RD("GEMV-COL-N", M, N, 0, yi+=cA(i,j)*cx[j], cy[i] = a*yi + b*cy[i],
              BLAS<T>::F(clblasSgemv, clblasDgemv)(clblasColumnMajor, clblasNoTrans, M, N, a, cl(A), off(A), ld(A), cl(x), off(x), inc(x), b, cl(y), off(y), inc(y), 1, &clqueue, 0, NULL, NULL), y, bufy, cy);
      ADD_TEST_2D_RD("GEMV-COL-T", N, M, 0, xi+=cA(j,i)*cy[j], cx[i] = a*xi + b*cx[i],
              BLAS<T>::F(clblasSgemv, clblasDgemv)(clblasColumnMajor, clblasTrans, M, N, a, cl(A), off(A), ld(A), cl(y), off(y), inc(y), b, cl(x), off(x), inc(x), 1, &clqueue, 0, NULL, NULL), x, bufx, cx);
  }
  if(y.context().backend()==sc::driver::CUDA)
  {
      ADD_TEST_2D_RD("GEMV-N", M, N, 0, yi+=cA(i,j)*cx[j], cy[i] = a*yi + b*cy[i],
              BLAS<T>::F(cublasSgemv, cublasDgemv)('N', M, N, a, (T*)CUHANDLE(A) + off(A), ld(A), (T*)CUHANDLE(x) + off(x), inc(x), b, (T*)CUHANDLE(y) + off(y), inc(y)), y, bufy, cy);
      ADD_TEST_2D_RD("GEMV-T", N, M, 0, xi+=cA(j,i)*cy[j], cx[i] = a*xi + b*cx[i],
              BLAS<T>::F(cublasSgemv, cublasDgemv)('T', M, N, a, (T*)CUHANDLE(A) + off(A), ld(A), (T*)CUHANDLE(y) + off(y), inc(y), b, (T*)CUHANDLE(x) + off(x), inc(x)), x, bufx, cx);
  }
}

template<typename T>
void test(sc::driver::Context const & ctx, int& nfail, int& npass)
{
  T epsilon = numeric_trait<T>::epsilon;
  int_t M = 173, N = 241;
  int_t SUBM = 7, SUBN = 11;

  INIT_VECTOR(M, SUBM, 7, 2, cy, y, ctx);
  INIT_VECTOR(N, SUBN, 5, 3, cx, x, ctx);
  INIT_MATRIX(M, SUBM, 9, 1, N, SUBN, 8, 1, cA, A, ctx);
  test_impl("FULL", epsilon, cy, cA, cx, y, A, x, nfail, npass);
  test_impl("SUB", epsilon, cy_s, cA_s, cx_s, y_s, A_s, x_s, nfail, npass);
}
int main()
{
  clblasSetup();
  int err = run_test(test<float>, test<double>);
  clblasTeardown();
  return err;
}
