#include <cmath>
#include "common.hpp"
#include "isaac/array.h"
#include "clBLAS.h"
#include "cublas.h"

namespace sc = isaac;

template<typename T>
void test_impl(std::string const & SLICE, simple_matrix_base<T> & cC, simple_matrix_base<T> const & cA, simple_matrix_base<T> const & cB, sc::array_base & C,
          sc::array_base const & A, sc::array_base const & AT,  sc::array_base const & B, sc::array_base const & BT, int& nfail, int& npass)
{
  T epsilon = numeric_trait<T>::epsilon;
  sc::int_t M = C.shape()[0], N = C.shape()[1], K = A.shape()[1];
  std::string TYPE = std::is_same<T, float>::value?"S":"D";
  T alpha = 1.43;
  T beta = 0;
  for(int i = 0 ; i < M ; ++i){
    for(int j = 0 ; j < N ; ++j){
      T cij = 0;
      for(int k = 0 ; k < K ; ++k)
        cij += cA(i,k)*cB(k,j);
      cC(i,j) = alpha*cij + beta*cC(i, j);
    }
  }
  std::vector<T> cCbuffer(M*N);
  for(int i = 0 ; i < M ; ++i)
    for(int j = 0 ; j < N ; ++j)
      cCbuffer[i + j*M] = cC(i,j);
  std::vector<T> buffer(M*N);

  sc::driver::CommandQueue queue = sc::driver::backend::queues::get(C.context(),0);
  ADD_TEST_MATMUL("C = A * B", C = alpha*dot(A,B) + beta*C)
  ADD_TEST_MATMUL("C = A' * B", C = alpha*dot(AT.T,B) + beta*C)
  ADD_TEST_MATMUL("C = A * B'", C = alpha*dot(A,BT.T) + beta*C)
  ADD_TEST_MATMUL("C = A' * B'", C = alpha*dot(AT.T,BT.T) + beta*C)

}

template<typename T>
void test(sc::driver::Context const & ctx, int& nfail, int& npass)
{
    sc::int_t M = 173, N = 241, K = 293;
    sc::int_t SUBM = 7, SUBN = 11, SUBK = 29;

    INIT_MATRIX(M, SUBM, 5, 2, N, SUBN, 7, 3, cC, C, ctx);
    INIT_MATRIX(M, SUBM, 8, 2, K, SUBK, 4, 3, cA, A, ctx);
    INIT_MATRIX(K, SUBK, 9, 4, N, SUBN, 6, 2, cB, B, ctx);
    test_impl("FULL", cC, cA, cB, C, A, AT, B, BT, nfail, npass);
    test_impl("SUB", cC_s, cA_s, cB_s, C_s, A_s, AT_s, B_s, BT_s, nfail, npass);
}

int main()
{
  int err = run_test(test<float>, test<double>);
  return err;
}
