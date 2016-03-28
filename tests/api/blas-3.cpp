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
  if(C.context().backend()==sc::driver::OPENCL)
  {
      cl_command_queue clqueue = queue.handle().cl();
     //Row-major
      ADD_TEST_MATMUL("GEMM(ROW, N, N)", BLAS<T>::F(clblasSgemm,clblasDgemm)(clblasRowMajor, clblasNoTrans, clblasNoTrans, N, M, K, alpha, CHANDLE(B), OFF(B), LD(B),
                                                                 CHANDLE(A), OFF(A), LD(A), beta, CHANDLE(C), OFF(C), LD(C), 1, &clqueue, 0, NULL, NULL));
      ADD_TEST_MATMUL("GEMM(ROW, N, T)", BLAS<T>::F(clblasSgemm,clblasDgemm)(clblasRowMajor, clblasTrans, clblasNoTrans, N, M, K, alpha, CHANDLE(BT), OFF(BT), LD(BT),
                                                                 CHANDLE(A), OFF(A), LD(A), beta, CHANDLE(C), OFF(C), LD(C), 1, &clqueue, 0, NULL, NULL));
      ADD_TEST_MATMUL("GEMM(ROW, T, N)", BLAS<T>::F(clblasSgemm,clblasDgemm)(clblasRowMajor, clblasNoTrans, clblasTrans, N, M, K, alpha, CHANDLE(B), OFF(B), LD(B),
                                                                 CHANDLE(AT), OFF(AT), LD(AT), beta, CHANDLE(C), OFF(C), LD(C), 1, &clqueue, 0, NULL, NULL));
      ADD_TEST_MATMUL("GEMM(ROW, T, T)", BLAS<T>::F(clblasSgemm,clblasDgemm)(clblasRowMajor, clblasTrans, clblasTrans, N, M, K, alpha, CHANDLE(BT), OFF(BT), LD(BT),
                                                                 CHANDLE(AT), OFF(AT), LD(AT), beta, CHANDLE(C), OFF(C), LD(C), 1, &clqueue, 0, NULL, NULL));
      //Column-major
      ADD_TEST_MATMUL("GEMM(COL, N, N)", BLAS<T>::F(clblasSgemm,clblasDgemm)(clblasColumnMajor, clblasNoTrans, clblasNoTrans, M, N, K, alpha, CHANDLE(A), OFF(A), LD(A),
                                                                 CHANDLE(B), OFF(B), LD(B), beta, CHANDLE(C), OFF(C), LD(C), 1, &clqueue, 0, NULL, NULL));
      ADD_TEST_MATMUL("GEMM(COL, N, T)", BLAS<T>::F(clblasSgemm,clblasDgemm)(clblasColumnMajor, clblasNoTrans, clblasTrans, M, N, K, alpha, CHANDLE(A), OFF(A), LD(A),
                                                                 CHANDLE(BT), OFF(BT), LD(BT), beta, CHANDLE(C), OFF(C), LD(C), 1, &clqueue, 0, NULL, NULL));
      ADD_TEST_MATMUL("GEMM(COL, T, N)", BLAS<T>::F(clblasSgemm,clblasDgemm)(clblasColumnMajor, clblasTrans, clblasNoTrans, M, N, K, alpha, CHANDLE(AT), OFF(AT), LD(AT),
                                                                 CHANDLE(B), OFF(B), LD(B), beta, CHANDLE(C), OFF(C), LD(C), 1, &clqueue, 0, NULL, NULL));
      ADD_TEST_MATMUL("GEMM(COL, T, T)", BLAS<T>::F(clblasSgemm,clblasDgemm)(clblasColumnMajor, clblasTrans, clblasTrans, M, N, K, alpha, CHANDLE(AT), OFF(AT), LD(AT),
                                                                 CHANDLE(BT), OFF(BT), LD(BT), beta, CHANDLE(C), OFF(C), LD(C), 1, &clqueue, 0, NULL, NULL));

  }

  if(C.context().backend()==sc::driver::CUDA)
  {
      ADD_TEST_MATMUL("GEMM-NN", BLAS<T>::F(cublasSgemm,cublasDgemm)('N', 'N', M, N, K, alpha, (T*)CUHANDLE(A) + OFF(A), LD(A),
                                                                 (T*)CUHANDLE(B) + OFF(B), LD(B), beta, (T*)CUHANDLE(C) + OFF(C), LD(C)));
      ADD_TEST_MATMUL("GEMM-NT", BLAS<T>::F(cublasSgemm,cublasDgemm)('N', 'T', M, N, K, alpha, (T*)CUHANDLE(A) + OFF(A), LD(A),
                                                                 (T*)CUHANDLE(BT) + OFF(BT), LD(BT), beta, (T*)CUHANDLE(C) + OFF(C), LD(C)));
      ADD_TEST_MATMUL("GEMM-TN", BLAS<T>::F(cublasSgemm,cublasDgemm)('T', 'N', M, N, K, alpha, (T*)CUHANDLE(AT) + OFF(AT), LD(AT),
                                                                 (T*)CUHANDLE(B) + OFF(B), LD(B), beta, (T*)CUHANDLE(C) + OFF(C), LD(C)));
      ADD_TEST_MATMUL("GEMM-TT", BLAS<T>::F(cublasSgemm,cublasDgemm)('T', 'T', M, N, K, alpha, (T*)CUHANDLE(AT) + OFF(AT), LD(AT),
                                                                 (T*)CUHANDLE(BT) + OFF(BT), LD(BT), beta, (T*)CUHANDLE(C) + OFF(C), LD(C)));
  }
}

template<typename T>
void test(sc::driver::Context const & ctx, int& nfail, int& npass)
{
    sc::int_t M = 173, N = 241, K = 293;
    sc::int_t SUBM = 7, SUBN = 11, SUBK = 29;

    INIT_MATRIX(M, SUBM, 5, 1, N, SUBN, 7, 1, cC, C, ctx);
    INIT_MATRIX(M, SUBM, 8, 1, K, SUBK, 4, 1, cA, A, ctx);
    INIT_MATRIX(K, SUBK, 9, 1, N, SUBN, 6, 1, cB, B, ctx);
    test_impl("FULL", cC, cA, cB, C, A, AT, B, BT, nfail, npass);
    test_impl("SUB", cC_s, cA_s, cB_s, C_s, A_s, AT_s, B_s, BT_s, nfail, npass);
}

int main()
{
  clblasSetup();
  int err = run_test(test<float>, test<double>);
  clblasTeardown();
  return err;
}
