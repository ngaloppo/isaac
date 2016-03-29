#include <cmath>
#include <iostream>
#include "common.hpp"
#include "isaac/array.h"
#include "isaac/driver/common.h"
#include "clBLAS.h"
#include "cublas.h"

namespace sc = isaac;
typedef isaac::int_t int_t;

template<typename T>
void test_impl(std::string const & SLICE, simple_vector_base<T> & cx, simple_vector_base<T>& cy, simple_vector_base<T>& cz,
                                                 sc::array_base& x, sc::array_base& y, sc::array_base& z, int& nfail, int& npass)
{
  T epsilon = numeric_trait<T>::epsilon;
  std::string TYPE = std::is_same<T, float>::value?"S":"D";
  sc::numeric_type dtype = x.dtype();
  sc::driver::Context const & context = x.context();
  sc::driver::CommandQueue queue = sc::driver::backend::queues::get(context,0);
  int_t N = cz.size();

  T aa = static_cast<T>(-4);
  T bb = static_cast<T>(3.5);
  isaac::value_scalar a(aa), b(bb);
  isaac::scalar da(a, context), db(b, context);


  ADD_TEST_1D_EW("z = 0", cz[i] = 0, z = sc::zeros({N}, dtype, context))
  ADD_TEST_1D_EW("z = x", cz[i] = cx[i], z = x)
  ADD_TEST_1D_EW("z = -x", cz[i] = -cx[i], z = -x)

  ADD_TEST_1D_EW("z = x + y", cz[i] = cx[i] + cy[i], z = x + y)
  ADD_TEST_1D_EW("z = x - y", cz[i] = cx[i] - cy[i], z = x - y)
  ADD_TEST_1D_EW("z = x + y + z", cz[i] = cx[i] + cy[i] + cz[i], z = x + y + z)

  ADD_TEST_1D_EW("z = a*x", cz[i] = aa*cx[i], z = a*x)
  ADD_TEST_1D_EW("z = da*x", cz[i] = aa*cx[i], z = da*x)
  ADD_TEST_1D_EW("z = a*x + b*y", cz[i] = aa*cx[i] + bb*cy[i], z= a*x + b*y)
  ADD_TEST_1D_EW("z = da*x + b*y", cz[i] = aa*cx[i] + bb*cy[i], z= da*x + b*y)
  ADD_TEST_1D_EW("z = a*x + db*y", cz[i] = aa*cx[i] + bb*cy[i], z= a*x + db*y)
  ADD_TEST_1D_EW("z = da*x + db*y", cz[i] = aa*cx[i] + bb*cy[i], z= da*x + db*y)

  ADD_TEST_1D_EW("z = exp(x)", cz[i] = exp(cx[i]), z= exp(x))
  ADD_TEST_1D_EW("z = abs(x)", cz[i] = abs(cx[i]), z= abs(x))
  ADD_TEST_1D_EW("z = acos(x)", cz[i] = acos(cx[i]), z= acos(x))
  ADD_TEST_1D_EW("z = asin(x)", cz[i] = asin(cx[i]), z= asin(x))
  ADD_TEST_1D_EW("z = atan(x)", cz[i] = atan(cx[i]), z= atan(x))
  ADD_TEST_1D_EW("z = ceil(x)", cz[i] = ceil(cx[i]), z= ceil(x))
  ADD_TEST_1D_EW("z = cos(x)", cz[i] = cos(cx[i]), z= cos(x))
  ADD_TEST_1D_EW("z = cosh(x)", cz[i] = cosh(cx[i]), z= cosh(x))
  ADD_TEST_1D_EW("z = floor(x)", cz[i] = floor(cx[i]), z= floor(x))
  ADD_TEST_1D_EW("z = log(x)", cz[i] = log(cx[i]), z= log(x))
  ADD_TEST_1D_EW("z = log10(x)", cz[i] = log10(cx[i]), z= log10(x))
  ADD_TEST_1D_EW("z = sin(x)", cz[i] = sin(cx[i]), z= sin(x))
  ADD_TEST_1D_EW("z = sinh(x)", cz[i] = sinh(cx[i]), z= sinh(x))
  ADD_TEST_1D_EW("z = sqrt(x)", cz[i] = sqrt(cx[i]), z= sqrt(x))
  ADD_TEST_1D_EW("z = tan(x)", cz[i] = tan(cx[i]), z= tan(x))
  ADD_TEST_1D_EW("z = tanh(x)", cz[i] = tanh(cx[i]), z= tanh(x))

  ADD_TEST_1D_EW("z = x.*y", cz[i] = cx[i]*cy[i], z= x*y)
  ADD_TEST_1D_EW("z = x./y", cz[i] = cx[i]/cy[i], z= x/y)

  ADD_TEST_1D_EW("z = pow(x,y)", cz[i] = pow(cx[i], cy[i]), z= pow(x,y))

  ADD_TEST_1D_EW("z = x==y", cz[i] = cx[i]==cy[i], z= cast(x==y, dtype))
  ADD_TEST_1D_EW("z = x>=y", cz[i] = cx[i]>=cy[i], z= cast(x>=y, dtype))
  ADD_TEST_1D_EW("z = x>y", cz[i] = cx[i]>cy[i], z= cast(x>y, dtype))
  ADD_TEST_1D_EW("z = x<=y", cz[i] = cx[i]<=cy[i], z= cast(x<=y, dtype))
  ADD_TEST_1D_EW("z = x<y", cz[i] = cx[i]<cy[i], z= cast(x<y, dtype))
  ADD_TEST_1D_EW("z = x!=y", cz[i] = cx[i]!=cy[i], z= cast(x!=y, dtype))
}

template<typename T>
void test(sc::driver::Context const & ctx, int& nfail, int& npass)
{
  int_t N = 10007;
  int_t SUBN = 7;

  INIT_VECTOR(N, SUBN, 0, 1, cx, x, ctx);
  INIT_VECTOR(N, SUBN, 0, 1, cy, y, ctx);
  INIT_VECTOR(N, SUBN, 0, 1, cz, z, ctx);

  test_impl("FULL", cx, cy, cz, x, y, z, nfail, npass);
  test_impl("SUB", cx_s, cy_s, cz_s, x_s, y_s, z_s, nfail, npass);
}

int main()
{
  return run_test(test<float>, test<double>);
}
