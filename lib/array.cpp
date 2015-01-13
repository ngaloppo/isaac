#include <cassert>

#include "atidlas/array.h"
#include "atidlas/cl/cl.hpp"
#include "atidlas/model/model.h"
#include "atidlas/symbolic/execute.h"

namespace atidlas
{


/*--- Constructors ---*/

//1D Constructors

array::array(int_t size1, numeric_type dtype, cl::Context context) :
  dtype_(dtype), shape_(size1, 1), start_(0, 0), stride_(1, 1), ld_(shape_._1),
  context_(context), data_(context_, CL_MEM_READ_WRITE, size_of(dtype)*dsize())
{ }

template<class T>
array::array(std::vector<T> const & x, cl::Context context):
  shape_(x.size(), 1), dtype_(to_numeric_type<T>::value), start_(0, 0), stride_(1, 1), ld_(shape_._1),
  context_(context), data_(context, CL_MEM_READ_WRITE, size_of(dtype_)*dsize())
{ *this = x; }

array::array(array & v, slice const & s1) : dtype_(v.dtype_), shape_(s1.size, 1), start_(v.start_._1 + v.stride_._1*s1.start, 0), stride_(v.stride_._1*s1.stride, 1),
                                            ld_(v.ld_), context_(v.data_.getInfo<CL_MEM_CONTEXT>()), data_(v.data_)
{
}

#define INSTANTIATE(T) template array::array<T>(std::vector<T> const &, cl::Context)
INSTANTIATE(cl_char);
INSTANTIATE(cl_uchar);
INSTANTIATE(cl_short);
INSTANTIATE(cl_ushort);
INSTANTIATE(cl_int);
INSTANTIATE(cl_uint);
INSTANTIATE(cl_long);
INSTANTIATE(cl_ulong);
INSTANTIATE(cl_float);
INSTANTIATE(cl_double);
#undef INSTANTIATE

// 2D
array::array(int_t size1, int_t size2, numeric_type dtype, cl::Context context) : dtype_(dtype), shape_(size1, size2), start_(0, 0), stride_(1, 1), ld_(size1),
                                                                              context_(context), data_(context_, CL_MEM_READ_WRITE, size_of(dtype_)*dsize())
{}

array::array(array & M, slice const & s1, slice const & s2) :  dtype_(M.dtype_), shape_(s1.size, s2.size),
                                                          start_(M.start_._1 + M.stride_._1*s1.start, M.start_._2 + M.stride_._2*s2.start),
                                                          stride_(M.stride_._1*s1.stride, M.stride_._2*s2.stride), ld_(M.ld_),
                                                          context_(M.data_.getInfo<CL_MEM_CONTEXT>()), data_(M.data_)
{ }

template<typename T>
array::array(int_t size1, int_t size2, std::vector<T> const & data, cl::Context context)
  : dtype_(to_numeric_type<T>::value),
    shape_(size1, size2), start_(0, 0), stride_(1, 1), ld_(size1),
    context_(context), data_(context_, CL_MEM_READ_WRITE, size_of(dtype_)*dsize())
{
  atidlas::copy(data, *this);
}

#define INSTANTIATE(T) template array::array<T>(int_t, int_t, std::vector<T> const &, cl::Context)
INSTANTIATE(cl_char);
INSTANTIATE(cl_uchar);
INSTANTIATE(cl_short);
INSTANTIATE(cl_ushort);
INSTANTIATE(cl_int);
INSTANTIATE(cl_uint);
INSTANTIATE(cl_long);
INSTANTIATE(cl_ulong);
INSTANTIATE(cl_float);
INSTANTIATE(cl_double);
#undef INSTANTIATE

// General
array::array(numeric_type dtype, cl::Buffer data, slice const & s1, slice const & s2, cl::Context context):
  dtype_(dtype), shape_(s1.size, s2.size), start_(s1.start, s2.start), stride_(s1.stride, s2.stride),
  context_(context), ld_(shape_._1), data_(data)
{ }

array::array(array_expression const & proxy) :
  dtype_(proxy.dtype()),
  shape_(proxy.shape()), start_(0,0), stride_(1, 1), context_(proxy.context()),
  ld_(shape_._1), data_(context_, CL_MEM_READ_WRITE, size_of(dtype_)*dsize())
{
  *this = proxy;
}

/*--- Getters ---*/
numeric_type array::dtype() const
{ return dtype_; }

size4 array::shape() const
{ return shape_; }

int_t array::nshape() const
{ return int_t((shape_._1 > 1) + (shape_._2 > 1)); }

size4 array::start() const
{ return start_; }

size4 array::stride() const
{ return stride_; }

int_t array::ld() const
{ return ld_; }

cl::Context const & array::context() const
{ return context_; }

cl::Buffer const & array::data() const
{ return data_; }

int_t array::dsize() const
{ return ld_*shape_._2; }

/*--- Setters ---*/
array& array::reshape(int_t size1, int_t size2)
{
  assert(size1*size2==prod(shape_));
  shape_ = size4(size1, size2);
  return *this;
}

/*--- Assignment Operators ----*/
//---------------------------------------
array & array::operator=(array const & rhs)
{
  array_expression expression(*this, rhs, op_element(OPERATOR_BINARY_TYPE_FAMILY, OPERATOR_ASSIGN_TYPE), context_, dtype_, shape_);
  cl::CommandQueue & queue = cl::queues[context_].front();
  model_map_t & mmap = atidlas::get_model_map(queue);
  execute(expression, mmap);
  return *this;
}

array & array::operator=(array_expression const & rhs)
{
    array_expression expression(*this, rhs, op_element(OPERATOR_BINARY_TYPE_FAMILY, OPERATOR_ASSIGN_TYPE), shape_);
    cl::CommandQueue & queue = cl::queues[context_].front();
    model_map_t & mmap = atidlas::get_model_map(queue);
    execute(expression, mmap);
    return *this;
}

template<class T>
array & array::operator=(std::vector<T> const & rhs)
{
  assert(nshape()==1);
  atidlas::copy(rhs, *this);
  return *this;
}

#define INSTANTIATE(T) template array & array::operator=<T>(std::vector<T> const &)

INSTANTIATE(cl_char);
INSTANTIATE(cl_uchar);
INSTANTIATE(cl_short);
INSTANTIATE(cl_ushort);
INSTANTIATE(cl_int);
INSTANTIATE(cl_uint);
INSTANTIATE(cl_long);
INSTANTIATE(cl_ulong);
INSTANTIATE(cl_float);
INSTANTIATE(cl_double);
#undef INSTANTIATE
//
array & array::operator*=(value_scalar const & rhs)
{ return *this = array_expression(*this, rhs, op_element(OPERATOR_BINARY_TYPE_FAMILY, OPERATOR_MULT_TYPE), context_, dtype_, shape_); }

array & array::operator*=(array const & rhs)
{ return *this = array_expression(*this, rhs, op_element(OPERATOR_BINARY_TYPE_FAMILY, OPERATOR_MULT_TYPE), context_, dtype_, shape_); }

array & array::operator*=(array_expression const & rhs)
{ return *this = array_expression(*this, rhs, op_element(OPERATOR_BINARY_TYPE_FAMILY, OPERATOR_MULT_TYPE), shape_); }

array & array::operator/=(value_scalar const & rhs)
{ return *this = array_expression(*this, rhs, op_element(OPERATOR_BINARY_TYPE_FAMILY, OPERATOR_DIV_TYPE), context_, dtype_, shape_); }

array & array::operator/=(array const & rhs)
{ return *this = array_expression(*this, rhs, op_element(OPERATOR_BINARY_TYPE_FAMILY, OPERATOR_DIV_TYPE), context_, dtype_, shape_); }

array & array::operator/=(array_expression const & rhs)
{ return *this = array_expression(*this, rhs, op_element(OPERATOR_BINARY_TYPE_FAMILY, OPERATOR_DIV_TYPE), shape_); }

/*--- Indexing operators -----*/
//---------------------------------------
scalar array::operator [](int_t idx)
{
  assert(nshape()==1);
  return scalar(dtype_, data_, idx, context_);
}

array array::operator[](slice const & e1)
{
  assert(nshape()==1);
  return array(*this, e1);
}

array array::operator()(slice const & e1, slice const & e2)
{ return array(*this, e1, e2); }
//---------------------------------------

/*--- Scalar ---*/
namespace detail
{

template<class T>
void copy(cl::Context & ctx, cl::Buffer const & data, T value)
{
  cl::queues[ctx].front().enqueueWriteBuffer(data, CL_TRUE, 0, sizeof(T), (void*)&value);
}

}

scalar::scalar(numeric_type dtype, const cl::Buffer &data, int_t offset, cl::Context context): array(dtype, data, _(offset, offset+1), _(1,1), context)
{ }

scalar::scalar(value_scalar value, cl::Context context) : array(1, value.dtype(), context)
{
  switch(dtype_)
  {
    case CHAR_TYPE: detail::copy(context_, data_, value.value<cl_char>()); break;
    case UCHAR_TYPE: detail::copy(context_, data_, value.value<cl_uchar>()); break;
    case SHORT_TYPE: detail::copy(context_, data_, value.value<cl_short>()); break;
    case USHORT_TYPE: detail::copy(context_, data_, value.value<cl_ushort>()); break;
    case INT_TYPE: detail::copy(context_, data_, value.value<cl_int>()); break;
    case UINT_TYPE: detail::copy(context_, data_, value.value<cl_uint>()); break;
    case LONG_TYPE: detail::copy(context_, data_, value.value<cl_long>()); break;
    case ULONG_TYPE: detail::copy(context_, data_, value.value<cl_ulong>()); break;
    case FLOAT_TYPE: detail::copy(context_, data_, value.value<cl_float>()); break;
    case DOUBLE_TYPE: detail::copy(context_, data_, value.value<cl_double>()); break;
    default: throw "unrecognized datatype";
  }
}


scalar::scalar(numeric_type dtype, cl::Context context) : array(1, dtype, context){ }

scalar::scalar(array_expression const & proxy) : array(proxy){ }

template<class T>
T scalar::cast() const
{
  values_holder v;

#define HANDLE_CASE(DTYPE, VAL) \
case DTYPE:\
  cl::queues[context_].front().enqueueReadBuffer(data_, CL_TRUE, start_._1, size_of(dtype_), (void*)&v.VAL);\
  return v.VAL

  switch(dtype_)
  {
    HANDLE_CASE(CHAR_TYPE, int8);
    HANDLE_CASE(UCHAR_TYPE, uint8);
    HANDLE_CASE(SHORT_TYPE, int16);
    HANDLE_CASE(USHORT_TYPE, uint16);
    HANDLE_CASE(INT_TYPE, int32);
    HANDLE_CASE(UINT_TYPE, uint32);
    HANDLE_CASE(LONG_TYPE, int64);
    HANDLE_CASE(ULONG_TYPE, uint64);
    HANDLE_CASE(FLOAT_TYPE, float32);
    HANDLE_CASE(DOUBLE_TYPE, float64);
    default: throw "Datatype not recognized";
  }
#undef HANDLE_CASE

}

scalar& scalar::operator=(value_scalar const & s)
{
  cl::CommandQueue& queue = cl::queues[context_].front();

#define HANDLE_CASE(TYPE, CLTYPE) case TYPE:\
                            {\
                            CLTYPE v = s.value<CLTYPE>();\
                            queue.enqueueWriteBuffer(data_, CL_TRUE, start_._1, size_of(dtype_), (void*)&v);\
                            return *this;\
                            }
  switch(dtype_)
  {
    HANDLE_CASE(CHAR_TYPE, cl_char)
    HANDLE_CASE(UCHAR_TYPE, cl_uchar)
    HANDLE_CASE(SHORT_TYPE, cl_short)
    HANDLE_CASE(USHORT_TYPE, cl_ushort)
    HANDLE_CASE(INT_TYPE, cl_int)
    HANDLE_CASE(UINT_TYPE, cl_uint)
    HANDLE_CASE(LONG_TYPE, cl_long)
    HANDLE_CASE(ULONG_TYPE, cl_ulong)
    HANDLE_CASE(FLOAT_TYPE, cl_float)
    HANDLE_CASE(DOUBLE_TYPE, cl_double)
    default: throw "Datatype not recognized";
  }
}

scalar& scalar::operator=(scalar const & s)
{ array::operator =(s); }

#define INSTANTIATE(type) scalar::operator type() const { return cast<type>(); }
  INSTANTIATE(cl_char)
  INSTANTIATE(cl_uchar)
  INSTANTIATE(cl_short)
  INSTANTIATE(cl_ushort)
  INSTANTIATE(cl_int)
  INSTANTIATE(cl_uint)
  INSTANTIATE(cl_long)
  INSTANTIATE(cl_ulong)
  INSTANTIATE(cl_float)
  INSTANTIATE(cl_double)
#undef INSTANTIATE

inline std::ostream & operator<<(std::ostream & os, scalar const & s)
{
  switch(s.dtype())
  {
    case CHAR_TYPE: return os << static_cast<cl_char>(s);
    case UCHAR_TYPE: return os << static_cast<cl_uchar>(s);
    case SHORT_TYPE: return os << static_cast<cl_short>(s);
    case USHORT_TYPE: return os << static_cast<cl_ushort>(s);
    case INT_TYPE: return os << static_cast<cl_int>(s);
    case UINT_TYPE: return os << static_cast<cl_uint>(s);
    case LONG_TYPE: return os << static_cast<cl_long>(s);
    case ULONG_TYPE: return os << static_cast<cl_ulong>(s);
    case HALF_TYPE: return os << static_cast<cl_half>(s);
    case FLOAT_TYPE: return os << static_cast<cl_float>(s);
    case DOUBLE_TYPE: return os << static_cast<cl_double>(s);
    default: throw "";
  }
}

/*--- Binary Operators ----*/
//-----------------------------------
template<class U, class V>
bool check_elementwise(U const & u, V const & v)
{
  return max(u.shape())==1 || max(v.shape())==1 || u.shape()==v.shape();
}

#define DEFINE_ELEMENT_BINARY_OPERATOR(OP, OPNAME) \
array_expression OPNAME (array_expression const & x, array_expression const & y) \
{ assert(check_elementwise(x, y));\
  return array_expression(x, y, op_element(OPERATOR_BINARY_TYPE_FAMILY, OP), std::max(max(x.shape()), max(y.shape())) ); } \
 \
array_expression OPNAME (array const & x, array_expression const & y) \
{ assert(check_elementwise(x, y));\
  return array_expression(x, y, op_element(OPERATOR_BINARY_TYPE_FAMILY, OP), std::max(max(x.shape()), max(y.shape()))); } \
\
array_expression OPNAME (array_expression const & x, array const & y) \
{ assert(check_elementwise(x, y));\
  return array_expression(x, y, op_element(OPERATOR_BINARY_TYPE_FAMILY, OP), std::max(max(x.shape()), max(y.shape()))); } \
\
array_expression OPNAME (array const & x, array const & y) \
{ assert(check_elementwise(x, y));\
  return array_expression(x, y, op_element(OPERATOR_BINARY_TYPE_FAMILY, OP), x.context(), x.dtype(), std::max(max(x.shape()), max(y.shape()))); }\
\
array_expression OPNAME (array_expression const & x, value_scalar const & y) \
{ return array_expression(x, y, op_element(OPERATOR_BINARY_TYPE_FAMILY, OP), x.shape()); } \
\
array_expression OPNAME (array const & x, value_scalar const & y) \
{ return array_expression(x, y, op_element(OPERATOR_BINARY_TYPE_FAMILY, OP), x.context(), x.dtype(), x.shape()); }\
\
array_expression OPNAME (value_scalar const & y, array_expression const & x) \
{ return array_expression(y, x, op_element(OPERATOR_BINARY_TYPE_FAMILY, OP), x.shape()); } \
\
array_expression OPNAME (value_scalar const & y, array const & x) \
{ return array_expression(y, x, op_element(OPERATOR_BINARY_TYPE_FAMILY, OP), x.context(), x.dtype(), x.shape()); }

DEFINE_ELEMENT_BINARY_OPERATOR(OPERATOR_ADD_TYPE, operator +)
DEFINE_ELEMENT_BINARY_OPERATOR(OPERATOR_SUB_TYPE, operator -)
DEFINE_ELEMENT_BINARY_OPERATOR(OPERATOR_MULT_TYPE, operator *)
DEFINE_ELEMENT_BINARY_OPERATOR(OPERATOR_DIV_TYPE, operator /)

DEFINE_ELEMENT_BINARY_OPERATOR(OPERATOR_ELEMENT_GREATER_TYPE, operator >)
DEFINE_ELEMENT_BINARY_OPERATOR(OPERATOR_ELEMENT_GEQ_TYPE, operator >=)
DEFINE_ELEMENT_BINARY_OPERATOR(OPERATOR_ELEMENT_LESS_TYPE, operator <)
DEFINE_ELEMENT_BINARY_OPERATOR(OPERATOR_ELEMENT_LEQ_TYPE, operator <=)
DEFINE_ELEMENT_BINARY_OPERATOR(OPERATOR_ELEMENT_EQ_TYPE, operator ==)
DEFINE_ELEMENT_BINARY_OPERATOR(OPERATOR_ELEMENT_NEQ_TYPE, operator !=)

DEFINE_ELEMENT_BINARY_OPERATOR(OPERATOR_ELEMENT_MAX_TYPE, max)
DEFINE_ELEMENT_BINARY_OPERATOR(OPERATOR_ELEMENT_MIN_TYPE, min)
DEFINE_ELEMENT_BINARY_OPERATOR(OPERATOR_ELEMENT_POW_TYPE, pow)

namespace detail
{
  DEFINE_ELEMENT_BINARY_OPERATOR(OPERATOR_ASSIGN_TYPE, assign)
}

#undef DEFINE_ELEMENT_BINARY_OPERATOR
//---------------------------------------

/*--- Math Operators----*/
//---------------------------------------
#define DEFINE_ELEMENT_UNARY_OPERATOR(OP, OPNAME) \
array_expression OPNAME (array  const & x) \
{ return array_expression(x, lhs_rhs_element(), op_element(OPERATOR_UNARY_TYPE_FAMILY, OP), x.context(), x.dtype(), x.shape()); }\
\
array_expression OPNAME (array_expression const & x) \
{ return array_expression(x, lhs_rhs_element(), op_element(OPERATOR_UNARY_TYPE_FAMILY, OP), x.shape()); }

DEFINE_ELEMENT_UNARY_OPERATOR((x.dtype()==FLOAT_TYPE || x.dtype()==DOUBLE_TYPE)?OPERATOR_FABS_TYPE:OPERATOR_ABS_TYPE,  abs)
DEFINE_ELEMENT_UNARY_OPERATOR(OPERATOR_ACOS_TYPE, acos)
DEFINE_ELEMENT_UNARY_OPERATOR(OPERATOR_ASIN_TYPE, asin)
DEFINE_ELEMENT_UNARY_OPERATOR(OPERATOR_ATAN_TYPE, atan)
DEFINE_ELEMENT_UNARY_OPERATOR(OPERATOR_CEIL_TYPE, ceil)
DEFINE_ELEMENT_UNARY_OPERATOR(OPERATOR_COS_TYPE,  cos)
DEFINE_ELEMENT_UNARY_OPERATOR(OPERATOR_COSH_TYPE, cosh)
DEFINE_ELEMENT_UNARY_OPERATOR(OPERATOR_EXP_TYPE,  exp)
DEFINE_ELEMENT_UNARY_OPERATOR(OPERATOR_FLOOR_TYPE, floor)
DEFINE_ELEMENT_UNARY_OPERATOR(OPERATOR_LOG_TYPE,  log)
DEFINE_ELEMENT_UNARY_OPERATOR(OPERATOR_LOG10_TYPE,log10)
DEFINE_ELEMENT_UNARY_OPERATOR(OPERATOR_SIN_TYPE,  sin)
DEFINE_ELEMENT_UNARY_OPERATOR(OPERATOR_SINH_TYPE, sinh)
DEFINE_ELEMENT_UNARY_OPERATOR(OPERATOR_SQRT_TYPE, sqrt)
DEFINE_ELEMENT_UNARY_OPERATOR(OPERATOR_TAN_TYPE,  tan)
DEFINE_ELEMENT_UNARY_OPERATOR(OPERATOR_TANH_TYPE, tanh)
#undef DEFINE_ELEMENT_UNARY_OPERATOR
//---------------------------------------

///*--- Misc----*/
////---------------------------------------
inline size4 trans(size4 const & shape)
{ return size4(shape._2, shape._1);}

inline size4 prod(size4 const & shape1, size4 const & shape2)
{ return size4(shape1._1*shape2._1, shape1._2*shape2._2);}

array_expression trans(array  const & x) \
{ return array_expression(x, lhs_rhs_element(), op_element(OPERATOR_UNARY_TYPE_FAMILY, OPERATOR_TRANS_TYPE), x.context(), x.dtype(), trans(x.shape())); }\
\
array_expression trans(array_expression const & x) \
{ return array_expression(x, lhs_rhs_element(), op_element(OPERATOR_UNARY_TYPE_FAMILY, OPERATOR_TRANS_TYPE), trans(x.shape())); }

array_expression repmat(array const & A, int_t const & rep1, int_t const & rep2)
{
  static array_repeat_infos infos(A.shape(), size4(rep1, rep2));
  infos = array_repeat_infos(A.shape(), size4(rep1, rep2));
  size4 newshape = prod(infos.sub, infos.rep);
  return array_expression(A, infos, op_element(OPERATOR_BINARY_TYPE_FAMILY, OPERATOR_MATRIX_REPEAT_TYPE), A.context(), A.dtype(), newshape);
}

array_expression repmat(array_expression const & A, int_t const & rep1, int_t const & rep2)
{
  static array_repeat_infos infos(A.shape(), size4(rep1, rep2));
  infos = array_repeat_infos(A.shape(), size4(rep1, rep2));
  size4 newshape = prod(infos.sub, infos.rep);
  return array_expression(A, infos, op_element(OPERATOR_BINARY_TYPE_FAMILY, OPERATOR_MATRIX_REPEAT_TYPE), newshape);
}

////---------------------------------------

///*--- Reductions ---*/
////---------------------------------------
#define DEFINE_REDUCTION(OP, OPNAME)\
array_expression OPNAME(array const & x, int_t axis)\
{\
  if(axis==-1)\
    return array_expression(x, lhs_rhs_element(), op_element(OPERATOR_VECTOR_REDUCTION_TYPE_FAMILY, OP), x.context(), x.dtype(), size4(1));\
  else if(axis==0)\
    return array_expression(x, lhs_rhs_element(), op_element(OPERATOR_ROWS_REDUCTION_TYPE_FAMILY, OP), x.context(), x.dtype(), size4(x.shape()._1));\
  else if(axis==1)\
    return array_expression(x, lhs_rhs_element(), op_element(OPERATOR_COLUMNS_REDUCTION_TYPE_FAMILY, OP), x.context(), x.dtype(), size4(x.shape()._2));\
  else\
    throw "invalid shape";\
}\
\
array_expression OPNAME(array_expression const & x, int_t axis)\
{\
  if(axis==-1)\
    return array_expression(x, lhs_rhs_element(), op_element(OPERATOR_VECTOR_REDUCTION_TYPE_FAMILY, OP), size4(1));\
  else if(axis==0)\
    return array_expression(x, lhs_rhs_element(), op_element(OPERATOR_ROWS_REDUCTION_TYPE_FAMILY, OP), size4(x.shape()._1));\
  else if(axis==1)\
    return array_expression(x, lhs_rhs_element(), op_element(OPERATOR_COLUMNS_REDUCTION_TYPE_FAMILY, OP), size4(x.shape()._2));\
  else\
    throw "invalid shape";\
}

DEFINE_REDUCTION(OPERATOR_ADD_TYPE, sum)
DEFINE_REDUCTION(OPERATOR_ELEMENT_ARGMAX_TYPE, argmax)
DEFINE_REDUCTION(OPERATOR_ELEMENT_MAX_TYPE, max)
DEFINE_REDUCTION(OPERATOR_ELEMENT_MIN_TYPE, min)
DEFINE_REDUCTION(OPERATOR_ELEMENT_ARGMIN_TYPE, argmin)

#undef DEFINE_REDUCTION

namespace detail
{

  array_expression matmatprod(array const & A, array const & B)
  {
    size4 shape(A.shape()._1, B.shape()._2);
    return array_expression(A, B, op_element(OPERATOR_MATRIX_PRODUCT_TYPE_FAMILY, OPERATOR_MATRIX_PRODUCT_NN_TYPE), A.context(), A.dtype(), shape);
  }

  array_expression matmatprod(array_expression const & A, array const & B)
  {
    operation_node_type type = OPERATOR_MATRIX_PRODUCT_NN_TYPE;
    size4 shape(A.shape()._1, B.shape()._2);

    symbolic_expression_node & A_root = const_cast<symbolic_expression_node &>(A.array()[A.root()]);
    bool A_trans = A_root.op.type==OPERATOR_TRANS_TYPE;
    if(A_trans){
      type = OPERATOR_MATRIX_PRODUCT_TN_TYPE;
      shape._1 = A.shape()._2;
    }

    array_expression res(A, B, op_element(OPERATOR_MATRIX_PRODUCT_TYPE_FAMILY, type), shape);
    symbolic_expression_node & res_root = const_cast<symbolic_expression_node &>(res.array()[res.root()]);
    if(A_trans) res_root.lhs = A_root.lhs;
    return res;
  }

  array_expression matmatprod(array const & A, array_expression const & B)
  {
    operation_node_type type = OPERATOR_MATRIX_PRODUCT_NN_TYPE;
    size4 shape(A.shape()._1, B.shape()._2);

    symbolic_expression_node & B_root = const_cast<symbolic_expression_node &>(B.array()[B.root()]);
    bool B_trans = B_root.op.type==OPERATOR_TRANS_TYPE;
    if(B_trans){
      type = OPERATOR_MATRIX_PRODUCT_NT_TYPE;
      shape._2 = B.shape()._1;
    }
    array_expression res(A, B, op_element(OPERATOR_MATRIX_PRODUCT_TYPE_FAMILY, type), shape);
    symbolic_expression_node & res_root = const_cast<symbolic_expression_node &>(res.array()[res.root()]);
    if(B_trans) res_root.rhs = B_root.lhs;
    return res;
  }

  array_expression matmatprod(array_expression const & A, array_expression const & B)
  {
    operation_node_type type = OPERATOR_MATRIX_PRODUCT_NN_TYPE;
    symbolic_expression_node & A_root = const_cast<symbolic_expression_node &>(A.array()[A.root()]);
    symbolic_expression_node & B_root = const_cast<symbolic_expression_node &>(B.array()[B.root()]);
    size4 shape(A.shape()._1, B.shape()._2);

    bool A_trans = A_root.op.type==OPERATOR_TRANS_TYPE;
    bool B_trans = B_root.op.type==OPERATOR_TRANS_TYPE;
    if(A_trans) shape._1 = A.shape()._2;
    if(B_trans) shape._2 = B.shape()._1;
    if(A_trans && B_trans)  type = OPERATOR_MATRIX_PRODUCT_TT_TYPE;
    else if(A_trans && !B_trans) type = OPERATOR_MATRIX_PRODUCT_TN_TYPE;
    else if(!A_trans && B_trans) type = OPERATOR_MATRIX_PRODUCT_NT_TYPE;
    else type = OPERATOR_MATRIX_PRODUCT_NN_TYPE;

    array_expression res(A, B, op_element(OPERATOR_MATRIX_PRODUCT_TYPE_FAMILY, type), shape);
    symbolic_expression_node & res_root = const_cast<symbolic_expression_node &>(res.array()[res.root()]);
    if(A_trans) res_root.lhs = A_root.lhs;
    if(B_trans) res_root.rhs = B_root.lhs;
    return res;
  }


  template<class T>
  array_expression matvecprod(array const & A, T const & x)
  {
    int_t M = A.shape()._1;
    int_t N = A.shape()._2;
    return sum(A*repmat(const_cast<T&>(x).reshape(1, N), M, 1), 0);
  }

  template<class T>
  array_expression matvecprod(array_expression const & A, T const & x)
  {
    int_t M = A.shape()._1;
    int_t N = A.shape()._2;
    symbolic_expression_node & A_root = const_cast<symbolic_expression_node &>(A.array()[A.root()]);
    bool A_trans = A_root.op.type==OPERATOR_TRANS_TYPE;
    if(A_trans){
      array_expression tmp(A, repmat(const_cast<T&>(x), 1, M), op_element(OPERATOR_BINARY_TYPE_FAMILY, OPERATOR_ELEMENT_PROD_TYPE), size4(N, M));
      //Remove trans
      tmp.array()[tmp.root()].lhs = A.array()[A.root()].lhs;
      return sum(tmp, 1);
    }
    else
      return sum(A*repmat(const_cast<T&>(x).reshape(1, N), M, 1), 0);

  }

  array_expression matvecprod(array_expression const & A, array_expression const & x)
  {
    return matvecprod(A, array(x));
  }


}

#define DEFINE_DOT(LTYPE, RTYPE) \
array_expression dot(LTYPE const & x, RTYPE const & y)\
{\
  if(x.nshape()==1 && y.nshape()==1)\
  {\
    return sum(x*y);\
  }\
  else if(x.nshape()==2 && y.nshape()==1)\
  {\
    return detail::matvecprod(x, y);\
  }\
  else if(x.nshape()==1 && y.nshape()==2)\
  {\
    return detail::matvecprod(trans(y), x);\
  }\
  else /*if(x.nshape()==2 && y.nshape()==2)*/\
  {\
    return detail::matmatprod(x, y);\
  }\
}

DEFINE_DOT(array, array)
DEFINE_DOT(array_expression, array)
DEFINE_DOT(array, array_expression)
DEFINE_DOT(array_expression, array_expression)

#undef DEFINE_DOT



/*--- Copy ----*/
//---------------------------------------

//void*
void copy(void const * data, array& x, cl::CommandQueue & queue, bool blocking)
{
  unsigned int dtypesize = size_of(x.dtype());
  if(x.ld()==x.shape()._1)
  {
    queue.enqueueWriteBuffer(x.data(), CL_FALSE, 0, x.dsize()*dtypesize, data);
  }
  else
  {
    array tmp(x.shape()._1, x.shape()._2, x.dtype(), x.context());
    queue.enqueueWriteBuffer(x.data(), CL_FALSE, 0, tmp.dsize()*dtypesize, data);
    x = tmp;
  }
  if(blocking)
    cl::synchronize(x.context());
}

void copy(array const & x, void* data, cl::CommandQueue & queue, bool blocking)
{
  unsigned int dtypesize = size_of(x.dtype());
  if(x.ld()==x.shape()._1)
  {
    queue.enqueueReadBuffer(x.data(), CL_FALSE, 0, x.dsize()*dtypesize, data);
  }
  else
  {
    array tmp(x.shape()._1, x.shape()._2, x.dtype(), x.context());
    tmp = x;
    queue.enqueueReadBuffer(tmp.data(), CL_FALSE, 0, tmp.dsize()*dtypesize, data);
  }
  if(blocking)
    cl::synchronize(x.context());
}

void copy(void const *data, array &x, bool blocking)
{ copy(data, x, cl::queues[x.context()].front(), blocking); }

void copy(array const & x, void* data, bool blocking)
{ copy(x, data, cl::queues[x.context()].front(), blocking); }

//std::vector<>
template<class T>
void copy(std::vector<T> const & cx, array & x, cl::CommandQueue & queue, bool blocking)
{
  if(x.ld()==x.shape()._1)
    assert(cx.size()==x.dsize());
  else
    assert(cx.size()==prod(x.shape()));
  copy((void const*)cx.data(), x, queue, blocking);
}

template<class T>
void copy(array const & x, std::vector<T> & cx, cl::CommandQueue & queue, bool blocking)
{
  if(x.ld()==x.shape()._1)
    assert(cx.size()==x.dsize());
  else
    assert(cx.size()==prod(x.shape()));
  copy(x, (void*)cx.data(), queue, blocking);
}

template<class T>
void copy(std::vector<T> const & cx, array & x, bool blocking)
{ copy(cx, x, cl::queues[x.context()].front(), blocking); }

template<class T>
void copy(array const & x, std::vector<T> & cx, bool blocking)
{ copy(x, cx, cl::queues[x.context()].front(), blocking); }

#define INSTANTIATE(T) \
  template void copy<T>(std::vector<T> const &, array &, cl::CommandQueue&, bool);\
  template void copy<T>(array const &, std::vector<T> &, cl::CommandQueue&, bool);\
  template void copy<T>(std::vector<T> const &, array &, bool);\
  template void copy<T>(array const &, std::vector<T> &, bool)

INSTANTIATE(cl_char);
INSTANTIATE(cl_uchar);
INSTANTIATE(cl_short);
INSTANTIATE(cl_ushort);
INSTANTIATE(cl_int);
INSTANTIATE(cl_uint);
INSTANTIATE(cl_long);
INSTANTIATE(cl_ulong);
INSTANTIATE(cl_float);
INSTANTIATE(cl_double);

#undef INSTANTIATE
/*--- Stream operators----*/
//---------------------------------------

namespace detail
{
  template<typename ItType>
  static std::ostream & prettyprint(std::ostream& os, ItType begin, ItType const & end, size_t stride = 1, bool col = false, size_t WINDOW = 10)
  {
    if(!col)
      os << "[ " ;
    size_t N = (end - begin)/stride;
    size_t upper = std::min(WINDOW,N);
    for(size_t j = 0; j < upper ; j++)
    {
      if(col)
        os << "\t|" << *begin << "|";
      else{
        os << *begin;
        if(j<upper - 1)
          os << ",";
      }
      begin+=stride;
    }
    if(upper < N)
    {
      if(N - upper > WINDOW)
        os << ", ... ";
      for(size_t j = std::max(N - WINDOW, upper) ; j < N ; j++)
      {
        if(col)
          os << "\t|" << *begin << "|";
        else{
          os << "," << *begin;
        }
        begin+=stride;
      }
    }
    if(!col)
      os << " ]" ;
    return os;
  }

}

std::ostream& operator<<(std::ostream & os, array const & a)
{
  size_t WINDOW = 10;

  size_t M = a.shape()._1;
  size_t N = a.shape()._2;

  std::vector<float> tmp(M*N);
  copy(a, tmp);

  os << "[ " ;
  size_t upper = std::min(WINDOW,M);
  for(unsigned int i = 0 ; i < upper ; ++i)
  {
    if(i>0)
      os << "  ";
    detail::prettyprint(os, tmp.begin() + i, tmp.end() + i, M, true, WINDOW);
    if(i < upper-1)
      os <<  std::endl;
  }
  if(upper < M)
  {
    if(N - upper > WINDOW)
      os << std::endl << "  ... ";
    for(size_t i = std::max(N - WINDOW, upper) ; i < N ; i++)
    {
      os << std::endl << "  ";
      detail::prettyprint(os, tmp.begin() + i, tmp.end() + i, M, true, WINDOW);
    }
  }
  os << " ]";
  return os;
}

}