/*
 * Copyright (c) 2015, PHILIPPE TILLET. All rights reserved.
 *
 * This file is part of ISAAC.
 *
 * ISAAC is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
 * MA 02110-1301  USA
 */

#include "isaac/array.h"
#include "isaac/driver/command_queue.h"
#include "isaac/driver/kernel.h"
#include "isaac/driver/ndrange.h"
#include "isaac/jit/syntax/expression/preset.h"
#include "isaac/jit/syntax/engine/process.h"
#include "isaac/jit/generation/matrix_product.h"
#include "isaac/jit/exceptions.h"
#include "tools/arguments.hpp"
#include "tools/vector_types.hpp"


#include <string>
#include "isaac/tools/cpp/align.hpp"

namespace isaac
{
namespace templates
{



  size_t matrix_product::lmem_usage(expression const & expression) const
  {
    size_t N = 0;
    N += kL * mL;
    N += nL * kL;
    return N*size_of(expression.dtype());
  }

  size_t matrix_product::registers_usage(expression const & expression) const
  {
    size_t N = mS * nS + mS * kS + kS * nS;
    return N*size_of(expression.dtype());
  }

  size_t matrix_product::temporary_workspace(expression const & expressions) const
  {
      std::vector<int_t> MNK = input_sizes(expressions);
      int_t M = MNK[0]; int_t N = MNK[1];
      if(depth > 1)
        return M*N*depth;
      return 0;
  }

  void matrix_product::check_valid_impl(driver::Device const &, expression const &) const
  {
    if(A_fetching_policy!=FETCH_FROM_LOCAL || B_fetching_policy!=FETCH_FROM_LOCAL)
      throw jit::code_generation_error("generated code uses unsupported fetching policy");

    if ((mS % simd_width) > 0)
      throw jit::code_generation_error("GEMM - MS must be a multiple of simd_width");

    if ((nS % simd_width) > 0)
      throw jit::code_generation_error("GEMM - NS must be a multiple of simd_width");

    if(mL > 256)
      throw jit::code_generation_error("GEMM - ML too large (> 256)");

    if(nL > 256)
      throw jit::code_generation_error("GEMM - NL too large (> 256)");

    if ( kS % kL == 0)
      throw jit::code_generation_error("GEMM - KS must be smaller than KL");

    if (A_fetching_policy==FETCH_FROM_LOCAL || B_fetching_policy==FETCH_FROM_LOCAL){
      if ((local_fetch_0*local_fetch_1) !=(local_size_0*local_size_1))
        throw jit::code_generation_error("GEMM - fetch_0*fetch_1 must equal local_size_0*local_size_1");
    }

    if (A_fetching_policy==FETCH_FROM_LOCAL)
    {
      size_t bound1 = (A_trans_=='N')?kL:mL;
      size_t bound0 = (A_trans_=='N')?mL:kL;

      if (local_fetch_1>0 && (bound1 % local_fetch_1)> 0){
        if(A_trans_=='N')
          throw jit::code_generation_error("GEMM - fetch_1 must be a multiple of KL");
        else
          throw jit::code_generation_error("GEMM - fetch_1 must be a multiple of ML");
      }

      if (local_fetch_0>0 && (bound0 % (local_fetch_0*simd_width)) > 0){
        if(A_trans_=='N')
          throw jit::code_generation_error("GEMM - fetch_0 must be a multiple of NL");
        else
          throw jit::code_generation_error("GEMM - fetch_0 must be a multiple of KL");
      }
    }

    if (B_fetching_policy==FETCH_FROM_LOCAL)
    {
      size_t bound1 = (B_trans_=='T')?kL:nL;
      size_t bound0 = (B_trans_=='T')?nL:kL;

      if (local_fetch_1>0 && (bound1 % local_fetch_1)> 0){
        if(B_trans_=='T')
          throw jit::code_generation_error("GEMM - fetch_1 must be a multiple of KL");
        else
          throw jit::code_generation_error("GEMM - fetch_1 must be a multiple of ML");

      }

      if (local_fetch_0>0 && (bound0 % (local_fetch_0*simd_width)) > 0){
        if(B_trans_=='T')
          throw jit::code_generation_error("GEMM - fetch_0 must be a multiple of NL");
        else
          throw jit::code_generation_error("GEMM - fetch_0 must be a multiple of KL");

      }
    }
  }

  std::string matrix_product::generate_impl(std::string const & suffix, expression const & tree, driver::Device const & device, symbolic::symbols_table const &) const
  {
    using std::string;
    using tools::to_string;

    driver::backend_type backend = device.backend();
    bool has_depth = depth > 1;
#define VLOAD(offset, ptr) vload(simd_width, sdtype, offset, ptr, "1", backend, true)
#define VLOAD_MISALIGNED(offset, ptr) vload(simd_width, sdtype, offset, ptr, "1", backend, false)
#define VSTORE(value, offset, ptr) vstore(simd_width, sdtype, value, offset, ptr, "1", backend)

    symbolic::preset::matrix_product::args args;
    infos(tree, args);
    std::string ASTRIDE1 = (args.A->ld[0] > 1)?"*Astride1":"";
    std::string BSTRIDE1 = (args.B->ld[0] > 1)?"*Bstride1":"";
    std::string CSTRIDE1 = (args.C->ld[0] > 1)?"*Cstride1":"";

    //////////////////
    /// INIT
    /// //////////////
    genstream stream(backend);
    numeric_type dtype = tree.dtype();
    std::string sdtype = to_string(dtype);
    std::string vdtype = append_width(sdtype, simd_width);

    //////////////////
    /// DECLARATIONS
    /// //////////////
    std::string matrix_product_name = "matrix_product";
    std::string reduce_name = "reduce";

    matrix_product_name += suffix;
    reduce_name += suffix;

    switch(backend)
    {
      case driver::OPENCL:
        stream << " __attribute__((reqd_work_group_size(" << local_size_0 << "," << local_size_1 << ",1)))" << std::endl;
        break;
      default:
        break;
    }

    stream << "$KERNEL void matrix_product" << suffix << "($SIZE_T M, $SIZE_T N, $SIZE_T K, "
                               << "$GLOBAL " << sdtype << "* C, $SIZE_T ldc, $SIZE_T offc, $SIZE_T Cstride1, "
                               << sdtype << " alpha,"
                               << "$GLOBAL " << sdtype << "* A, $SIZE_T lda, $SIZE_T offa, $SIZE_T Astride1,"
                               << "$GLOBAL " << sdtype << "* B, $SIZE_T ldb, $SIZE_T offb, $SIZE_T Bstride1,"
                               << sdtype << " beta)"
                               << std::endl;
    stream << "{" << std::endl;
    stream.inc_tab();

    ///Declare
    stream << "//blocks" << std::endl;
    stream << sdtype << " rC[" << mS << "][" << nS << "] = {{0}};" << std::endl;
    stream << vdtype << " rA[" << kS << "][" << mS/simd_width << "];" << std::endl;
    stream << vdtype << " rB[" << kS << "][" << nS/simd_width << "];" << std::endl;
    stream << std::endl;

    stream << "//pointers" << std::endl;
    size_t llda = (A_trans_=='N')?mL:kL;
    size_t lldb = (B_trans_=='T')?nL:kL;
    stream << "$LOCAL " << sdtype << " lA[" << kL*mL << "];" << std::endl;
    stream << "$LOCAL " << sdtype << " lB[" << kL*nL << "];" << std::endl;
    unsigned int npA = mL/(A_trans_=='N'?local_fetch_0*simd_width:local_fetch_1);
    unsigned int npB = nL/(B_trans_=='T'?local_fetch_0*simd_width:local_fetch_1);
    stream << "$GLOBAL " << sdtype << "* Ai[" << npA << "];" << std::endl;
    stream << "$GLOBAL " << sdtype << "* Bi[" << npB << "];" << std::endl;
    stream << std::endl;

    stream << "//identifiers" << std::endl;
    stream << "int2 idT;" << std::endl;
    stream << "int idt;" << std::endl;
    if(has_depth)
        stream << "int gidz, div, offz;" << std::endl;
    stream << "uint4 ids;" << std::endl;
    stream << "ids.x = $GROUP_IDX_0;" << std::endl;
    stream << "ids.y = $GROUP_IDX_1;" << std::endl;
    stream << "ids.z = $LOCAL_IDX_0;" << std::endl;
    stream << "ids.w = $LOCAL_IDX_1;" << std::endl;
    stream << std::endl;

    stream << "//offsets" << std::endl;
    stream << "A += offa;" << std::endl;
    stream << "B += offb;" << std::endl;
    stream << "C += offc;" << std::endl;

    if(has_depth)
    {
      stream << "gidz = $GROUP_IDX_2;" << std::endl;
      stream << "div = (K+" << depth-1 << ")/" << depth << ";" << std::endl;
      stream << "offz = div*gidz;" << std::endl;
      stream << "K = min(K - div*gidz, ($SIZE_T)div);" << std::endl;
    }

    stream << "idt = " << local_size_0 << "*ids.w + ids.z;" << std::endl;
    stream << "idT.y = idt/" << local_fetch_0 << ";" << std::endl;
    stream << "idT.x = idt - " << local_fetch_0 << "*idT.y;" << std::endl;
    stream << std::endl;

    stream << "//Adjust pointers and bounds per work-item" << std::endl;
    stream << "ids.x *= " << mL << ";" << std::endl;
    stream << "ids.y *= " << nL << ";" << std::endl;
    stream << "idT.x *= " << simd_width << ";" << std::endl;

    stream << "M -= ids.x;" << std::endl;
    if(A_trans_=='N')
        stream << "M -= idT.x;" << std::endl;
    else
        stream << "M -= idT.y;" << std::endl;

    stream << "N -= ids.y;" << std::endl;
    if(B_trans_=='T')
        stream << "N -= idT.x;" << std::endl;
    else
        stream << "N -= idT.y;" << std::endl;

    if (A_trans_=='N')
    {
        stream << "A += ids.x" << ASTRIDE1 << ";" << std::endl;
        stream << "A += idT.y*lda;" << std::endl;
        if(has_depth)
            stream << "A += offz*lda;" << std::endl;

    }
    else
    {
        stream << "A += ids.x*lda;" << std::endl;
        stream << "A += idT.x" << ASTRIDE1 << ";" << std::endl;
        if(has_depth)
            stream << "A += offz;" << std::endl;
    }

    if(B_trans_=='T')
    {
        stream << "B += ids.y" << BSTRIDE1 << ";" << std::endl;
        stream << "B += idT.y*ldb;" << std::endl;
        if(has_depth)
            stream << "B += offz*ldb;" << std::endl;
    }
    else
    {
        stream << "B += ids.y*ldb;" << std::endl;
        stream << "B += idT.x" << BSTRIDE1 << ";" << std::endl;
        if(has_depth)
            stream << "B += offz;" << std::endl;
    }

    stream << "#pragma unroll" << std::endl;
    stream << "for(int i = 0 ; i < " << npA << " ; ++i){" << std::endl;
    stream.inc_tab();
    stream << "Ai[i] = A;" << std::endl;
    stream.dec_tab();
    stream << "}" << std::endl;
    stream << std::endl;

    stream << "#pragma unroll" << std::endl;
    stream << "for(int i = 0 ; i < " << npB << " ; ++i){" << std::endl;
    stream.inc_tab();
    stream << "Bi[i] = B;" << std::endl;
    stream.dec_tab();
    stream << "}" << std::endl;
    stream << std::endl;


    for(unsigned int i = 0 ; i < npA ; i++ )
        if (A_trans_=='N')
          stream << "Ai[" << i << "] += (" << i*local_fetch_0*simd_width << " < M)? (int)((idT.x + " << i*local_fetch_0*simd_width << ")" + ASTRIDE1 + "):0;" << std::endl;
        else
          stream << "Ai[" << i << "] += (" << i*local_fetch_1 << " < M)? (int)((idT.y + " << i*local_fetch_1 << ")*lda):0;" << std::endl;

    for(unsigned int i = 0 ; i < npB ; i++ )
        if (B_trans_=='T')
            stream << "Bi[" << i << "] += (" << i*local_fetch_0*simd_width << " < N)?(int)((idT.x + " << i*local_fetch_0*simd_width << ")" + BSTRIDE1 + "):0;" << std::endl;
        else
            stream << "Bi[" << i << "] += (" << i*local_fetch_1 << " < N)?(int)((idT.y + " << i*local_fetch_1 << ")*ldb):0;" << std::endl;

    stream << std::endl;
    stream << "//Outer loop" << std::endl;
    stream << "while(K >=" << kL << ")" << std::endl;
    stream << "{" << std::endl;
    stream.inc_tab();


    auto fetch_to_lds = [&](bool last_iteration)
    {
        stream << "$LOCAL_BARRIER;" << std::endl;
        stream << "$LOCAL_PTR " << sdtype << "* ldsA = lA + idT.y*" << llda << " + idT.x;" << std::endl;
        stream << "$LOCAL_PTR " << sdtype << "* ldsB = lB + idT.y*" << lldb << " + idT.x;" << std::endl;

        stream << "//Fetch A to local memory" << std::endl;
        if (A_trans_=='N')
        {
          for(unsigned int k = 0; k < kL; k += local_fetch_1)
            for(unsigned int m = 0; m < mL; m += local_fetch_0*simd_width)
            {
              std::string mm = to_string(m/(simd_width*local_fetch_0));
              std::string kk = to_string(k);
              if(last_iteration)
                  for(unsigned int s = 0 ; s < simd_width ; ++s)
                      stream << "ldsA[" << k*llda + m + s << "] = (condy" << k << " && " << s << "< M)? Ai[" << mm << "][" << k << "*lda + " << s << "] : 0;" << std::endl;
              else
                stream << VSTORE(VLOAD_MISALIGNED("0" ,"&Ai[" + mm +"][" + kk + "*lda]"), "0", "ldsA + " + to_string(k*llda+m)) << ";" << std::endl;
            }
        }
        else
        {
            for(unsigned int k = 0; k < kL; k += local_fetch_0*simd_width)
            for(unsigned int m = 0; m < mL; m += local_fetch_1)
              {
                std::string mm = to_string(m/local_fetch_1);
                std::string kk = to_string(k);
                if(last_iteration)
                    for(unsigned int s = 0 ; s < simd_width ; ++s)
                        stream << "ldsA[" << m*llda + k + s << "] = condx" << k + s << "? Ai[" << mm << "][" << k + s << ASTRIDE1 << "] : 0;" << std::endl;

                else
                    stream << VSTORE(VLOAD_MISALIGNED("0", "&Ai[" + mm + "][" + kk + ASTRIDE1 + "]"), "0", "ldsA + " + to_string(m*llda+k)) << ";" << std::endl;
              }
        }

        stream << "//Fetch B to local memory" << std::endl;
        if (B_trans_=='T')
        {
          for(unsigned int k = 0; k < kL; k += local_fetch_1)
            for(unsigned int n = 0; n < nL; n += local_fetch_0*simd_width)
            {
              std::string nn = to_string(n/(simd_width*local_fetch_0));
              std::string kk = to_string(k);
              if(last_iteration)
                  for(unsigned int s = 0 ; s < simd_width ; ++s)
                      stream << "ldsB[" << k*lldb + n + s << "] = (condy" << k << " && " << s << "< N)? Bi[" <<  nn << "][" << kk << "*ldb +" << s << "] : 0;" << std::endl;
              else
                stream << VSTORE(VLOAD_MISALIGNED("0" ,"&Bi[" + nn +"][" + kk + "*ldb]"), "0", "ldsB + " + to_string(k*lldb+n)) << ";" << std::endl;
            }
        }
        else
        {
          for(unsigned int k = 0; k < kL; k += local_fetch_0*simd_width)
            for(unsigned int n = 0; n < nL; n += local_fetch_1)
            {
              std::string nn = to_string(n/local_fetch_1);
              std::string kk = to_string(k);
              if(last_iteration)
                  for(unsigned int s = 0 ; s < simd_width ; ++s)
                      stream << "ldsB[" << n*lldb + k + s << "] = condx" << k + s << "? Bi[" << nn << "][" << k + s << BSTRIDE1 << "] : 0;" << std::endl;

              else
                  stream << VSTORE(VLOAD_MISALIGNED("0", "&Bi[" + nn + "][" + kk + BSTRIDE1 + "]"), "0", "ldsB + " + to_string(n*lldb+k)) << ";" << std::endl;
            }
        }

        if(A_trans_=='N')
            stream << "ldsA = lA + ids.z*" << simd_width << ";" << std::endl;
        else
            stream << "ldsA = lA + ids.z*" << llda*simd_width << ";" << std::endl;

        if(B_trans_=='T')
            stream << "ldsB = lB + ids.w*" << simd_width << ";" << std::endl;
        else
            stream << "ldsB = lB + ids.w*" << lldb*simd_width << ";" << std::endl;

        stream << "$LOCAL_BARRIER;" << std::endl;

        stream << "//Inner loop" << std::endl;
        stream << "for(unsigned int k = 0; k < " << kL << "; k+=" << kS << "){" << std::endl;
        stream.inc_tab();

        stream << "//Fetch A to registers" << std::endl;
        stream << "#pragma unroll" << std::endl;
        stream << "for(unsigned int kk = 0; kk < " << kS << "; kk++)" << std::endl;
        stream << "#pragma unroll " << mS/simd_width << std::endl;
        stream << "for(unsigned int mm = 0; mm < " << mS/simd_width << "; mm++)" << std::endl;
        stream << "{" << std::endl;
        stream.inc_tab();
        if(A_trans_=='N')
            stream << "rA[kk][mm] = "  << VLOAD("0", "ldsA + k*" + to_string(llda) + " + mm*" + to_string(local_size_0*simd_width) + "+ kk*" + to_string(llda)) << ";" << std::endl;
        else
        {
            if(simd_width==1)
                stream << "rA[kk][mm] = ldsA[k + mm*" << local_size_0*llda <<  "+ kk"  << "];" << std::endl;
            else
                for(unsigned int s = 0 ; s < simd_width ; ++s)
                    stream << access_vector_type("rA[kk][mm]", s) << " = ldsA[k + (mm*" << simd_width*local_size_0 << " + " << s << ")*" << llda <<  "+ kk];" << std::endl;
        }

        stream.dec_tab();
        stream << "}" << std::endl;

        stream << "//Fetch B to registers" << std::endl;
        stream << "#pragma unroll " << kS << std::endl;
        stream << "for(unsigned int kk = 0; kk < " << kS << "; kk++)" << std::endl;
        stream << "#pragma unroll " << nS/simd_width << std::endl;
        stream << "for(unsigned int nn = 0; nn < " << nS/simd_width << "; nn++)" << std::endl;
        stream << "{" << std::endl;
        stream.inc_tab();
        if(B_trans_=='T')
            stream << "rB[kk][nn] = " << VLOAD("0", "ldsB + k*" + to_string(lldb) + " + nn*" + to_string(local_size_1*simd_width)  + "+ kk*" + to_string(lldb)) << ";" << std::endl;
        else
        {
            if(simd_width==1)
                stream << "rB[kk][nn] = ldsB[k"  << " + nn*" << local_size_1*lldb <<  "+ kk"  << "];" << std::endl;
            else
                for(unsigned int s = 0 ; s < simd_width ; ++s)
                    stream << access_vector_type("rB[kk][nn]", s) << " = ldsB[k"  << " + (nn*" << simd_width*local_size_1 << " + " << s << ")*" << lldb <<  "+ kk];" << std::endl;
        }
        stream.dec_tab();
        stream << "}" << std::endl;

        stream << "//FMA computations" << std::endl;
        for(unsigned int kk=0 ; kk < kS; ++kk)
        for(unsigned int nn=0; nn < nS; ++nn)
        for(unsigned int mm=0; mm < mS; ++mm){
          string res_str, lhs_str, rhs_str;
          res_str = "rC[" + to_string(mm) + "][" + to_string(nn) + "]";
          if (simd_width==1)
            lhs_str = "rA[" + to_string(kk) + "][" + to_string(mm) + "]";
          else
            lhs_str = access_vector_type("rA[" + to_string(kk) + "][" + to_string(mm/simd_width) + "]", mm%simd_width);
          if (simd_width==1)
            rhs_str = "rB[" + to_string(kk) + "]["+to_string(nn)+"]";
          else
            rhs_str = access_vector_type("rB[" + to_string(kk) + "]["+to_string(nn/simd_width)+"]", nn%simd_width);
          stream << res_str << "= $MAD(" << lhs_str << "," << rhs_str << "," << res_str << ");" << std::endl;
        }

        stream.dec_tab();
        stream << "}" << std::endl;
        stream << "K -= " << kL << ";" << std::endl;

        //Increment A pointers to global memory
        if (A_trans_=='N')
          for(unsigned int i = 0 ; i < npA ; ++i)
              stream << "Ai[" << i << "] += "  << kL << "*lda;" << std::endl;
        else
          for(unsigned int i = 0 ; i < npA ; ++i)
              stream << "Ai[" << i << "] += "  << kL << ASTRIDE1 << ";" << std::endl;

        //Increment B pointers to global memory
        if (B_trans_=='T')
          for(unsigned int i = 0 ; i < npB ; ++i)
              stream << "Bi[" << i << "] += " << kL << "*ldb;" << std::endl;
        else
          for(unsigned int i = 0 ; i < npB ; ++i)
              stream << "Bi[" << i << "] += " << kL << BSTRIDE1 << ";" << std::endl;
    };
    fetch_to_lds(false);
    stream.dec_tab();
    stream << "}" << std::endl;


    if(A_trans_=='N' || B_trans_=='T')
    {
        stream << "int Ky = K - idT.y;" << std::endl;
        for(unsigned int k = 0; k < kL; k += local_fetch_1)
            stream << "int condy" << k << " = " << k << " < Ky;" << std::endl;
    }

    if(A_trans_=='T' || B_trans_=='N')
    {
        stream << "int Kx = K - idT.x;" << std::endl;
        for(unsigned int k = 0 ; k < kL ; k += local_fetch_0*simd_width)
            for(unsigned int s = 0 ; s < simd_width ; ++s)
                stream << "int condx" << k + s << " = " << k + s << " < Kx;" << std::endl;
    }
    fetch_to_lds(true);

    stream << "//Write back C" << std::endl;
    stream << "M += ids.x;" << std::endl;
    if(A_trans_=='N')
        stream << "M += idT.x;" << std::endl;
    else
        stream << "M += idT.y;" << std::endl;

    if(B_trans_=='T')
        stream << "N += idT.x;" << std::endl;
    else
        stream << "N += idT.y;" << std::endl;
    stream << "N += ids.y;" << std::endl;

    stream << "C += ids.x" << CSTRIDE1 << ";" << std::endl;
    stream << "C += ids.z*" << simd_width << CSTRIDE1 << ";" << std::endl;
    stream << "C += ids.y*ldc;" << std::endl;
    stream << "C += ids.w*" << simd_width << "*ldc;" << std::endl;
    if(has_depth)
        stream << "C += gidz*ldc*N;" << std::endl;

    stream << "M -= ids.x;" << std::endl;
    stream << "M -= ids.z*" << simd_width << ";" << std::endl;

    stream << "N -= ids.y;" << std::endl;
    stream << "N -= ids.w*" << simd_width <<  ";" << std::endl;

    for(unsigned int n=0; n < nS; ++n)
    {
        string Cj = to_string((n/simd_width)*(local_size_1*simd_width) + n%simd_width);
        stream << "if(" << Cj << " >= N) return;" << std::endl;
        for(unsigned int m=0; m < mS; ++m)
            stream << "rC[" << m << "][" << n << "] *= alpha;" << std::endl;
        for(unsigned int m=0; m < mS; ++m)
        {
            string Ci = to_string((m/simd_width)*(local_size_0*simd_width) + m%simd_width);
            stream << "if(" << Ci << "< M) ";
            if(has_depth)
                stream << "C[" << Ci << CSTRIDE1 << "] = rC[" << m << "][" << n << "];" << std::endl;
            else
                stream << "C[" << Ci << CSTRIDE1 << "] = rC[" << m << "][" << n << "] + ((beta != (" << sdtype << ")0)?(beta*" << "C[" << Ci << CSTRIDE1 << "]):0);" << std::endl;
        }
        if((n+1)%simd_width==0){
            stream << "C += ldc*" << local_size_1*simd_width - simd_width + 1 << ";" << std::endl;
        }
        else{
            stream << "C += ldc;" << std::endl;
        }

    }

    stream.dec_tab();
    stream << "}" << std::endl;

    if(has_depth)
    {
      stream << "$KERNEL void reduce" << suffix << "($SIZE_T M, $SIZE_T N, $SIZE_T D, "
                                 << "$GLOBAL " << sdtype << "* Z, $SIZE_T Zld,"
                                 << "$GLOBAL " << sdtype << "* C, $SIZE_T ldc, $SIZE_T Cstart, $SIZE_T Cstride,"
                                 << sdtype << " beta)"
                                 << std::endl;
      stream << "{" << std::endl;
      stream.inc_tab();

      stream << "C += Cstart;" << std::endl;
      stream << "for(unsigned int i = $GLOBAL_IDX_0 ;  i < M ;  i += $GLOBAL_SIZE_0)" << std::endl;
      stream << "{" << std::endl;
      stream.inc_tab();
      stream << "for(unsigned int j = $GLOBAL_IDX_1 ;  j < N ;  j += $GLOBAL_SIZE_1)" << std::endl;
      stream << "{" << std::endl;
      stream.inc_tab();
      stream << sdtype << " acc = 0;" << std::endl;
      stream << "for(unsigned int k = 0 ;  k < D ;  k++)" << std::endl;
      stream.inc_tab();
      stream << "acc += Z[i + j*Zld + k*Zld*N];" << std::endl;
      stream.dec_tab();
      stream << "C[i*Cstride + j*ldc] = acc + beta*C[i*Cstride + j*ldc];" << std::endl;
      stream.dec_tab();
      stream << "}" << std::endl;
      stream.dec_tab();
      stream << "}" << std::endl;

      stream.dec_tab();
      stream << "}" << std::endl;
    }

    return stream.str();

#undef VLOAD
#undef VST0RE
  }

  void matrix_product::enqueue_block(driver::CommandQueue & queue, int_t M, int_t N, int_t K,
                     expression::node const & A, expression::node const & B, expression::node const & C,
                     scalar const & alpha, scalar const & beta,
                     driver::Program const & program, std::string const & suffix, runtime::environment const & options)
  {
    using tools::align;

    if(M==0 || N==0 || K==0)
      return;

    driver::backend_type backend = queue.context().backend();

    std::string matrix_product_name = "matrix_product";
    std::string reduce_name = "reduce";

    matrix_product_name += suffix;
    reduce_name += suffix;

    driver::Kernel matrix_product(program, matrix_product_name.c_str());
    driver::NDRange local(local_size_0, local_size_1, 1);
    driver::NDRange global(align(align(M,mS)/mS, local_size_0), align(align(N,nS)/nS, local_size_1), depth);

    size_t current_arg = 0;

    driver::Buffer& workspace = driver::backend::workspaces::get(options.queue(queue.context()));
    matrix_product.setSizeArg(current_arg++, M);
    matrix_product.setSizeArg(current_arg++, N);
    matrix_product.setSizeArg(current_arg++, K);
    if(depth==1)
    {
        if(backend==driver::OPENCL)
          matrix_product.setArg(current_arg++, C.array.handle.cl);
        else
          matrix_product.setArg(current_arg++, C.array.handle.cu);
        matrix_product.setSizeArg(current_arg++, C.ld[1]);
        matrix_product.setSizeArg(current_arg++, C.array.start);
        matrix_product.setSizeArg(current_arg++, C.ld[0]);
    }
    else
    {
        matrix_product.setArg(current_arg++, workspace);
        matrix_product.setSizeArg(current_arg++, M);
        matrix_product.setSizeArg(current_arg++, 0);
        matrix_product.setSizeArg(current_arg++, 1);
    }


    matrix_product.setArg(current_arg++, alpha);
    if(backend==driver::OPENCL)
      matrix_product.setArg(current_arg++, A.array.handle.cl);
    else
      matrix_product.setArg(current_arg++, A.array.handle.cu);
    matrix_product.setSizeArg(current_arg++, A.ld[1]);
    matrix_product.setSizeArg(current_arg++, A.array.start);
    matrix_product.setSizeArg(current_arg++, A.ld[0]);

    if(backend==driver::OPENCL)
      matrix_product.setArg(current_arg++, B.array.handle.cl);
    else
      matrix_product.setArg(current_arg++, B.array.handle.cu);
    matrix_product.setSizeArg(current_arg++, B.ld[1]);
    matrix_product.setSizeArg(current_arg++, B.array.start);
    matrix_product.setSizeArg(current_arg++, B.ld[0]);

    matrix_product.setArg(current_arg++, beta);
    options.enqueue(queue.context(), matrix_product, global, local);

    if(depth > 1)
    {
      size_t current_arg = 0;
      driver::Kernel reduce(program, reduce_name.c_str());
      driver::NDRange local(local_size_0, local_size_1);
      driver::NDRange global(align(M, local_size_0), align(N, local_size_1));
      reduce.setSizeArg(current_arg++, M);
      reduce.setSizeArg(current_arg++, N);
      reduce.setSizeArg(current_arg++, depth);
      reduce.setArg(current_arg++, workspace);
      reduce.setSizeArg(current_arg++, M);
      if(backend==driver::OPENCL)
        reduce.setArg(current_arg++, C.array.handle.cl);
      else
        reduce.setArg(current_arg++, C.array.handle.cu);
      reduce.setSizeArg(current_arg++, C.ld[1]);
      reduce.setSizeArg(current_arg++, C.array.start);
      reduce.setSizeArg(current_arg++, C.ld[0]);
      reduce.setArg(current_arg++, beta);
      options.enqueue(queue.context(), reduce, global, local);
    }

  }

  std::vector<int_t> matrix_product::infos(expression const & tree, symbolic::preset::matrix_product::args& arguments) const
  {
    expression::data_type const & array = tree.data();
    size_t root = tree.root();
    arguments = symbolic::preset::matrix_product::check(array, root);
    int_t M = arguments.C->shape[0];
    int_t N = arguments.C->shape[1];
    int_t K = (A_trans_=='T')?arguments.A->shape[0]:arguments.A->shape[1];
    return {M, N, K};
  }

  matrix_product::matrix_product(size_t s
                                 ,size_t ls0, size_t KL, size_t ls1, size_t D
                                 ,size_t ms, size_t ks, size_t ns
                                 ,fetching_policy_type Afetch, fetching_policy_type Bfetch
                                 ,size_t fetch0, size_t fetch1
                                 ,char A_trans, char B_trans) : base(s, ls0, ls1),
    kL(KL), depth(D), mS(ms), kS(ks), nS(ns), A_fetching_policy(Afetch), B_fetching_policy(Bfetch),
    local_fetch_0(fetch0), local_fetch_1(fetch1),
    mL(ms*local_size_0), nL(ns*local_size_1), A_trans_(A_trans), B_trans_(B_trans)
  {
    if(A_trans_=='N' && B_trans_=='N') type_ = MATRIX_PRODUCT_NN;
    else if(A_trans_=='T' && B_trans_=='N') type_ = MATRIX_PRODUCT_TN;
    else if(A_trans_=='N' && B_trans_=='T') type_ = MATRIX_PRODUCT_NT;
    else if(A_trans_=='T' && B_trans_=='T') type_ = MATRIX_PRODUCT_TT;
    else throw;
  }

  std::vector<int_t> matrix_product::input_sizes(expression const & expressions) const
  {
    symbolic::preset::matrix_product::args dummy;
    return infos((expression&)expressions, dummy);
  }

  void matrix_product::enqueue(driver::CommandQueue & queue, driver::Program const & program, std::string const & suffix,
                               expression const & tree, runtime::environment const & options)
  {
    symbolic::preset::matrix_product::args args;
    std::vector<int_t> MNK = infos(tree, args);
    int_t M = MNK[0];
    int_t N = MNK[1];
    int_t K = MNK[2];
    if(M==0 || N == 0 || K ==0)
      return;
    enqueue_block(queue,  M, N, K, *args.A, *args.B, *args.C, args.alpha, args.beta, program, suffix, options);
  }

  //
  matrix_product_nn::matrix_product_nn(size_t simd
                           , size_t ls0, size_t KL, size_t ls1, size_t D
                           , size_t ms, size_t ks, size_t ns
                           , fetching_policy_type Afetch , fetching_policy_type Bfetch
                           , size_t lfetch0, size_t lfetch1) :
    matrix_product(simd, ls0, KL, ls1, D, ms, ks, ns, Afetch, Bfetch, lfetch0, lfetch1, 'N', 'N')
  {
  }

  //
  matrix_product_tn::matrix_product_tn(size_t simd
                           , size_t ls0, size_t KL, size_t ls1, size_t D
                           , size_t ms, size_t ks, size_t ns
                           , fetching_policy_type Afetch , fetching_policy_type Bfetch
                           , size_t lfetch0, size_t lfetch1) :
    matrix_product(simd, ls0, KL, ls1, D, ms, ks, ns, Afetch, Bfetch, lfetch0, lfetch1, 'T', 'N')
  { }

  //
  matrix_product_nt::matrix_product_nt(size_t simd
                           , size_t ls0, size_t KL, size_t ls1, size_t D
                           , size_t ms, size_t ks, size_t ns
                           , fetching_policy_type Afetch , fetching_policy_type Bfetch
                           , size_t lfetch0, size_t lfetch1) :
    matrix_product(simd, ls0, KL, ls1, D, ms, ks, ns, Afetch, Bfetch, lfetch0, lfetch1, 'N', 'T')
  { }

  //
  matrix_product_tt::matrix_product_tt(size_t simd
                           , size_t ls0, size_t KL, size_t ls1, size_t D
                           , size_t ms, size_t ks, size_t ns
                           , fetching_policy_type Afetch , fetching_policy_type Bfetch
                           , size_t lfetch0, size_t lfetch1):
    matrix_product(simd, ls0, KL, ls1, D, ms, ks, ns, Afetch, Bfetch, lfetch0, lfetch1, 'T', 'T')
  { }

}
}
