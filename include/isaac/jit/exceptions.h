#ifndef _ISAAC_JIT_EXCEPTIONS_H
#define _ISAAC_JIT_EXCEPTIONS_H

#include <string>
#include <exception>
#include "isaac/common.h"

namespace isaac
{
namespace jit
{

/** @brief Exception for the case the generator is unable to deal with the operation */
DISABLE_MSVC_WARNING_C4275
class code_generation_error : public std::exception
{
public:
  code_generation_error(std::string message);
  virtual const char* what() const throw();
private:
DISABLE_MSVC_WARNING_C4251
  std::string message_;
RESTORE_MSVC_WARNING_C4251
};
RESTORE_MSVC_WARNING_C4275


/** @brief Exception for the case the generator is unable to deal with the operation */
DISABLE_MSVC_WARNING_C4275
class ISAACAPI semantic_error : public std::exception
{
public:
  semantic_error(std::string const & message);
  virtual const char* what() const throw();
private:
DISABLE_MSVC_WARNING_C4251
  std::string message_;
RESTORE_MSVC_WARNING_C4251
};
RESTORE_MSVC_WARNING_C4275

}
}

#endif
