#ifndef _ISAAC_RUNTIME_EXCEPTIONS_H
#define _ISAAC_RUNTIME_EXCEPTIONS_H

#include <string>
#include <exception>
#include "isaac/common.h"

namespace isaac
{
namespace runtime
{

DISABLE_MSVC_WARNING_C4275
class runtime_error : public std::exception
{
public:
  runtime_error(std::string const & message);
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
