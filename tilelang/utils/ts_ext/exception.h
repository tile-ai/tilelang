#pragma once

#include <cuda_runtime.h>
#include <exception>
#include <string>

class TSException : public std::exception {
  std::string message;

public:
  TSException(const char *name, const char *file, int line,
              const std::string &error) {
    message = std::string("Failed: ") + name + " error " + file + ":" +
              std::to_string(line) + " '" + error + "'";
  }
  const char *what() const noexcept override { return message.c_str(); }
};

#ifndef TS_HOST_ASSERT
#define TS_HOST_ASSERT(cond)                                                   \
  do {                                                                         \
    if (!(cond)) {                                                             \
      throw TSException("Assertion", __FILE__, __LINE__, #cond);               \
    }                                                                          \
  } while (0)
#endif

#ifndef CUDA_CHECK
#define CUDA_CHECK(cmd)                                                        \
  do {                                                                         \
    cudaError_t e = (cmd);                                                     \
    if (e != cudaSuccess) {                                                    \
      throw TSException("CUDA", __FILE__, __LINE__, cudaGetErrorString(e));    \
    }                                                                          \
  } while (0)
#endif

