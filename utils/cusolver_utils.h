#pragma once

#include <cuComplex.h>
#include <cublas_api.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>
#include <library_types.h>

#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>

// cusolver API error checking
#define CUSOLVER_CHECK(err)                                             \
  do {                                                                  \
    cusolverStatus_t err_ = (err);                                      \
    if (err_ != CUSOLVER_STATUS_SUCCESS) {                              \
      printf("cusolver error %d at %s:%d\n", err_, __FILE__, __LINE__); \
      throw std::runtime_error("cusolver error");                       \
    }                                                                   \
  } while (0)
