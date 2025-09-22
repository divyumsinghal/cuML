#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <utils/cublas_utils.h>
#include <utils/cuda_utils.h>
#include <utils/cusolver_utils.h>

#include <iostream>
#include <vector>

template <typename data_type>
class LinearRegression {
 public:
  LinearRegression();
  ~LinearRegression();

  // Disable copying (handle is not copyable)
  LinearRegression(const LinearRegression&) = delete;
  LinearRegression& operator=(const LinearRegression&) = delete;

  // Allow moving
  LinearRegression(LinearRegression&& other) noexcept;
  LinearRegression& operator=(LinearRegression&& other) noexcept;

  void fit(std::vector<data_type> h_X, std::vector<data_type> h_y,
           std::vector<data_type> h_coefficients, int m, int n);

 private:
  cublasHandle_t cublas_handle;
  cusolverDnHandle_t cusolver_handle;
};
