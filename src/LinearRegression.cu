#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <utils/cublas_utils.h>
#include <utils/cuda_utils.h>
#include <utils/cusolver_utils.h>

#include <include/LinearRegression.cuh>
#include <iostream>
#include <vector>

template <typename data_type>
LinearRegression<data_type>::LinearRegression() {
  CUBLAS_CHECK(cublasCreate(&cublas_handle));
  CUSOLVER_CHECK(cusolverDnCreate(&cusolver_handle));
}

template <typename data_type>
LinearRegression<data_type>::~LinearRegression() {
  if (cublas_handle) {
    cublasDestroy(cublas_handle);
  }
  if (cusolver_handle) {
    cusolverDnDestroy(cusolver_handle);
  }
}

template <typename data_type>
LinearRegression<data_type>::LinearRegression(
    LinearRegression &&other) noexcept {
  cublas_handle = other.cublas_handle;
  cusolver_handle = other.cusolver_handle;
  other.cublas_handle = nullptr;
  other.cusolver_handle = nullptr;
}

template <typename data_type>
LinearRegression<data_type> &LinearRegression<data_type>::operator=(
    LinearRegression &&other) noexcept {
  if (this != &other) {
    if (cublas_handle) {
      cublasDestroy(cublas_handle);
    }
    if (cusolver_handle) {
      cusolverDnDestroy(cusolver_handle);
    }
    cublas_handle = other.cublas_handle;
    cusolver_handle = other.cusolver_handle;
    other.cublas_handle = nullptr;
    other.cusolver_handle = nullptr;
  }
  return *this;
}

template <typename data_type>
void LinearRegression<data_type>::fit(std::vector<data_type> h_X,
                                      std::vector<data_type> h_y,
                                      std::vector<data_type> h_coefficients,
                                      int m, int n) {
  data_type *d_X, *d_y, *d_XTX, *d_XTy, *d_work;
  int *d_info;
  int lwork;

  // Allocate device memory
  cudaMalloc(&d_X, m * n * sizeof(data_type));
  cudaMalloc(&d_y, m * sizeof(data_type));
  cudaMalloc(&d_XTX, n * n * sizeof(data_type));
  cudaMalloc(&d_XTy, n * sizeof(data_type));
  cudaMalloc(&d_info, sizeof(int));

  // Query workspace size for Cholesky decomposition
  cusolverDnSpotrf_bufferSize(cusolver_handle, CUBLAS_FILL_MODE_LOWER, n, d_XTX,
                              n, &lwork);

  cudaMalloc(&d_work, lwork * sizeof(data_type));

  // Copy data to device
  cudaMemcpy(d_X, h_X.data(), m * n * sizeof(data_type),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, h_y.data(), m * sizeof(data_type), cudaMemcpyHostToDevice);

  const data_type alpha = 1.0f, beta = 0.0f;

  // 1. Compute X^T * X using cublasSgemm
  cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, n, n, m, &alpha, d_X, m,
              d_X, m, &beta, d_XTX, n);

  // 2. Compute X^T * y using cublasSgemv
  cublasSgemv(cublas_handle, CUBLAS_OP_T, m, n, &alpha, d_X, m, d_y, 1, &beta,
              d_XTy, 1);

  // 3. Solve (X^T * X) * coefficients = X^T * y using Cholesky decomposition
  cusolverDnSpotrf(cusolver_handle, CUBLAS_FILL_MODE_LOWER, n, d_XTX, n, d_work,
                   lwork, d_info);

  cusolverDnSpotrs(cusolver_handle, CUBLAS_FILL_MODE_LOWER, n, 1, d_XTX, n,
                   d_XTy, n, d_info);

  // Copy results back
  cudaMemcpy(h_coefficients.data(), d_XTy, n * sizeof(data_type),
             cudaMemcpyDeviceToHost);

  // Cleanup
  cudaFree(d_X);
  cudaFree(d_y);
  cudaFree(d_XTX);
  cudaFree(d_XTy);
  cudaFree(d_work);
  cudaFree(d_info);
}

// Explicit template instantiation
template class LinearRegression<float>;