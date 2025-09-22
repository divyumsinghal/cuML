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
                                      std::vector<data_type> &h_coefficients,
                                      int num_samples, int num_features) {
  data_type *d_X, *d_y, *d_XTX, *d_XTy, *d_work;
  int *d_info;
  int lwork;

  // Allocate device memory
  CUDA_CHECK(cudaMalloc(&d_X, num_samples * num_features * sizeof(data_type)));
  CUDA_CHECK(cudaMalloc(&d_y, num_samples * sizeof(data_type)));
  CUDA_CHECK(
      cudaMalloc(&d_XTX, num_features * num_features * sizeof(data_type)));
  CUDA_CHECK(cudaMalloc(&d_XTy, num_features * sizeof(data_type)));
  CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

  // Query workspace size for Cholesky decomposition
  CUSOLVER_CHECK(
      cusolverDnSpotrf_bufferSize(cusolver_handle, CUBLAS_FILL_MODE_LOWER,
                                  num_features, d_XTX, num_features, &lwork));

  CUDA_CHECK(cudaMalloc(&d_work, lwork * sizeof(data_type)));

  // Copy data to device
  CUDA_CHECK(cudaMemcpy(d_X, h_X.data(),
                        num_samples * num_features * sizeof(data_type),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_y, h_y.data(), num_samples * sizeof(data_type),
                        cudaMemcpyHostToDevice));

  const data_type alpha = 1.0f, beta = 0.0f;

  // 1. Compute X^T * X using cublasSgemm
  CUBLAS_CHECK(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N,
                           num_features, num_features, num_samples, &alpha, d_X,
                           num_samples, d_X, num_samples, &beta, d_XTX,
                           num_features));

  // 2. Compute X^T * y using cublasSgemv
  CUBLAS_CHECK(cublasSgemv(cublas_handle, CUBLAS_OP_T, num_samples,
                           num_features, &alpha, d_X, num_samples, d_y, 1,
                           &beta, d_XTy, 1));

  // 3. Solve (X^T * X) * coefficients = X^T * y using Cholesky decomposition
  CUSOLVER_CHECK(cusolverDnSpotrf(cusolver_handle, CUBLAS_FILL_MODE_LOWER,
                                  num_features, d_XTX, num_features, d_work,
                                  lwork, d_info));

  CUSOLVER_CHECK(cusolverDnSpotrs(cusolver_handle, CUBLAS_FILL_MODE_LOWER,
                                  num_features, 1, d_XTX, num_features, d_XTy,
                                  num_features, d_info));

  // Copy results back
  CUDA_CHECK(cudaMemcpy(h_coefficients.data(), d_XTy,
                        num_features * sizeof(data_type),
                        cudaMemcpyDeviceToHost));

  // Cleanup
  cudaFree(d_X);
  cudaFree(d_y);
  cudaFree(d_XTX);
  cudaFree(d_XTy);
  cudaFree(d_work);
  cudaFree(d_info);
}

// Alternative implementation using QR decomposition (more stable)
template <>
void LinearRegression<float>::fit(std::vector<float> h_X,
                                  std::vector<float> h_y,
                                  std::vector<float> &h_coefficients,
                                  int num_samples, int num_features) {
  float *d_X, *d_y, *d_tau, *d_work;
  int *d_info;
  int lwork;

  // Allocate device memory
  CUDA_CHECK(cudaMalloc(&d_X, num_samples * num_features * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_y, num_samples * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_tau, num_features * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

  // Copy data to device
  CUDA_CHECK(cudaMemcpy(d_X, h_X.data(),
                        num_samples * num_features * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_y, h_y.data(), num_samples * sizeof(float),
                        cudaMemcpyHostToDevice));

  // Query workspace size for QR decomposition
  CUSOLVER_CHECK(cusolverDnSgeqrf_bufferSize(
      cusolver_handle, num_samples, num_features, d_X, num_samples, &lwork));

  CUDA_CHECK(cudaMalloc(&d_work, lwork * sizeof(float)));

  // 1. QR decomposition of X
  CUSOLVER_CHECK(cusolverDnSgeqrf(cusolver_handle, num_samples, num_features,
                                  d_X, num_samples, d_tau, d_work, lwork,
                                  d_info));

  // 2. Solve using QR decomposition
  CUSOLVER_CHECK(cusolverDnSormqr(cusolver_handle, CUBLAS_SIDE_LEFT,
                                  CUBLAS_OP_T, num_samples, 1, num_features,
                                  d_X, num_samples, d_tau, d_y, num_samples,
                                  d_work, lwork, d_info));

  // 3. Solve triangular system R * x = Q^T * y
  const float alpha = 1.0f;
  CUBLAS_CHECK(cublasStrsv(cublas_handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                           CUBLAS_DIAG_NON_UNIT, num_features, d_X, num_samples,
                           d_y, 1));

  // Copy results back (only first num_features elements)
  CUDA_CHECK(cudaMemcpy(h_coefficients.data(), d_y,
                        num_features * sizeof(float), cudaMemcpyDeviceToHost));

  // Cleanup
  cudaFree(d_X);
  cudaFree(d_y);
  cudaFree(d_tau);
  cudaFree(d_work);
  cudaFree(d_info);
}

// Explicit template instantiation
template class LinearRegression<float>;