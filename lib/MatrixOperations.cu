#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <utils/cublas_utils.h>
#include <utils/cuda_utils.h>

#include <MatrixOperations.cuh>
#include <iostream>
#include <vector>

template <typename data_type>
MatrixOperations<data_type>::MatrixOperations() {
  CUBLAS_CHECK(cublasCreate(&handle_));
}

template <typename data_type>
MatrixOperations<data_type>::~MatrixOperations() {
  if (handle_) {
    cublasDestroy(handle_);
  }
}

template <typename data_type>
MatrixOperations<data_type>::MatrixOperations(
    MatrixOperations&& other) noexcept {
  handle_ = other.handle_;
  other.handle_ = nullptr;
}

template <typename data_type>
MatrixOperations<data_type>& MatrixOperations<data_type>::operator=(
    MatrixOperations&& other) noexcept {
  if (this != &other) {
    if (handle_) {
      cublasDestroy(handle_);
    }
    handle_ = other.handle_;
    other.handle_ = nullptr;
  }
  return *this;
}

// Matrix multiplication wrapper: C = A * B
// A: m x k, B: k x n, C: m x n
// Generic template declaration (not defined)

// Specialization for float
template <>
void MatrixOperations<float>::matrixMultiplication(const float* A,
                                                   const float* B, float* C,
                                                   int m, int n, int k,
                                                   float alpha, float beta) {
  // cuBLAS uses column-major storage by default
  CUBLAS_CHECK(cublasSgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha,
                           A, m, B, k, &beta, C, m));
}

template <>
void MatrixOperations<double>::matrixMultiplication(const double* A,
                                                    const double* B, double* C,
                                                    int m, int n, int k,
                                                    double alpha, double beta) {
  // cuBLAS uses column-major storage by default
  CUBLAS_CHECK(cublasDgemm(handle_, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha,
                           A, m, B, k, &beta, C, m));
}

// Explicit template instantiation
template class MatrixOperations<float>;
template class MatrixOperations<double>;