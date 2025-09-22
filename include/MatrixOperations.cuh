#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <utils/cublas_utils.h>
#include <utils/cuda_utils.h>

#include <stdexcept>
#include <string>

template <typename data_type>
class MatrixOperations {
 public:
  MatrixOperations();
  ~MatrixOperations();

  // Disable copying (handle is not copyable)
  MatrixOperations(const MatrixOperations &) = delete;
  MatrixOperations &operator=(const MatrixOperations &) = delete;

  // Allow moving
  MatrixOperations(MatrixOperations &&other) noexcept;
  MatrixOperations &operator=(MatrixOperations &&other) noexcept;

  // Matrix multiplication: C = alpha * A * B + beta * C
  // A: m x k, B: k x n, C: m x n
  void matrixMultiplication(const data_type *A, const data_type *B,
                            data_type *C, int m, int n, int k,
                            data_type alpha = 1.0f, data_type beta = 0.0f);

 private:
  cublasHandle_t handle_{nullptr};
};
