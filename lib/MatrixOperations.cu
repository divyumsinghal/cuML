#include <iostream>
#include <vector>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <utils/cuda_utils.h>
#include <utils/cublas_utils.h>

#include <MatrixOperations.cuh>

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
MatrixOperations<data_type>::MatrixOperations(MatrixOperations&& other) noexcept {
    handle_ = other.handle_;
    other.handle_ = nullptr;
}

template <typename data_type>
MatrixOperations<data_type>& MatrixOperations<data_type>::operator=(MatrixOperations&& other) noexcept {
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
template <typename data_type>
void MatrixOperations<data_type>::matrixMultiplication(const data_type* A, const data_type* B, data_type* C,
                           int m, int n, int k,
                           data_type alpha, data_type beta) {
    // cuBLAS uses column-major storage by default
    CUBLAS_CHECK(
        cublasSgemm(handle_,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    m, n, k,
                    &alpha,
                    A, m,
                    B, k,
                    &beta,
                    C, m));
}

// Explicit template instantiation for float
template class MatrixOperations<float>;
