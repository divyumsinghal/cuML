#include <MatrixOperations.cuh>
#include <vector>
#include <iostream>
#include <cuda_runtime.h>

int main() {
    int m = 4, k = 4, n = 4;
    
    /*
     *   A = |  1.0 |  5.0 |  9.0 | 13.0 |
     *       |  2.0 |  6.0 | 10.0 | 14.0 |
     *       |  3.0 |  7.0 | 11.0 | 15.0 |
     *       |  4.0 |  8.0 | 12.0 | 16.0 |
     *
     *   B = |  1.0 |  2.0 |  3.0 |  4.0 |
     *       |  5.0 |  6.0 |  7.0 |  8.0 |
     *       |  9.0 | 10.0 | 11.0 | 12.0 |
     *       | 13.0 | 14.0 | 15.0 | 16.0 |
     */

    const std::vector<float> A = { 1.0f,  2.0f,  3.0f,  4.0f,
                                   5.0f,  6.0f,  7.0f,  8.0f,
                                   9.0f, 10.0f, 11.0f, 12.0f,
                                  13.0f, 14.0f, 15.0f, 16.0f}; 
    const std::vector<float> B = { 1.0f,  5.0f,  9.0f, 13.0f,
                                   2.0f,  6.0f, 10.0f, 14.0f,
                                   3.0f,  7.0f, 11.0f, 15.0f,
                                   4.0f,  8.0f, 12.0f, 16.0f}; 

    /*
     *   C = | 276.0 | 304.0 | 332.0 | 360.0 |
     *       | 304.0 | 336.0 | 368.0 | 400.0 |
     *       | 332.0 | 368.0 | 404.0 | 440.0 |
     *       | 360.0 | 400.0 | 440.0 | 480.0 |
     */
    std::vector<float> C(m * n, 0.0f);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, m * k * sizeof(float));
    cudaMalloc(&d_B, k * n * sizeof(float));
    cudaMalloc(&d_C, m * n * sizeof(float));
    
    // Copy data to device
    cudaMemcpy(d_A, A.data(), m * k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), k * n * sizeof(float), cudaMemcpyHostToDevice);
    
    // Perform matrix multiplication
    MatrixOperations<float> matOps;
    matOps.matrixMultiplication(d_A, d_B, d_C, m, n, k);

    // Copy result back to host
    cudaMemcpy(C.data(), d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost);

    // Print Calculated output
    std::cout << std::endl << "Calculated C:" << std::endl;
    std::cout << "C = A * B:" << std::endl;
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < n; col++) {
            std::cout << C[row + col * m] << " ";
        }
        std::cout << "" << std::endl;
    }

    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return 0;
}
