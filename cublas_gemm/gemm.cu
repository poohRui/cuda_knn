#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include "cublas_v2.h"

/**
 * This is a stub function which call gemm to calculate "matrix * matrix" problem
 *
 *@param A    an input matrix with size m*dim
 *@param B    an input matrix with size dim*n
 *@param C    an output matrix with size m*n
 *@param m    size of the first dimension in A
 *@param n    size of the second dimension in B
 *@param dim  size of the second dimension in A
 */
void gemm_cuda(float* A,
               float* B,
               float* C,
               int    m,
               int    n,
               int    dim){
    
    float* d_A;
    float* d_B;
    float* d_C;
    
    size_t size_of_float = sizeof(float);
    size_t size_pitch_bytes_A;
    size_t size_pitch_bytes_B;
    size_t size_pitch_bytes_C;
    
    // A(m*dim), B(dim*n), C(m*n)
    // Allocate space and transfer data to device
    cudaMallocPitch((void**)&d_A, &size_pitch_bytes_A, dim * size_of_float, m);
    cudaMallocPitch((void**)&d_B, &size_pitch_bytes_B, n * size_of_float, dim);
    cudaMallocPitch((void**)&d_C, &size_pitch_bytes_C, n * size_of_float, m);
    
    cudaMemcpy2D(d_A, size_pitch_bytes_A, A, dim * size_of_float, dim * size_of_float, m, cudaMemcpyHostToDevice);
    cudaMemcpy2D(d_B, size_pitch_bytes_B, B, n * size_of_float, n * size_of_float, dim, cudaMemcpyHostToDevice);
    
    // Call gemm function in cublas
    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle);
    
    // Pay attention to the parameters
    float alpha = 1.0;
    float beta = 0.0;
    cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, dim, &alpha, d_B, n, d_A, dim, &beta, d_C, n);
    
    // Transfer result back
    cudaMemcpy2D(C, n * size_of_float, d_C, size_pitch_bytes_C, n * size_of_float, m, cudaMemcpyDeviceToHost);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    cublasDestroy(cublasHandle);
}
