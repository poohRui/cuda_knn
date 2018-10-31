#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include "cublas_v2.h"

# define TILE_WIDTH 32

// 每一个线程负责计算一列A和一列B向量之间的欧拉距离
/**
 * This kernel use each thread in charge of the calculation between a column of A and a column of B
 *
 *@param A   a matrix shape as dim*m
 *@param B   a matrix shape as dim*n
 *@param C   a matrix shape as m*n
 *@param m   the column number of A
 *@param n   the column number of B
 *@param dim the row number of both A and B
 */
__global__
void EulerDistance_1(float* A,
                     float* B,
                     float* C,
                     int    m,
                     int    n,
                     int    dim){
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;
    
    if(row < m && col < n){
        float dist = 0;
        for(int i = 0; i < dim;i++){
            float diff = A[i * m + row] - B[i * n + col];
            dist += diff * diff;
        }
        C[row * n + col] = sqrtf(dist);
    }
}

/**
 * This kernel use the shared memory reduce the times of visiting global memory
 *
 *@param A   a matrix shape as dim*m
 *@param B   a matrix shape as dim*n
 *@param C   a matrix shape as m*n
 *@param m   the column number of A
 *@param n   the column number of B
 *@param dim the row number of both A and B
 */
__global__
void EulerDistance_3(float* A,
                     float* B,
                     float* C,
                     int    m,
                     int    n,
                     int    dim){
    
    __shared__ float Ads[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bds[TILE_WIDTH][TILE_WIDTH];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;
    
    int phase_num = ceil(dim / (float)TILE_WIDTH);
    
    int cond0 = row < m;
    int cond1 = col < n;
    
    float dist = 0.0;
    // 负责加载A的第row列和B的第col列
    for (int ph = 0; ph < phase_num;ph++) {
        if(ph * TILE_WIDTH + ty < dim){
            Ads[tx][ty] = (cond0) ? A[(ph * TILE_WIDTH + tx) * m + row] : 0;
            Bds[ty][tx] = (cond1) ? B[(ph * TILE_WIDTH + ty) * n + col] : 0;
        }
        else{
            Ads[tx][ty] = 0;
            Bds[ty][tx] = 0;
        }
        
        __syncthreads();
        
        if(cond1 &&cond0){
            for(int i = 0;i < TILE_WIDTH;i++){
                float diff = Ads[i][ty] - Bds[i][tx];
                dist += diff * diff;
            }
        }
        
        __syncthreads();
    }
    
    if(cond1 && cond0){
        C[row * n + col] = sqrtf(dist);
    }
}

/**
 * This kernel use to calculate the diag of matrix * matrix
 *
 *@param array   a matrix shape as dim*m
 *@param row     the number of row
 *@param col     the number of column
 *@param norm    return the diag of matrix * matrix and put it into a one dimension matrix
 */
__global__
void compute_squared_norm(float* array,
                          int    row,
                          int    col,
                          float* norm){
    
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if(index < col){
        float sum = 0.0;
        for(int i = 0;i < row;i++){
            float val = array[i * col + index];
            sum += val * val;
        }
        norm[index] = sum;
    }
}

/**
 * This kernel use to broadcast the diag value to all the rows/columns
 *
 *@param array   a matrix shape as dim*m
 *@param row     the number of row
 *@param col     the number of column
 *@param norm    return the diag of matrix * matrix and put it into a one dimension matrix
 */
__global__
void broadcast_points(float* A_norm,
                      float* B_norm,
                      int    m,
                      int    n,
                      float* C){
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;
    
    __shared__ float Ads[TILE_WIDTH];
    __shared__ float Bds[TILE_WIDTH];
    
    // Load values into shared memory
    if(tx == 0 && col < n){
        Ads[ty] = A_norm[row];
    }
    if(ty == 0 && row < m){
        Bds[tx] = B_norm[col];
    }
    __syncthreads();
    
    // C中的每个元素i,j，都需要加A_norm[i]和B_norm[j]
    if(row < m && col < n){
        float val = C[row * n + col] + Ads[ty] + Bds[tx];
        C[row * n + col] = sqrtf(val);
    }
}

/**
 * This is the stub function of EulerDistance_1
 *
 *@param A   a matrix shape as dim*m
 *@param B   a matrix shape as dim*n
 *@param C   a matrix shape as m*n
 *@param m   the column number of A
 *@param n   the column number of B
 *@param dim the row number of both A and B
 */
void Cuda_EulerDistance_1(float* A,
                          float* B,
                          float* C,
                          int    m,
                          int    n,
                          int    dim){
    
    dim3 dimGrid(ceil(n/32.0), ceil(m/32.0), 1);
    dim3 dimBlock(32, 32, 1);
    
    float* d_A;
    float* d_B;
    float* d_C;
    int size_A = m * dim * sizeof(float);
    int size_B = n * dim * sizeof(float);
    int size_C = m * n * sizeof(float);
    
    cudaMalloc((void**)&d_A, size_A);
    cudaMalloc((void**)&d_B, size_B);
    cudaMalloc((void**)&d_C, size_C);
    
    cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice);
    
    EulerDistance_1<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, m, n, dim);
    
    cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
}

/**
 * This is the stub function of EulerDistance_2
 *
 *@param A   a matrix shape as dim*m
 *@param B   a matrix shape as dim*n
 *@param C   a matrix shape as m*n
 *@param m   the column number of A
 *@param n   the column number of B
 *@param dim the row number of both A and B
 */
void Cuda_EulerDistance_2(float* A,
                          float* B,
                          float* C,
                          int    m,
                          int    n,
                          int    dim){
    
    dim3 dimGrid(ceil(n/32.0), ceil(m/32.0), 1);
    dim3 dimBlock(32, 32, 1);
    
    float* d_A;
    float* d_B;
    float* d_C;
    int size_A = m * dim * sizeof(float);
    int size_B = n * dim * sizeof(float);
    int size_C = m * n * sizeof(float);
    
    cudaMalloc((void**)&d_A, size_A);
    cudaMalloc((void**)&d_B, size_B);
    cudaMalloc((void**)&d_C, size_C);
    
    cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice);
    
    EulerDistance_3<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, m, n, dim);
    
    cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

/**
 * This is the stub function of EulerDistance_3
 *
 *@param A   a matrix shape as dim*m
 *@param B   a matrix shape as dim*n
 *@param C   a matrix shape as m*n
 *@param m   the column number of A
 *@param n   the column number of B
 *@param dim the row number of both A and B
 */
void Cuda_EulerDistance_3(float* A,
                          float* B,
                          float* C,
                          int    m,
                          int    n,
                          int    dim){
    
    dim3 dimGrid(ceil(n/32.0), ceil(m/32.0), 1);
    dim3 dimBlock(32, 32, 1);
    
    float* d_A;
    float* d_B;
    float* d_C;
    
    size_t size_of_float = sizeof(float);
    size_t size_patch_bytes_A;
    size_t size_patch_bytes_B;
    size_t size_patch_bytes_C;
    
    cudaMallocPitch((void**)&d_A, &size_patch_bytes_A, m * size_of_float, dim);
    cudaMallocPitch((void**)&d_B, &size_patch_bytes_B, n * size_of_float, dim);
    cudaMallocPitch((void**)&d_C, &size_patch_bytes_C, n * size_of_float, m);
    
    cudaMemcpy2D(d_A, size_patch_bytes_A, A, m * size_of_float, m * size_of_float, dim, cudaMemcpyHostToDevice);
    cudaMemcpy2D(d_B, size_patch_bytes_B, B, n * size_of_float, n * size_of_float, dim, cudaMemcpyHostToDevice);
    
    EulerDistance_3<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, m, n, dim);
    
    cudaMemcpy2D(C, size_patch_bytes_C, d_C, n * size_of_float, n * size_of_float, m, cudaMemcpyDeviceToHost);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

/**
 * This is the stub function of EulerDistance_4
 *
 *@param A   a matrix shape as dim*m
 *@param B   a matrix shape as dim*n
 *@param C   a matrix shape as m*n
 *@param m   the column number of A
 *@param n   the column number of B
 *@param dim the row number of both A and B
 */
void Cuda_EulerDistance_4(float* A,
                          float* B,
                          float* C,
                          int    m,
                          int    n,
                          int    dim){
    
    float* d_A;
    float* d_B;
    float* d_C;
    float* d_A_norm;
    float* d_B_norm;
    
    size_t size_of_float = sizeof(float);
    size_t size_patch_bytes_A;
    size_t size_patch_bytes_B;
    size_t size_patch_bytes_C;
    
    cudaMallocPitch((void**)&d_A, &size_patch_bytes_A, m * size_of_float, dim);
    cudaMallocPitch((void**)&d_B, &size_patch_bytes_B, n * size_of_float, dim);
    cudaMallocPitch((void**)&d_C, &size_patch_bytes_C, n * size_of_float, m);
    
    cudaMemcpy2D(d_A, size_patch_bytes_A, A, m * size_of_float, m * size_of_float, dim, cudaMemcpyHostToDevice);
    cudaMemcpy2D(d_B, size_patch_bytes_B, B, n * size_of_float, n * size_of_float, dim, cudaMemcpyHostToDevice);
    
    // Convert Euler Distance calculation to gemm
    cublasHandle_t cublasHandle;
    cublasCreate(&cublasHandle);
    
    // A(dim*m), B(dim*n), C(m*n)
    float alpha = -2.0;
    float beta = 0.0;
    cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, n, m, dim, &alpha, d_B, n, d_A, m, &beta, d_C, n);
    
    // broadcasting to add A*A and B*B
    cudaMalloc((void**)&d_A_norm, m * size_of_float);
    cudaMalloc((void**)&d_B_norm, n * size_of_float);
    
    dim3 dimGrid0(ceil(m/1024.0), 1, 1);
    dim3 dimGrid1(ceil(n/1024.0), 1, 1);
    dim3 dimBlock0(1024, 1, 1);
    
    compute_squared_norm<<<dimGrid0, dimBlock0>>>(d_A, dim, m, d_A_norm);
    compute_squared_norm<<<dimGrid1, dimBlock0>>>(d_B, dim, n, d_B_norm);
    
    dim3 dimBlock(32, 32, 1);
    dim3 dimGrid(ceil(n/32.0), ceil(m/32.0), 1);
    
    broadcast_points<<<dimGrid, dimBlock>>>(d_A_norm, d_B_norm, m, n, d_C);
    
    
    cudaMemcpy2D(C, size_patch_bytes_C, d_C, n * size_of_float, n * size_of_float, m, cudaMemcpyDeviceToHost);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cublasDestroy(cublasHandle);
    
}
