#include <stdio.h>
#include <math.h>
#include <cuda.h>

# define TILE_WIDTH 32

// 每一个线程负责计算一行A和一行B向量之间的欧拉距离
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

__global__
void EulerDistance_2(float* A,
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
