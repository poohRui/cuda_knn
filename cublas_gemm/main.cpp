//
//  main.cpp
//  Euler Distance
//
//  This program is going to test and optimize Euler Distance computation in knn algorithm
//
//  Created by poohRui on 2018/10/30.
//  Copyright © 2018 poohRui. All rights reserved.
//

#include <iostream>
#include <cmath>
#include <stdlib.h>
#include <time.h>

#include "gemm.h"

using namespace std;

/**
 *
 * @param A    m * dim matrix, keep the data in one of the matrix
 * @param B    dim * n matrix, keep the data in one of the matrix
 * @param m    the data number of matrix A
 * @param n    the data number of matrix B
 * @param dim  the feature number of both A and B
 */
void initialVector(float*  A,
                   float*  B,
                   int     m,
                   int     n,
                   int     dim){
    
    srand(time(NULL));
    
    for(int i = 0;i < m;i++){
        for(int j = 0;j < dim;j++){
            A[i * dim + j] = rand() % 100;
        }
    }
    
    for(int i = 0;i < dim;i++){
        for(int j = 0;j < n;j++){
            B[i * n + j] = rand() % 100;
        }
    }
}

void gemmSerial(float*   A,
                float*   B,
                float*   C,
                int      m,
                int      n,
                int      dim){
    
    for(int i = 0;i < m;i++){
        for(int j = 0;j < n;j++){
            float val = 0.0;
            for(int k = 0;k < dim;k++){
                val += A[i * dim + k] * B[k * n + j];
            }
            C[i * n + j] = val;
        }
    }
}

int main(){
    
    int m = 16384;
    int n = 4096;
    int dim = 128;
    
    // Memory allocation for A, B, and C
    float* A = new float[m * dim];
    float* B = new float[n * dim];
    float* C = new float[m * n];
    
    initialVector(A, B, m, n, dim);
    
    // Start timer
    clock_t start, end;
    start = clock();
    
    // Compute Euler Distance,
    gemmSerial(A, B, C, m, n, dim);
    
    // Stop timer
    end = clock();
    double dur = (double)(end - start);
    
    cout<<"Serial invoke gemmSerial function need "<<dur/CLOCKS_PER_SEC<<"s."<<endl;
    
    for(int i = 0; i< 100;i++){
        cout<<C[9000 * n +i]<<endl;
    }
    
    // Call kernel to avoid the time of initial cuda
    gemm_cuda(A, B, C, m, n, dim);
    
    // Start timer
    start = clock();
    
    // Compute Euler Distance,
    gemm_cuda(A, B, C, m, n, dim);
    
    // Stop timer
    end = clock();
    dur = (double)(end - start);
    
    cout<<"Parallel invoke gemm function using cublas need "<<dur/CLOCKS_PER_SEC<<"s."<<endl;
    
    for(int i = 0; i< 100;i++){
        cout<<p_C[9000 * n +i]<<endl;
    }
}
