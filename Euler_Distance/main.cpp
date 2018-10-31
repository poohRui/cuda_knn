//
//  main.cpp
//  Euler Distance
//
//  This program is going to test and optimize Euler Distance computation in knn algorithm
//
//  Created by poohRui on 2018/10/27.
//  Copyright © 2018 poohRui. All rights reserved.
//

#include <iostream>
#include <fstream>
#include <cmath>
#include <stdlib.h>
#include <time.h>

#include "EulerDist.h"

using namespace std;

/**
 *
 * @param A    m * dim matrix, keep the data in one of the matrix
 * @param B    n * dim matrix, keep the data in one of the matrix
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
    
    for(int i = 0;i<m;i++){
        for(int j = 0;j < dim;j++){
            A[i * dim + j] = rand() % 500;
        }
    }
    
    for(int i = 0;i < n;i++){
        for(int j = 0;j < dim;j++){
            B[i * dim + j] = rand() % 500;
        }
    }
}

void reorderVector(float* M,
                   float* n_M,
                   int    m,
                   int    dim){
    
    
    for(int i = 0;i < dim;i++){
        for(int j = 0;j < m;j++){
            n_M[i * m + j] = M[j * dim + i];
        }
    }
    
}

void EulerDistanceSerial(float*   A,
                         float*   B,
                         float*   C,
                         int      m,
                         int      n,
                         int      dim){
    
    for(int i = 0;i < m;i++){
        for(int j = 0;j < n;j++){
            float dist = 0.0;
            for(int k = 0;k < dim;k++){
                float diff = A[i * dim + k] - B[j * dim + k];
                dist += diff * diff;
            }
            C[i * n + j] = sqrtf(dist);
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
            C[i * n + j] = -2.0* val;
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
    float* p_A = new float[m * dim];
    float* p_B = new float[n * dim];
    float* p_C = new float[m * n];
    
    initialVector(A, B, m, n, dim);
    
    // Start timer
    clock_t start, end;
    start = clock();

    // Compute Euler Distance,
    EulerDistanceSerial(A, B, C, m, n, dim);
    
    // Stop timer
    end = clock();
    double dur = (double)(end - start);
    
    cout<<"Serial invoke EulerDistance function need "<<dur/CLOCKS_PER_SEC<<"s."<<endl;
    
//    for(int i = 0; i< 100;i++){
//        cout<<C[9000 * n +i]<<endl;
//    }
    
    reorderVector(A, p_A, m, dim);
    reorderVector(B, p_B, n, dim);
    
    // Call a kernel first to avoid including the time of starting cuda environment.
    Cuda_EulerDistance_4(p_A, p_B, p_C, m, n, dim);
    
    // Start timer
    start = clock();
    
    // Compute Euler Distance,
    Cuda_EulerDistance_1(p_A, p_B, p_C, m, n, dim);
    
    // Stop timer
    end = clock();
    dur = (double)(end - start);

    cout<<"Parallel invoke EulerDistance1 function need "<<dur/CLOCKS_PER_SEC<<"s."<<endl;
    
    // Start timer
    start = clock();
    
    // Compute Euler Distance,
    Cuda_EulerDistance_2(p_A, p_B, p_C, m, n, dim);
    
    // Stop timer
    end = clock();
    dur = (double)(end - start);
    
    cout<<"Parallel invoke EulerDistance2 function need "<<dur/CLOCKS_PER_SEC<<"s."<<endl;
    
    // Start timer
    start = clock();
    
    // Compute Euler Distance,
    Cuda_EulerDistance_3(p_A, p_B, p_C, m, n, dim);
    
    // Stop timer
    end = clock();
    dur = (double)(end - start);
    
    cout<<"Parallel invoke EulerDistance3 function need "<<dur/CLOCKS_PER_SEC<<"s."<<endl;
    
    // Start timer
    start = clock();
    
    // Compute Euler Distance,
    Cuda_EulerDistance_4(p_A, p_B, p_C, m, n, dim);
    
    // Stop timer
    end = clock();
    dur = (double)(end - start);
    
    cout<<"Parallel invoke EulerDistance4 function need "<<dur/CLOCKS_PER_SEC<<"s."<<endl;
    
//    for(int i = 0;i< 100;i++){
//        cout<<p_C[9000 * n +i]<<endl;
//    }
//
    
//
//    gemmSerial(A, p_B, p_C, m, n, dim);
//
//    for(int i = 0;i<100;i++){
//        cout<<p_C[9000 * n +i]<<endl;
//    }
    
    // Output the result inorder to compare with parallel
//    ofstream fs("dist.dat");
//    for(int i = 0;i < m;i++){
//        for(int j = 0;j < n;j++){
//            fs << C[i * n + j]<<" ";
//        }
//        fs <<"\n";
//    }
//    fs.close();
//
}
