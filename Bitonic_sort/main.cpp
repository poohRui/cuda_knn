//
//  main.cpp
//  Bitonic Sort
//
//  This program is going to show the Bitonic Sort alogriam and how to parallel it.
//
//  Created by poohRui on 2018/11/01.
//  Copyright © 2018 poohRui. All rights reserved.
//

#include <iostream>
#include <cmath>
#include <stdlib.h>
#include <time.h>

using namespace std;

/**
 * To randomly initial value of array
 *
 * @param array    m * dim matrix, keep the data in one of the matrix
 * @param m        the size of the array
 */
void initialVector(int*  arr,
                   int   len){
    
    srand(time(NULL));
    
    for(int i = 0;i < len;i++){
        arr[i] = rand() % 100;
    }
}

// 对双调序列进行排序
void BitonicSort(int* arr,
                 int  start,
                 int  end,
                 int  step,
                 bool ascend){
    
    while(step != 0){
        for(int i = start;i<=end;i += 2*step){
            for(int j = 0;j < step;j++){
                if(ascend){
                    if(arr[i + j] > arr[i + j + step]){
                        swap(arr[i + j], arr[i + j + step]);
                    }
                }
                else{
                    if(arr[i + j] < arr[i + j + step]){
                        swap(arr[i + j], arr[i + j + step]);
                    }
                }
            }
        }
        step /= 2;
    }
}

void Bitonic(int* arr,
             int  len){
    // i = cBS, step = cd
    for(int i = 2;i <=len;i*= 2){
        int step = i/2;
        for(int j = 0;j < len;j+= i*2){
            BitonicSort(arr, j, j+i-1, step, true);
            // 要去除最后一个降序排列的数组
            if(j + i + step < len){
                BitonicSort(arr, j+i, j+i+step, step, false);
            }
        }
    }
}


int main(){
    
    int len = 16;
    int* arr = new int[len];
    
    //    int arr[16] = {86, 90, 82, 70, 64, 95, 56, 8, 74, 97, 52, 19, 64, 19, 95, 2};
    initialVector(arr, len);
    cout<<"The initial vector is:"<<endl;
    for(int i = 0; i< len;i++){
        cout<<arr[i]<<" ";
    }
    cout<<endl;
    
    //    BitonicSort(arr, 0, 7,4, true);
    //    BitonicSort(arr, 8, 15,4, false);
    Bitonic(arr, len);
    
    cout<<"After using Bitonic Sort, the vector is:"<<endl;
    for(int i = 0; i< len;i++){
        cout<<arr[i]<<" ";
    }
    cout<<endl;
}

