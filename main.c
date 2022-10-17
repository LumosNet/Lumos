#include "cpu_gpu.h"

int main()
{
    float x[10];
    float *data;
    cudaMalloc((void**)&data, 10*sizeof(float));
    fill_gpu(data, 10, 1, 1);
    cudaMemcpy(x, data, 10*sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 10; ++i){
        printf("%f ", x[i]);
    }
    printf("\n");
    return 0;
}
