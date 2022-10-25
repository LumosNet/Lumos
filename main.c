#include "active_gpu.h"
#include "active.h"

int main()
{
    float x[10] = {-2, -1, 0, 1, 2, 3, 4, 5, 6, 7};
    float *data;
    cudaMalloc((void**)&data, 10*sizeof(float));
    cudaMemcpy(data, x, 10*sizeof(float), cudaMemcpyHostToDevice);
    ActivateGpu a = load_activate_gpu(LOGISTIC);
    activate_list_gpu(data, 10, a);
    cudaMemcpy(x, data, 10*sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 10; ++i){
        printf("%f ", x[i]);
    }
    printf("\n");
    cudaFree(data);
    return 0;
}
