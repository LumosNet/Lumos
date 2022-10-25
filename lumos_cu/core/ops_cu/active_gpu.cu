#include "active_gpu.h"

ActivateGpu load_activate_gpu(Activation TYPE)
{
    activate_gpu func;
    if (TYPE == STAIR)
        func = stair_activate_kernel;
    else if (TYPE == HARDTAN)
        func = hardtan_activate_kernel;
    else if (TYPE == LINEAR)
        func = linear_activate_kernel;
    else if (TYPE == LOGISTIC)
        func = logistic_activate_kernel;
    else if (TYPE == LOGGY)
        func = loggy_activate_kernel;
    else if (TYPE == RELU)
        func = relu_activate_kernel;
    else if (TYPE == ELU)
        func = elu_activate_kernel;
    else if (TYPE == SELU)
        func = selu_activate_kernel;
    else if (TYPE == RELIE)
        func = relie_activate_kernel;
    else if (TYPE == RAMP)
        func = ramp_activate_kernel;
    else if (TYPE == LEAKY)
        func = leaky_activate_kernel;
    else if (TYPE == TANH)
        func = tanh_activate_kernel;
    else if (TYPE == PLSE)
        func = plse_activate_kernel;
    else if (TYPE == LHTAN)
        func = lhtan_activate_kernel;
    return func;
}

GradientGpu load_gradient_gpu(Activation TYPE)
{
    gradient_gpu func;
    if (TYPE == STAIR)
        func = stair_gradient_kernel;
    else if (TYPE == HARDTAN)
        func = hardtan_gradient_kernel;
    else if (TYPE == LINEAR)
        func = linear_gradient_kernel;
    else if (TYPE == LOGISTIC)
        func = logistic_gradient_kernel;
    else if (TYPE == LOGGY)
        func = loggy_gradient_kernel;
    else if (TYPE == RELU)
        func = relu_gradient_kernel;
    else if (TYPE == ELU)
        func = elu_gradient_kernel;
    else if (TYPE == SELU)
        func = selu_gradient_kernel;
    else if (TYPE == RELIE)
        func = relie_gradient_kernel;
    else if (TYPE == RAMP)
        func = ramp_gradient_kernel;
    else if (TYPE == LEAKY)
        func = leaky_gradient_kernel;
    else if (TYPE == TANH)
        func = tanh_gradient_kernel;
    else if (TYPE == PLSE)
        func = plse_gradient_kernel;
    else if (TYPE == LHTAN)
        func = lhtan_gradient_kernel;
    return func;
}

__global__ activate_list_kernel(float *origin, int num, ActivateGpu a)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= num) return;
    origin[index] = a(origin[index]);
}

void activate_list_gpu(float *origin, int num, ActivateGpu a)
{
    activate_list_kernel<<<(num+BLOCK-1)/BLOCK, BLOCK>>>(origin, num, a);
}

__global__ gradient_list_kernel(float *origin, int num, GradientGpu g)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= num) return;
    origin[index] = g(origin[index]);
}

void gradient_list_gpu(float *origin, int num, GradientGpu g)
{
    gradient_list_kernel<<<(num+BLOCK-1)/BLOCK, BLOCK>>>(origin, num, g);
}
