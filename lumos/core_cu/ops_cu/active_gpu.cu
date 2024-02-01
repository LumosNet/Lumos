#include "active_gpu.h"

__device__ float stair_activate_kernel(float x)
{
    int n = floor(x);
    if (n%2 == 0) return floor(x/2.);
    else return (x - n) + floor(x/2.);
}

__device__ float hardtan_activate_kernel(float x)
{
    if (x < -1) return -1;
    if (x > 1) return 1;
    return x;
}

__device__ float linear_activate_kernel(float x){return x;}
__device__ float logistic_activate_kernel(float x){return 1./(1. + exp(-x));}
__device__ float loggy_activate_kernel(float x){return 2./(1. + exp(-x)) - 1;}
__device__ float relu_activate_kernel(float x){return x*(x>0);}
__device__ float elu_activate_kernel(float x){return (x >= 0)*x + (x < 0)*(exp(x)-1);}
__device__ float selu_activate_kernel(float x){return (x >= 0)*1.0507*x + (x < 0)*1.0507*1.6732*(exp(x)-1);}
__device__ float relie_activate_kernel(float x){return (x>0) ? x : .01*x;}
__device__ float ramp_activate_kernel(float x){return x*(x>0)+.1*x;}
__device__ float leaky_activate_kernel(float x){return (x>0) ? x : .1*x;}
__device__ float tanh_activate_kernel(float x){return (exp(2*x)-1)/(exp(2*x)+1);}

__device__ float plse_activate_kernel(float x)
{
    if(x < -4) return .01 * (x + 4);
    if(x > 4)  return .01 * (x - 4) + 1;
    return .125*x + .5;
}

__device__ float lhtan_activate_kernel(float x)
{
    if(x < 0) return .001*x;
    if(x > 1) return .001*(x-1) + 1;
    return x;
}

__device__ float lhtan_gradient_kernel(float x)
{
    if(x > 0 && x < 1) return 1;
    return .001;
}

__device__ float hardtan_gradient_kernel(float x)
{
    if (x > -1 && x < 1) return 1;
    return 0;
}

__device__ float linear_gradient_kernel(float x){return 1;}
__device__ float logistic_gradient_kernel(float x){return (1-x)*x;}

__device__ float loggy_gradient_kernel(float x)
{
    float y = (x+1.)/2.;
    return 2*(1-y)*y;
}

__device__ float stair_gradient_kernel(float x)
{
    if (floor(x) == x) return 0;
    return 1;
}

__device__ float relu_gradient_kernel(float x){return (x>0);}
__device__ float elu_gradient_kernel(float x){return (x >= 0) + (x < 0)*(x + 1);}
__device__ float selu_gradient_kernel(float x){return (x >= 0)*1.0507 + (x < 0)*(x + 1.0507*1.6732);}
__device__ float relie_gradient_kernel(float x){return (x>0) ? 1 : .01;}
__device__ float ramp_gradient_kernel(float x){return (x>0)+.1;}
__device__ float leaky_gradient_kernel(float x){return (x>0) ? 1 : .1;}
__device__ float tanh_gradient_kernel(float x){return 1-x*x;}
__device__ float plse_gradient_kernel(float x){return (x < 0 || x > 1) ? .01 : .125;}

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

__device__ float activate_x_kernel(Activation TYPE, float x)
{
    float res = 0;
    if (TYPE == STAIR)
        res = stair_activate_kernel(x);
    else if (TYPE == HARDTAN)
        res = hardtan_activate_kernel(x);
    else if (TYPE == LINEAR)
        res = linear_activate_kernel(x);
    else if (TYPE == LOGISTIC)
        res = logistic_activate_kernel(x);
    else if (TYPE == LOGGY)
        res = loggy_activate_kernel(x);
    else if (TYPE == RELU)
        res = relu_activate_kernel(x);
    else if (TYPE == ELU)
        res = elu_activate_kernel(x);
    else if (TYPE == SELU)
        res = selu_activate_kernel(x);
    else if (TYPE == RELIE)
        res = relie_activate_kernel(x);
    else if (TYPE == RAMP)
        res = ramp_activate_kernel(x);
    else if (TYPE == LEAKY)
        res = leaky_activate_kernel(x);
    else if (TYPE == TANH)
        res = tanh_activate_kernel(x);
    else if (TYPE == PLSE)
        res = plse_activate_kernel(x);
    else if (TYPE == LHTAN)
        res = lhtan_activate_kernel(x);
    return res;
}

__global__ void activate_list_kernel(float *origin, int num, Activation TYPE)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= num) return;
    origin[index] = activate_x_kernel(TYPE, origin[index]);
}

void activate_list_gpu(float *origin, int num, Activation TYPE)
{
    activate_list_kernel<<<(num+BLOCK-1)/BLOCK, BLOCK>>>(origin, num, TYPE);
}

__device__ float gradient_x_kernel(Activation TYPE, float x)
{
    float res = 0;
    if (TYPE == STAIR)
        res = stair_gradient_kernel(x);
    else if (TYPE == HARDTAN)
        res = hardtan_gradient_kernel(x);
    else if (TYPE == LINEAR)
        res = linear_gradient_kernel(x);
    else if (TYPE == LOGISTIC)
        res = logistic_gradient_kernel(x);
    else if (TYPE == LOGGY)
        res = loggy_gradient_kernel(x);
    else if (TYPE == RELU)
        res = relu_gradient_kernel(x);
    else if (TYPE == ELU)
        res = elu_gradient_kernel(x);
    else if (TYPE == SELU)
        res = selu_gradient_kernel(x);
    else if (TYPE == RELIE)
        res = relie_gradient_kernel(x);
    else if (TYPE == RAMP)
        res = ramp_gradient_kernel(x);
    else if (TYPE == LEAKY)
        res = leaky_gradient_kernel(x);
    else if (TYPE == TANH)
        res = tanh_gradient_kernel(x);
    else if (TYPE == PLSE)
        res = plse_gradient_kernel(x);
    else if (TYPE == LHTAN)
        res = lhtan_gradient_kernel(x);
    return res;
}

__global__ void gradient_list_kernel(float *origin, int num, Activation TYPE)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= num) return;
    origin[index] = gradient_x_kernel(TYPE, origin[index]);
}

void gradient_list_gpu(float *origin, int num, Activation TYPE)
{
    gradient_list_kernel<<<(num+BLOCK-1)/BLOCK, BLOCK>>>(origin, num, TYPE);
}