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

Activate load_activate_gpu(char *activate)
{
    if (activate == "stair")
        return stair_activate_kernel;
    else if (activate == "hardtan")
        return hardtan_activate_kernel;
    else if (activate == "linear")
        return linear_activate_kernel;
    else if (activate == "logistic")
        return logistic_activate_kernel;
    else if (activate == "loggy")
        return loggy_activate_kernel;
    else if (activate == "relu")
        return relu_activate_kernel;
    else if (activate == "elu")
        return elu_activate_kernel;
    else if (activate == "selu")
        return selu_activate_kernel;
    else if (activate == "relie")
        return relie_activate_kernel;
    else if (activate == "ramp")
        return ramp_activate_kernel;
    else if (activate == "leaky")
        return leaky_activate_kernel;
    else if (activate == "tanh")
        return tanh_activate_kernel;
    else if (activate == "plse")
        return plse_activate_kernel;
    else if (activate == "lhtan")
        return lhtan_activate_kernel;
    else {
        fprintf(stderr, "Active Error: Unknown Activate Name!");
    }
    return NULL;
}

Gradient load_gradient_gpu(char *activate)
{
    if (activate == "stair")
        return stair_gradient_kernel;
    else if (activate == "hardtan")
        return hardtan_gradient_kernel;
    else if (activate == "linear")
        return linear_gradient_kernel;
    else if (activate == "logistic")
        return logistic_gradient_kernel;
    else if (activate == "loggy")
        return loggy_gradient_kernel;
    else if (activate == "relu")
        return relu_gradient_kernel;
    else if (activate == "elu")
        return elu_gradient_kernel;
    else if (activate == "selu")
        return selu_gradient_kernel;
    else if (activate == "relie")
        return relie_gradient_kernel;
    else if (activate == "ramp")
        return ramp_gradient_kernel;
    else if (activate == "leaky")
        return leaky_gradient_kernel;
    else if (activate == "tanh")
        return tanh_gradient_kernel;
    else if (activate == "plse")
        return plse_gradient_kernel;
    else if (activate == "lhtan")
        return lhtan_gradient_kernel;
    else {
        fprintf(stderr, "Active Error: Unknown Activate Name!");
    }
    return NULL;
}

__global__ void activate_list_kernel(float *origin, int num, Activate func)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= num) return;
    origin[index] = func(origin[index]);
}

void activate_list_gpu(float *origin, int num, Activate func)
{
    activate_list_kernel<<<(num+BLOCK-1)/BLOCK, BLOCK>>>(origin, num, func);
}

__global__ void gradient_list_kernel(float *origin, int num, Gradient func)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= num) return;
    origin[index] = func(origin[index]);
}

void gradient_list_gpu(float *origin, int num, Gradient func)
{
    gradient_list_kernel<<<(num+BLOCK-1)/BLOCK, BLOCK>>>(origin, num, func);
}
