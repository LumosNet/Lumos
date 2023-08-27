#include "active.h"

Activate load_activate(char *activate)
{
    if (0 == strcmp(activate, "stair"))
        return stair_activate;
    else if (0 == strcmp(activate, "hardtan"))
        return hardtan_activate;
    else if (0 == strcmp(activate, "linear"))
        return linear_activate;
    else if (0 == strcmp(activate, "logistic"))
        return logistic_activate;
    else if (0 == strcmp(activate, "loggy"))
        return loggy_activate;
    else if (0 == strcmp(activate, "relu"))
        return relu_activate;
    else if (0 == strcmp(activate, "elu"))
        return elu_activate;
    else if (0 == strcmp(activate, "selu"))
        return selu_activate;
    else if (0 == strcmp(activate, "relie"))
        return relie_activate;
    else if (0 == strcmp(activate, "ramp"))
        return ramp_activate;
    else if (0 == strcmp(activate, "leaky"))
        return leaky_activate;
    else if (0 == strcmp(activate, "tanh"))
        return tanh_activate;
    else if (0 == strcmp(activate, "plse"))
        return plse_activate;
    else if (0 == strcmp(activate, "lhtan"))
        return lhtan_activate;
    else {
        fprintf(stderr, "Active Error: Unknown Activate Name!");
    }
    return NULL;
}

Gradient load_gradient(char *activate)
{
    if (0 == strcmp(activate, "stair"))
        return stair_gradient;
    else if (0 == strcmp(activate, "hardtan"))
        return hardtan_gradient;
    else if (0 == strcmp(activate, "linear"))
        return linear_gradient;
    else if (0 == strcmp(activate, "logistic"))
        return logistic_gradient;
    else if (0 == strcmp(activate, "loggy"))
        return loggy_gradient;
    else if (0 == strcmp(activate, "relu"))
        return relu_gradient;
    else if (0 == strcmp(activate, "elu"))
        return elu_gradient;
    else if (0 == strcmp(activate, "selu"))
        return selu_gradient;
    else if (0 == strcmp(activate, "relie"))
        return relie_gradient;
    else if (0 == strcmp(activate, "ramp"))
        return ramp_gradient;
    else if (0 == strcmp(activate, "leaky"))
        return leaky_gradient;
    else if (0 == strcmp(activate, "tanh"))
        return tanh_gradient;
    else if (0 == strcmp(activate, "plse"))
        return plse_gradient;
    else if (0 == strcmp(activate, "lhtan"))
        return lhtan_gradient;
    else {
        fprintf(stderr, "Active Error: Unknown Activate Name!");
    }
    return NULL;
}

void activate_list(float *origin, int num, Activate func)
{
    for (int i = 0; i < num; ++i){
        origin[i] = func(origin[i]);
    }
}

void gradient_list(float *origin, int num, Gradient func)
{
    for (int i = 0; i < num; ++i){
        origin[i] = func(origin[i]);
    }
}