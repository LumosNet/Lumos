#include "active.h"

Activation load_activate_type(char *activate)
{
    if (0 == strcmp(activate, "stair"))
        return STAIR;
    else if (0 == strcmp(activate, "hardtan"))
        return HARDTAN;
    else if (0 == strcmp(activate, "linear"))
        return LINEAR;
    else if (0 == strcmp(activate, "logistic"))
        return LOGISTIC;
    else if (0 == strcmp(activate, "loggy"))
        return LOGGY;
    else if (0 == strcmp(activate, "relu"))
        return RELU;
    else if (0 == strcmp(activate, "elu"))
        return ELU;
    else if (0 == strcmp(activate, "selu"))
        return SELU;
    else if (0 == strcmp(activate, "relie"))
        return RELIE;
    else if (0 == strcmp(activate, "ramp"))
        return RAMP;
    else if (0 == strcmp(activate, "leaky"))
        return LEAKY;
    else if (0 == strcmp(activate, "tanh"))
        return TANH;
    else if (0 == strcmp(activate, "plse"))
        return PLSE;
    else if (0 == strcmp(activate, "lhtan"))
        return LHTAN;
    return LEAKY;
}

Activate load_activate(Activation TYPE)
{
    activate func;
    if (TYPE == STAIR)
        func = stair_activate;
    else if (TYPE == HARDTAN)
        func = hardtan_activate;
    else if (TYPE == LINEAR)
        func = linear_activate;
    else if (TYPE == LOGISTIC)
        func = logistic_activate;
    else if (TYPE == LOGGY)
        func = loggy_activate;
    else if (TYPE == RELU)
        func = relu_activate;
    else if (TYPE == ELU)
        func = elu_activate;
    else if (TYPE == SELU)
        func = selu_activate;
    else if (TYPE == RELIE)
        func = relie_activate;
    else if (TYPE == RAMP)
        func = ramp_activate;
    else if (TYPE == LEAKY)
        func = leaky_activate;
    else if (TYPE == TANH)
        func = tanh_activate;
    else if (TYPE == PLSE)
        func = plse_activate;
    else if (TYPE == LHTAN)
        func = lhtan_activate;
    return func;
}

Gradient load_gradient(Activation TYPE)
{
    gradient func;
    if (TYPE == STAIR)
        func = stair_gradient;
    else if (TYPE == HARDTAN)
        func = hardtan_gradient;
    else if (TYPE == LINEAR)
        func = linear_gradient;
    else if (TYPE == LOGISTIC)
        func = logistic_gradient;
    else if (TYPE == LOGGY)
        func = loggy_gradient;
    else if (TYPE == RELU)
        func = relu_gradient;
    else if (TYPE == ELU)
        func = elu_gradient;
    else if (TYPE == SELU)
        func = selu_gradient;
    else if (TYPE == RELIE)
        func = relie_gradient;
    else if (TYPE == RAMP)
        func = ramp_gradient;
    else if (TYPE == LEAKY)
        func = leaky_gradient;
    else if (TYPE == TANH)
        func = tanh_gradient;
    else if (TYPE == PLSE)
        func = plse_gradient;
    else if (TYPE == LHTAN)
        func = lhtan_gradient;
    return func;
}

float activate_x(Activation TYPE, float x)
{
    float res = 0;
    if (TYPE == STAIR)
        res = stair_activate(x);
    else if (TYPE == HARDTAN)
        res = hardtan_activate(x);
    else if (TYPE == LINEAR)
        res = linear_activate(x);
    else if (TYPE == LOGISTIC)
        res = logistic_activate(x);
    else if (TYPE == LOGGY)
        res = loggy_activate(x);
    else if (TYPE == RELU)
        res = relu_activate(x);
    else if (TYPE == ELU)
        res = elu_activate(x);
    else if (TYPE == SELU)
        res = selu_activate(x);
    else if (TYPE == RELIE)
        res = relie_activate(x);
    else if (TYPE == RAMP)
        res = ramp_activate(x);
    else if (TYPE == LEAKY)
        res = leaky_activate(x);
    else if (TYPE == TANH)
        res = tanh_activate(x);
    else if (TYPE == PLSE)
        res = plse_activate(x);
    else if (TYPE == LHTAN)
        res = lhtan_activate(x);
    return res;
}

void activate_list(float *origin, int num, Activation TYPE)
{
    for (int i = 0; i < num; ++i)
    {
        origin[i] = activate_x(TYPE, origin[i]);
    }
}

float gradient_x(Activation TYPE, float x)
{
    float res = 0;
    if (TYPE == STAIR)
        res = stair_gradient(x);
    else if (TYPE == HARDTAN)
        res = hardtan_gradient(x);
    else if (TYPE == LINEAR)
        res = linear_gradient(x);
    else if (TYPE == LOGISTIC)
        res = logistic_gradient(x);
    else if (TYPE == LOGGY)
        res = loggy_gradient(x);
    else if (TYPE == RELU)
        res = relu_gradient(x);
    else if (TYPE == ELU)
        res = elu_gradient(x);
    else if (TYPE == SELU)
        res = selu_gradient(x);
    else if (TYPE == RELIE)
        res = relie_gradient(x);
    else if (TYPE == RAMP)
        res = ramp_gradient(x);
    else if (TYPE == LEAKY)
        res = leaky_gradient(x);
    else if (TYPE == TANH)
        res = tanh_gradient(x);
    else if (TYPE == PLSE)
        res = plse_gradient(x);
    else if (TYPE == LHTAN)
        res = lhtan_gradient(x);
    return res;
}

void gradient_list(float *origin, int num, Activation TYPE)
{
    for (int i = 0; i < num; ++i)
    {
        origin[i] = gradient_x(TYPE, origin[i]);
    }
}