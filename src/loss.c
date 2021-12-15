#include "loss.h"

// label表示是第几类
float *one_hot_encoding(int n, int label)
{
    float *code = calloc(n, sizeof(float));
    code[label-1] = (float)1;
    return code;
}

float mse(float *yi, float *yh, int n, float *space)
{
    subtract(yh, yi, n, space);
    gemm(1, 0, n, 1, n, 1, 1, space, space, space+n);
    float res = space[n] / n;
    return res;
}

float mae(float *yi, float *yh, int n)
{
    int sum = 0;
    for (int i = 0; i < n; ++i){
        sum += fabs(yi[i] - yh[i]);
    }
    return sum / n;
}

float huber(float *yi, float *yh, int n, float theta)
{
    float huber = 0;
    for (int i = 0; i < n; ++i){
        float differ = fabs(yi[i] - yh[i]);
        if (differ <= theta) huber += pow(differ, 2) / 2;
        else huber += theta * differ - 0.5 * pow(theta, 2);
    }
    return huber / n;
}

float quantile(float *yi, float *yh, int n, float gamma)
{
    float quant = 0;
    for (int i = 0; i < n; ++i){
        float differ = fabs(yi[i] - yh[i]);
        if (yi[i] <  yh[i]){
            quant += (1-gamma) * differ;
        }
        else quant += gamma * differ;
    }
    return quant / n;
}

float cross_entropy(float *yi, float *yh, int n)
{
    float entropy = 0;
    for (int i = 0; i < n; ++i){
        entropy += yi[i] * log(yh[i]);
    }
    return -entropy;
}

float hinge(float *yi, float *yh, int n)
{
    float hinge = 0;
    for (int i = 0; i < n; ++i){
        float x = 1 - SGN(yi[i])*yh[i];
        hinge += MAX(0, x);
    }
    return hinge;
}

void forward_mse_loss(Layer l, Network net)
{
    for (int i = 0; i < net.batch; ++i){
        l.output[i]->data[0] = mse(net.labels[i]->data, l.input[i]->data, net.labels[i]->num, net.workspace);
    }
}

void forward_mae_loss(Layer l, Network net)
{
    for (int i = 0; i < net.batch; ++i){
        l.output[i]->data[0] = mae(net.labels[i]->data, l.input[i]->data, net.labels[i]->num);
    }
}

void forward_huber_loss(Layer l, Network net)
{
    for (int i = 0; i < net.batch; ++i){
        l.output[i]->data[0] = huber(net.labels[i]->data, l.input[i]->data, net.labels[i]->num, l.theta);
    }
}

void forward_quantile_loss(Layer l, Network net)
{
    for (int i = 0; i < net.batch; ++i){
        l.output[i]->data[0] = quantile(net.labels[i]->data, l.input[i]->data, net.labels[i]->num, l.gamma);
    }
}

void forward_cross_entropy_loss(Layer l, Network net)
{
    for (int i = 0; i < net.batch; ++i){
        l.output[i]->data[0] = cross_entropy(net.labels[i]->data, l.input[i]->data, net.labels[i]->num);
    }
}

void forward_hinge_loss(Layer l, Network net)
{
    for (int i = 0; i < net.batch; ++i){
        l.output[i]->data[0] = hinge(net.labels[i]->data, l.input[i]->data, net.labels[i]->num);
    }
}

void backward_mse_loss(Layer l, Network net)
{
    for (int i = 0; i < net.batch; ++i){
        memcpy_float_list(l.delta[i]->data, l.input[i]->data, 0, 0, l.delta[i]->num);
        subtract(l.delta[i]->data, net.labels[i]->data, l.delta[i]->num, l.delta[i]->data);
        mult_x(l.delta[i]->data, l.delta[i]->num, 2/(float)l.delta[i]->num);
    }
}

void backward_mae_loss(Layer l, Network net)
{
    for (int i = 0; i < net.batch; ++i){
        for (int j = 0; j < l.delta[i]->num; ++j){
            l.delta[i]->data[j] = 1/l.delta[i]->num;
            if (net.labels[i]->data[j] > l.input[i]->data[j]) l.delta[i]->data[j] *= -1;
        }
    }
}

void backward_huber_loss(Layer l, Network net)
{
    for (int i = 0; i < net.batch; ++i){
        for (int j = 0; j < l.delta[i]->num; ++j){
            float differ = fabs(net.labels[i]->data[j] - l.input[i]->data[j]);
            if (differ <= l.theta) l.delta[i]->data[j] = -differ;
            else {
                if (net.labels[i]->data[j] - l.input[i]->data[j] >= 0) l.delta[i]->data[j] = -l.theta;
                else l.delta[i]->data[j] = l.theta;
            }
        }
    }
}

void backward_quantile_loss(Layer l, Network net)
{
    for (int i = 0; i < net.batch; ++i){
        for (int j = 0; j < l.delta[i]->num; ++j){
            float differ = net.labels[i]->data[j] - l.input[i]->data[j];
            if (differ < 0) l.delta[i]->data[j] = 1-l.gamma;
            else l.delta[i]->data[j] = -l.gamma;
        }
    }
}

void backward_cross_entropy_loss(Layer l, Network net)
{
    for (int i = 0; i < net.batch; ++i){
        for (int j = 0; j < l.delta[i]->num; ++j){
            if (net.labels[i]->data[j] == 0) l.delta[i]->data[j] = 0;
            else l.delta[i]->data[j] = -net.labels[i]->data[j] / (l.input[i]->data[j] + .00000001);
        }
    }
}

void backward_hinge_loss(Layer l, Network net)
{
    for (int i = 0; i < net.batch; ++i){
        for (int j = 0; j < l.delta[i]->num; ++j){
            float differ = l.input[i]->data[j] * net.labels[i]->data[j];
            if (differ >= 1) l.delta[i]->data[j] = 0;
            else l.delta[i]->data[j] = -net.labels[i]->data[j];
        }
    }
}

LossType load_loss_type(char *loss)
{
    if (0 == strcmp(loss, "mse"))                   return MSE;
    else if (0 == strcmp(loss, "mae"))              return MAE;
    else if (0 == strcmp(loss, "huber"))            return HUBER;
    else if (0 == strcmp(loss, "quantile"))         return QUANTILE;
    else if (0 == strcmp(loss, "cross_entropy"))    return CROSS_ENTROPY;
    else if (0 == strcmp(loss, "hinge"))            return HINGE;
    return MSE;
}

Forward load_forward_loss(LossType TYPE)
{
    forward func;
    if (TYPE == MSE)                    func = forward_mse_loss;
    else if (TYPE == MAE)               func = forward_mae_loss;
    else if (TYPE == HUBER)             func = forward_huber_loss;
    else if (TYPE == QUANTILE)          func = forward_quantile_loss;
    else if (TYPE == CROSS_ENTROPY)     func = forward_cross_entropy_loss;
    else if (TYPE == HINGE)             func = forward_hinge_loss;
    return func;
}

Backward load_backward_loss(LossType TYPE)
{
    backward func;
    if (TYPE == MSE)                    func = backward_mse_loss;
    else if (TYPE == MAE)               func = backward_mae_loss;
    else if (TYPE == HUBER)             func = backward_huber_loss;
    else if (TYPE == QUANTILE)          func = backward_quantile_loss;
    else if (TYPE == CROSS_ENTROPY)     func = backward_cross_entropy_loss;
    else if (TYPE == HINGE)             func = backward_hinge_loss;
    return func;
}