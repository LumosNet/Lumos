#include "loss.h"

// label表示是第几类
float *one_hot_encoding(int n, int label)
{
    float *code = calloc(n, sizeof(float));
    code[label] = (float)1;
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
        if (yh[i] <= 0.000001) entropy += yi[i] * log(0.001);
        else entropy += yi[i] * log(yh[i]);
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
        int offset_i = i*l.input_h*l.input_w*l.input_c;
        int offset_o = i*l.output_h*l.output_w*l.output_c;
        float *yi = one_hot_encoding(net.kinds, net.labels[i].data[0]);
        float *loss = l.output+offset_o;
        loss[0] = mse(yi, l.input+offset_i, net.kinds, net.workspace);
    }
}

void forward_mae_loss(Layer l, Network net)
{
    for (int i = 0; i < net.batch; ++i){
        int offset_i = i*l.input_h*l.input_w*l.input_c;
        int offset_o = i*l.output_h*l.output_w*l.output_c;
        float *yi = one_hot_encoding(net.kinds, net.labels[i].data[0]);
        float *loss = l.output+offset_o;
        loss[0] = mae(yi, l.input+offset_i, net.kinds);
    }
}

void forward_huber_loss(Layer l, Network net)
{
    for (int i = 0; i < net.batch; ++i){
        int offset_i = i*l.input_h*l.input_w*l.input_c;
        int offset_o = i*l.output_h*l.output_w*l.output_c;
        float *yi = one_hot_encoding(net.kinds, net.labels[i].data[0]);
        float *loss = l.output+offset_o;
        loss[0] = huber(yi, l.input+offset_i, net.kinds, l.theta);
    }
}

void forward_quantile_loss(Layer l, Network net)
{
    for (int i = 0; i < net.batch; ++i){
        int offset_i = i*l.input_h*l.input_w*l.input_c;
        int offset_o = i*l.output_h*l.output_w*l.output_c;
        float *yi = one_hot_encoding(net.kinds, net.labels[i].data[0]);
        float *loss = l.output+offset_o;
        loss[0] = quantile(yi, l.input+offset_i, net.kinds, l.gamma);
    }
}

void forward_cross_entropy_loss(Layer l, Network net)
{
    float n = 0;
    for (int i = 0; i < net.batch; ++i){
        int offset_i = i*l.input_h*l.input_w*l.input_c;
        int offset_o = i*l.output_h*l.output_w*l.output_c;
        float *yi = one_hot_encoding(net.kinds, net.labels[i].data[0]);
        float *loss = l.output+offset_o;
        loss[0] = cross_entropy(yi, l.input+offset_i, net.kinds);
        n += loss[0];
    }
}

void forward_hinge_loss(Layer l, Network net)
{
    for (int i = 0; i < net.batch; ++i){
        int offset_i = i*l.input_h*l.input_w*l.input_c;
        int offset_o = i*l.output_h*l.output_w*l.output_c;
        float *yi = one_hot_encoding(net.kinds, net.labels[i].data[0]);
        float *loss = l.output+offset_o;
        loss[i] = hinge(yi, l.input+offset_i, net.kinds);
    }
}

void backward_mse_loss(Layer l, Network net)
{
    for (int i = 0; i < net.batch; ++i){
        int offset_i = i*l.input_h*l.input_w*l.input_c;
        float *yi = one_hot_encoding(net.kinds, net.labels[i].data[0]);
        memcpy_float_list(l.delta+offset_i, l.input+offset_i, 0, 0, l.input_h*l.input_w*l.input_c);
        subtract(l.delta+offset_i, yi, l.input_h*l.input_w*l.input_c, l.delta+offset_i);
        mult_x(l.delta+offset_i, l.input_h*l.input_w*l.input_c, 2/(float)(l.input_h*l.input_w*l.input_c));
    }
}

void backward_mae_loss(Layer l, Network net)
{
    for (int i = 0; i < net.batch; ++i){
        int offset_i = i*l.input_h*l.input_w*l.input_c;
        float *yi = one_hot_encoding(net.kinds, net.labels[i].data[0]);
        for (int j = 0; j < l.input_h*l.input_w*l.input_c; ++j){
            float *delta = l.delta+offset_i;
            float *input = l.input+offset_i;
            delta[j] = 1/(l.input_h*l.input_w*l.input_c);
            if (yi[j] > input[j]) delta[j] *= -1;
        }
    }
}

void backward_huber_loss(Layer l, Network net)
{
    for (int i = 0; i < net.batch; ++i){
        int offset_i = i*l.input_h*l.input_w*l.input_c;
        float *yi = one_hot_encoding(net.kinds, net.labels[i].data[0]);
        float *input = l.input+offset_i;
        float *delta = l.delta+offset_i;
        for (int j = 0; j < l.input_h*l.input_w*l.input_c; ++j){
            float differ = fabs(yi[j] - input[j]);
            if (differ <= l.theta) delta[j] = -differ;
            else {
                if (yi[j] - input[j] >= 0) delta[j] = -l.theta;
                else delta[j] = l.theta;
            }
        }
    }
}

void backward_quantile_loss(Layer l, Network net)
{
    for (int i = 0; i < net.batch; ++i){
        int offset_i = i*l.input_h*l.input_w*l.input_c;
        float *yi = one_hot_encoding(net.kinds, net.labels[i].data[0]);
        float *input = l.input+offset_i;
        float *delta = l.delta+offset_i;
        for (int j = 0; j < l.input_h*l.input_w*l.input_c; ++j){
            float differ = yi[j] - input[j];
            if (differ < 0) delta[j] = 1-l.gamma;
            else delta[j] = -l.gamma;
        }
    }
}

void backward_cross_entropy_loss(Layer l, Network net)
{
    for (int i = 0; i < net.batch; ++i){
        int offset_i = i*l.input_h*l.input_w*l.input_c;
        float *yi = one_hot_encoding(net.kinds, net.labels[i].data[0]);
        float *input = l.input+offset_i;
        float *delta = l.delta+offset_i;
        for (int j = 0; j < l.input_h*l.input_w*l.input_c; ++j){
            if (yi[j] == 0) delta[j] = 0;
            else {
                if (input[j] <= 0.000001) {
                    delta[j] = -yi[j] / 0.000001;
                }
                else {
                    delta[j] = -yi[j] / input[j];
                }
            }
            printf("0 %f %f delta_j:%f\n", yi[j], input[j], delta[j]);
        }
        printf("\n");
    }
}

void backward_hinge_loss(Layer l, Network net)
{
    for (int i = 0; i < net.batch; ++i){
        int offset_i = i*l.input_h*l.input_w*l.input_c;
        float *yi = one_hot_encoding(net.kinds, net.labels[i].data[0]);
        float *input = l.input+offset_i;
        float *delta = l.delta+offset_i;
        for (int j = 0; j < l.input_h*l.input_w*l.input_c; ++j){
            float differ = input[j] * yi[j];
            if (differ >= 1) delta[j] = 0;
            else delta[j] = -yi[j];
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