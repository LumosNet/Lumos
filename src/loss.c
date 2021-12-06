#include "loss.h"

// label表示是第几类
int *one_hot_encoding(int n, int label)
{
    int *code = calloc(n, sizeof(int));
    code[label-1] = 1;
    return code;
}

float mse(Tensor *yi, Tensor *yh)
{
    Tensor *y = tensor_copy(yi);
    ts_subtract(y, yh);
    Tensor *x = tensor_copy(y);
    transposition(x);
    float *space = calloc(x->size[1]*y->size[0], sizeof(float));
    gemm(x, y, space);
    float res = space[0] / yi->num;
    free_tensor(y);
    free_tensor(x);
    free(space);
    return res;
}

float mae(Tensor *yi, Tensor *yh)
{
    int sum = 0;
    for (int i = 0; i < yi->num; ++i){
        sum += fabs(yi->data[i] - yh->data[i]);
    }
    return sum / yi->num;
}

float huber(Tensor *yi, Tensor *yh, float theta)
{
    float huber = 0;
    for (int i = 0; i < yi->num; ++i){
        float differ = fabs(yi->data[i] - yh->data[i]);
        if (differ <= theta) huber += pow(differ, 2) / 2;
        else huber += theta * differ - 0.5 * pow(theta, 2);
    }
    return huber / yi->num;
}

float quantile(Tensor *yi, Tensor *yh, float gamma)
{
    float quant = 0;
    for (int i = 0; i < yi->num; ++i){
        float differ = fabs(yi->data[i] - yh->data[i]);
        if (yi->data[i] <  yh->data[i]){
            quant += (1-gamma) * differ;
        }
        else quant += gamma * differ;
    }
    return quant / yi->num;
}

float cross_entropy(Tensor *yi, Tensor *yh)
{
    float entropy = 0;
    for (int i = 0; i < yi->num; ++i){
        entropy += yi->data[i] * log(yh->data[i]);
    }
    return -entropy;
}

float hinge(Tensor *yi, Tensor *yh)
{
    float hinge = 0;
    for (int i = 0; i < yi->num; ++i){
        float x = 1 - SGN(yi->data[i])*yh->data[i];
        hinge += MAX(0, x);
    }
    return hinge;
}

void forward_mse_loss(Layer *l, Network *net)
{
    for (int i = 0; i < net->batch; ++i){
        Tensor *output = l->output[i];
        output->data[0] = mse(net->labels[i], l->input[i]);
    }
}

void forward_mae_loss(Layer *l, Network *net)
{
    for (int i = 0; i < net->batch; ++i){
        Tensor *output = l->output[i];
        output->data[0] = mae(net->labels[i], l->input[i]);
    }
}

void forward_huber_loss(Layer *l, Network *net)
{
    for (int i = 0; i < net->batch; ++i){
        Tensor *output = l->output[i];
        output->data[0] = huber(net->labels[i], l->input[i], l->theta);
    }
}

void forward_quantile_loss(Layer *l, Network *net)
{
    for (int i = 0; i < net->batch; ++i){
        Tensor *output = l->output[i];
        output->data[0] = quantile(net->labels[i], l->input[i], l->gamma);
    }
}

void forward_cross_entropy_loss(Layer *l, Network *net)
{
    for (int i = 0; i < net->batch; ++i){
        Tensor *output = l->output[i];
        output->data[0] = cross_entropy(net->labels[i], l->input[i]);
    }
}

void forward_hinge_loss(Layer *l, Network *net)
{
    for (int i = 0; i < net->batch; ++i){
        Tensor *output = l->output[i];
        output->data[0] = hinge(net->labels[i], l->input[i]);
    }
}

void backward_mse_loss(Layer *l, Network *net)
{
    for (int i = 0; i < net->batch; ++i){
        Tensor *delta = net->delta[i];
        Tensor *input = l->input[i];
        memcpy_float_list(delta->data, input->data, 0, 0, delta->num);
        ts_subtract(delta, net->labels[i]);
        ts_mult_x(delta, 2/delta->num);
    }
}

void backward_mae_loss(Layer *l, Network *net)
{
    for (int i = 0; i < net->batch; ++i){
        Tensor *delta = net->delta[i];
        Tensor *input = l->input[i];
        Tensor *label = net->labels[i];
        for (int j = 0; j < delta->num; ++j){
            delta->data[j] = 1/delta->num;
            if (label->data[j] - input->data[j] >= 0) delta->data[j] *= -1;
        }
    }
}

void backward_huber_loss(Layer *l, Network *net)
{
    for (int i = 0; i < net->batch; ++i){
        Tensor *delta = net->delta[i];
        Tensor *input = l->input[i];
        Tensor *label = net->labels[i];
        for (int j = 0; j < delta->num; ++j){
            float differ = fabs(label->data[j] - input->data[j]);
            if (differ <= l->theta) delta->data[j] = -differ;
            else {
                if (label->data[j] - input->data[j] >= 0) delta->data[j] = -l->theta;
                else delta->data[j] = l->theta;
            }
        }
    }
}

void backward_quantile_loss(Layer *l, Network *net)
{
    for (int i = 0; i < net->batch; ++i){
        Tensor *delta = net->delta[i];
        Tensor *input = l->input[i];
        Tensor *label = net->labels[i];
        for (int j = 0; j < delta->num; ++j){
            float differ = label->data[j] - input->data[j];
            if (differ < 0) delta->data[j] = 1-l->gamma;
            else delta->data[j] = -l->gamma;
        }
    }
}

void backward_cross_entropy_loss(Layer *l, Network *net)
{
    for (int i = 0; i < net->batch; ++i){
        Tensor *delta = net->delta[i];
        Tensor *input = l->input[i];
        Tensor *label = net->labels[i];
        for (int j = 0; j < delta->num; ++j){
            if (label->data[j] == 0) delta->data[j] = 0;
            else delta->data[j] = -label->data[j] / (input->data[j] + .00000001);
        }
    }
}

void backward_hinge_loss(Layer *l, Network *net)
{
    for (int i = 0; i < net->batch; ++i){
        Tensor *delta = net->delta[i];
        Tensor *input = l->input[i];
        Tensor *label = net->labels[i];
        for (int j = 0; j < delta->num; ++j){
            float differ = input->data[j] * label->data[j];
            if (differ >= 1) delta->data[j] = 0;
            else delta->data[j] = -label->data[j];
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