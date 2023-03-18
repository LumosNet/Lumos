#include "weights_init.h"

void val_init(Layer *l, float val, float scale)
{
    for (int i = 0; i < l->kernel_weights_size; ++i){
        l->kernel_weights[i] = val * scale;
    }
}

void uniform_init(Layer *l, float mean, float variance, float scale)
{
    uniform_list(-variance, variance, l->kernel_weights_size, l->kernel_weights);
    multy_cpu(l->kernel_weights, l->kernel_weights_size, scale, 1);
}

void normal_init(Layer *l, float mean, float variance, float scale)
{
    int seed = (unsigned)time(NULL);
    guass_list(mean, variance, seed, l->kernel_weights_size, l->kernel_weights);
    multy_cpu(l->kernel_weights, l->kernel_weights_size, scale, 1);
}

void xavier_uniform_init(Layer *l, float scale)
{
    float x = sqrt((float)6/(l->inputs+l->outputs));
    uniform_list(-x, x, l->kernel_weights_size, l->kernel_weights);
    multy_cpu(l->kernel_weights, l->kernel_weights_size, scale, 1);
}

void xavier_normal_init(Layer *l, float scale)
{
    int seed = (unsigned)time(NULL);
    float x = (float)2/(l->inputs+l->outputs);
    guass_list(0, x, seed, l->kernel_weights_size, l->kernel_weights);
    multy_cpu(l->kernel_weights, l->kernel_weights_size, scale, 1);
}

void kaiming_uniform_init(Layer *l, float scale, char *mode)
{
    float x = 0;
    float inp = 0;
    float out = 0;
    if (l->type == CONVOLUTIONAL){
        inp = l->ksize*l->ksize*l->input_c;
        out = l->ksize*l->ksize*l->output_c;
    } else {
        inp = l->inputs;
        out = l->outputs;
    }
    if (0 == strcmp(mode, "fan_in")){
        x = sqrt((float)6/inp);
    } else {
        x = sqrt((float)6/out);
    }
    uniform_list(-x, x, l->kernel_weights_size, l->kernel_weights);
    multy_cpu(l->kernel_weights, l->kernel_weights_size, scale, 1);
}

void kaiming_normal_init(Layer *l, float scale, char *mode)
{
    float x = 0;
    float inp = 0;
    float out = 0;
    if (l->type == CONVOLUTIONAL){
        inp = l->ksize*l->ksize*l->input_c;
        out = l->ksize*l->ksize*l->output_c;
    } else {
        inp = l->inputs;
        out = l->outputs;
    }
    if (0 == strcmp(mode, "fan_in")){
        x = sqrt((float)6/inp);
    } else {
        x = sqrt((float)6/out);
    }
    int seed = (unsigned)time(NULL);
    guass_list(0, x, seed, l->kernel_weights_size, l->kernel_weights);
    multy_cpu(l->kernel_weights, l->kernel_weights_size, scale, 1);
}

void he_init(Layer *l)
{
    float scale;
    if (l->type == CONNECT){
        scale = sqrt((float)2 / l->inputs);
        uniform_list(-1, 1, l->kernel_weights_size, l->kernel_weights);
        multy_cpu(l->kernel_weights, l->kernel_weights_size, scale, 1);
    } else if (l->type == CONVOLUTIONAL){
        scale = sqrt((float)2 / (l->ksize*l->ksize*l->input_c));
        normal_list(l->kernel_weights_size, l->kernel_weights);
        multy_cpu(l->kernel_weights, l->kernel_weights_size, scale, 1);
    }
}

Initializer val_initializer(float val, float scale)
{
    Initializer init = {0};
    init.type = "val_init";
    init.val = val;
    init.scale = scale;
    return init;
}

Initializer uniform_initializer(float mean, float variance, float scale)
{
    Initializer init = {0};
    init.type = "uniform_init";
    init.mean = mean;
    init.variance = variance;
    init.scale = scale;
    return init;
}

Initializer normal_initializer(float mean, float variance, float scale)
{
    Initializer init = {0};
    init.type = "normal_init";
    init.mean = mean;
    init.variance = variance;
    init.scale = scale;
    return init;
}

Initializer xavier_uniform_initializer(float scale)
{
    Initializer init = {0};
    init.type = "xavier_uniform_init";
    init.scale = scale;
    return init;
}

Initializer xavier_normal_initializer(float scale)
{
    Initializer init = {0};
    init.type = "xavier_normal_init";
    init.scale = scale;
    return init;
}

Initializer kaiming_uniform_initializer(float scale, char *mode)
{
    Initializer init = {0};
    init.type = "kaiming_uniform_init";
    init.scale = scale;
    init.mode = mode;
    return init;
}

Initializer kaiming_normal_initializer(float scale, char *mode)
{
    Initializer init = {0};
    init.type = "kaiming_normal_init";
    init.scale = scale;
    init.mode = mode;
    return init;
}

Initializer he_initializer()
{
    Initializer init = {0};
    init.type = "he_init";
    return init;
}
