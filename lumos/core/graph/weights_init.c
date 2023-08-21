#include "weights_init.h"

void connect_layer_weights_init(Layer *l, Initializer init)
{
    int kernel_weights_size = l->inputs * l->outputs;
    if (init.type == VALFILL){
        val_init(l->kernel_weights, kernel_weights_size, init.val);
    } else if (init.type == UNIFORM){
        uniform_init(l->kernel_weights, kernel_weights_size, init.l, init.r);
    } else if (init.type == NORMAL){
        normal_init(l->kernel_weights, kernel_weights_size, init.mean, init.variance);
    } else if (init.type == XAVIERU){
        float x = sqrt((float)6/(l->inputs+l->outputs));
        xavier_uniform_init(l->kernel_weights, kernel_weights_size, x);
    } else if (init.type == XAVIERN){
        float x = sqrt((float)6/(l->inputs+l->outputs));
        xavier_normal_init(l->kernel_weights, kernel_weights_size, x);
    } else if (init.type == KAIMINGU){
        int inp = l->inputs;
        int out = l->outputs;
        kaiming_uniform_init(init.mode, l->kernel_weights, kernel_weights_size, inp, out);
    } else if (init.type == KAIMINGN){
        int inp = l->inputs;
        int out = l->outputs;
        kaiming_normal_init(init.mode, l->kernel_weights, kernel_weights_size, inp, out);
    } else if (init.type == HE){
        float scale = sqrt((float)2 / l->inputs);
        uniform_list(-1, 1, kernel_weights_size, l->kernel_weights);
        multy_cpu(l->kernel_weights, kernel_weights_size, scale, 1);
    }
    if (l->bias){
        fill_cpu(l->bias_weights, l->outputs, 0.01, 1);
    }
}

void convolutional_layer_weights_init(Layer *l, Initializer init)
{
    int kernel_weights_size = l->ksize * l->ksize;
    if (init.type == VALFILL){
        for (int i = 0; i < l->filters; ++i){
            val_init(l->kernel_weights+i*kernel_weights_size*l->input_c, kernel_weights_size, init.val);
        }
    } else if (init.type == UNIFORM){
        for (int i = 0; i < l->filters; ++i){
            uniform_init(l->kernel_weights+i*kernel_weights_size*l->input_c, kernel_weights_size, init.l, init.r);
        }
    } else if (init.type == NORMAL){
        for (int i = 0; i < l->filters; ++i){
            normal_init(l->kernel_weights+i*kernel_weights_size*l->input_c, kernel_weights_size, init.mean, init.variance);
        }
    } else if (init.type == XAVIERU){
        float x = sqrt((float)6/(l->inputs+l->outputs));
        for (int i = 0; i < l->filters; ++i){
            xavier_uniform_init(l->kernel_weights+i*kernel_weights_size*l->input_c, kernel_weights_size, x);
        }
    } else if (init.type == XAVIERN){
        float x = sqrt((float)6/(l->inputs+l->outputs));
        for (int i = 0; i < l->filters; ++i){
            xavier_normal_init(l->kernel_weights+i*kernel_weights_size*l->input_c, kernel_weights_size, x);
        }
    } else if (init.type == KAIMINGU){
        int inp = l->ksize*l->ksize*l->input_c;
        int out = l->ksize*l->ksize*l->output_c;
        for (int i = 0; i < l->filters; ++i){
            kaiming_uniform_init(init.mode, l->kernel_weights+i*kernel_weights_size*l->input_c, kernel_weights_size, inp, out);
        }
    } else if (init.type == KAIMINGN){
        int inp = l->ksize*l->ksize*l->input_c;
        int out = l->ksize*l->ksize*l->output_c;
        for (int i = 0; i < l->filters; ++i){
            kaiming_normal_init(init.mode, l->kernel_weights+i*kernel_weights_size*l->input_c, kernel_weights_size, inp, out);
        }
    } else if (init.type == HE){
        float scale = sqrt((float)2 / (l->ksize*l->ksize*l->input_c));
        for (int i = 0; i < l->filters; ++i){
            normal_list(kernel_weights_size, l->kernel_weights+i*kernel_weights_size*l->input_c);
            multy_cpu(l->kernel_weights+i*kernel_weights_size*l->input_c, kernel_weights_size, scale, 1);
        }
    }
    for (int i = 0; i < l->filters; ++i){
        for (int j = 0; j < l->input_c-1; ++i){
            float *weights_src = l->kernel_weights+i*kernel_weights_size*l->input_c;
            float *weights_dst = l->kernel_weights+i*kernel_weights_size*l->input_c+(j+1)*kernel_weights_size;
            memcpy(weights_dst, weights_src, kernel_weights_size*sizeof(float));
        }
    }
    if (l->bias){
        fill_cpu(l->bias_weights, l->filters, 0.01, 1);
    }
}

void val_init(float *space, int num, float val)
{
    fill_cpu(space, num, 0, val);
}

void uniform_init(float *space, int num, float l, float r)
{
    uniform_list(l, r, num, space);
}

void normal_init(float *space, int num, float mean, float variance)
{
    int seed = (unsigned)time(NULL);
    guass_list(mean, variance, seed, num, space);
}

void xavier_uniform_init(float *space, int num, float x)
{
    uniform_list(-x, x, num, space);
}

void xavier_normal_init(float *space, int num, float x)
{
    int seed = (unsigned)time(NULL);
    guass_list(0, x, seed, num, space);
}

void kaiming_uniform_init(char *mode, float *space, int num, int inp, int out)
{
    float x = 0;
    if (0 == strcmp(mode, "fan_in")){
        x = sqrt((float)6/inp);
    } else {
        x = sqrt((float)6/out);
    }
    uniform_list(-x, x, num, space);
}

void kaiming_normal_init(char *mode, float *space, int num, int inp, int out)
{
    float x = 0;
    if (0 == strcmp(mode, "fan_in")){
        x = sqrt((float)6/inp);
    } else {
        x = sqrt((float)6/out);
    }
    int seed = (unsigned)time(NULL);
    guass_list(0, x, seed, num, space);
}

Initializer val_initializer(float val)
{
    Initializer init = {0};
    init.type = "val_init";
    init.val = val;
    return init;
}

Initializer uniform_initializer(float l, float r)
{
    Initializer init = {0};
    init.type = "uniform_init";
    init.l = l;
    init.r = r;
    return init;
}

Initializer normal_initializer(float mean, float variance)
{
    Initializer init = {0};
    init.type = "normal_init";
    init.mean = mean;
    init.variance = variance;
    return init;
}

Initializer xavier_uniform_initializer()
{
    Initializer init = {0};
    init.type = "xavier_uniform_init";
    return init;
}

Initializer xavier_normal_initializer()
{
    Initializer init = {0};
    init.type = "xavier_normal_init";
    return init;
}

Initializer kaiming_uniform_initializer(char *mode)
{
    Initializer init = {0};
    init.type = "kaiming_uniform_init";
    init.mode = mode;
    return init;
}

Initializer kaiming_normal_initializer(char *mode)
{
    Initializer init = {0};
    init.type = "kaiming_normal_init";
    init.mode = mode;
    return init;
}

Initializer he_initializer()
{
    Initializer init = {0};
    init.type = "he_init";
    return init;
}