#include "mse_layer.h"

Layer *make_mse_layer(int group)
{
     Layer *l = malloc(sizeof(Layer));
     l->type = MSE;
     l->group = group;

     l->forward = forward_mse_layer;
     l->backward = backward_mse_layer;
     l->update = NULL;

     restore_mse_layer(l);

     fprintf(stderr, "Mse             Layer    :    [output=%4d]\n", 1);
     return l;
}

Layer *make_mse_layer_by_cfg(CFGParams *p)
{
    int group = 0;

    CFGParam *param = p->head;
    while (param){
        if (0 == strcmp(param->key, "group")){
            group = atoi(param->val);
        }
        param = param->next;
    }

    Layer *l = make_mse_layer(group);
    return l;
}

void init_mse_layer(Layer *l, int w, int h, int c)
{
    l->input_h = h;
    l->input_w = w;
    l->input_c = c;
    l->inputs = l->input_h*l->input_w*l->input_c;

    l->output_h = 1;
    l->output_w = 1;
    l->output_c = 1;
    l->outputs = l->output_h*l->output_w*l->output_c;

    l->deltas = l->inputs;

    fprintf(stderr, "Mse             Layer    %3d*%3d*%3d ==> %3d*%3d*%3d\n", \
            l->input_w, l->input_h, l->input_c, l->output_w, l->output_h, l->output_c);
}

void restore_mse_layer(Layer *l)
{
    l->input_h = -1;
    l->input_w = -1;
    l->input_c = -1;
    l->inputs = -1;

    l->output_h = -1;
    l->output_w = -1;
    l->output_c = -1;
    l->outputs = -1;

    l->workspace_size = -1;

    l->deltas = -1;

    l->input = NULL;
    l->output = NULL;
    l->delta = NULL;
    l->kernel_weights = NULL;
    l->bias_weights = NULL;
    l->label = NULL;
}

void forward_mse_layer(Layer l, int num)
{
    printf("num: %d\n", num);
    float *truth = calloc(l.group*num, sizeof(float));
    truth[0] = 0;
    truth[1] = 1;
    truth[2] = 0;
    truth[3] = 1;
    fill_cpu(l.delta, l.deltas*num, 0, 1);
    for (int i = 0; i < num; ++i){
        printf("input: %f\n", l.input[i]);
        int input_offset = i*l.inputs;
        int output_offset = i*l.outputs;
        float *input = l.input+input_offset;
        float *output = l.output+output_offset;
        subtract(truth+i*l.group, input, l.inputs, l.workspace);
        gemm(1, 0, l.input_h, l.input_w, l.input_h, l.input_w, 1, \
            l.workspace, l.workspace, output);
        multy_cpu(output, l.outputs, 1/(float)l.group, 1);
    }
    float loss = 0;
    for (int i = 0; i < num; ++i){
        loss += l.output[i];
        printf("p:%f t:%f\n", l.input[i], truth[i]);
    }
    printf("all loss: %f\n", loss);
    printf("loss: %f\n", loss/num);
    free(truth);
}

void backward_mse_layer(Layer l, int num, float *n_delta)
{
    float *truth = calloc(l.group*num, sizeof(float));
    truth[0] = 0;
    truth[1] = 1;
    truth[2] = 0;
    truth[3] = 1;
    for (int i = 0; i < num; ++i){
        int label_offset = i*l.label_num;
        int input_offset = i*l.inputs;
        subtract(l.input+input_offset, truth+i*l.group, l.inputs, l.delta+input_offset);
        multy_cpu(l.delta+input_offset, l.inputs, (float)2/l.group, 1);
    }
    printf("finish mse back\n");
}