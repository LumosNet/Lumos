#include "softmax_layer.h"

void forward_softmax_layer(Layer l, Network net)
{
    for (int i = 0; i < net.batch; ++i){
        one_hot_encoding(l.group, net.labels[i].data[0], l.truth+i*l.group);
    }
    softmax_cpu(l.input, l.group, net.batch, l.inputs, l.output);
    if (!l.noloss){
        printf("计算loss\n");
        softmax_x_ent_cpu(net.batch*l.outputs, l.output, l.truth, l.delta, l.loss);
    }
    for (int i = 0; i < net.batch; ++i){
        for (int j = 0; j < l.group; ++j){
            printf("%f ", l.truth[i*l.group+j]);
        }
        for (int j = 0; j < l.group; ++j){
            printf("%f ", l.output[i*l.group+j]);
        }
        for (int j = 0; j < l.group; ++j){
            printf("%f ", l.delta[i*l.group+j]);
        }
        printf("\n");
    }
    printf("\n");
}

void backward_softmax_layer(Layer l, Network net)
{
    if (net.delta) saxpy(l.delta, net.delta, net.batch*l.inputs, 1, l.delta);
}

Layer make_softmax_layer(LayerParams *p, int batch, int h, int w, int c)
{
    Layer l = {0};
    l.type = SOFTMAX;
    l.input_h = h;
    l.input_w = w;
    l.input_c = c;
    Node *n = p->head;
    while (n){
        Params *param = n->val;
        if (0 == strcmp(param->key, "group")){
            l.group = atoi(param->val);
        } else if (0 == strcmp(param->key, "noloss")){
            l.noloss = atoi(param->val);
        }
        n = n->next;
    }
    l.output_h = l.group;
    l.output_w = 1;
    l.output_c = 1;
    l.forward = forward_softmax_layer;
    l.backward = backward_softmax_layer;

    l.workspace_size = l.group*l.group;

    int size_o = l.output_w * l.output_h * l.output_c;
    int size_d = l.input_w * l.input_h * l.input_c;
    l.truth = calloc(batch*l.group, sizeof(float));
    l.output = calloc(batch*size_o, sizeof(float));
    l.delta = calloc(batch*size_d, sizeof(float));

    l.inputs = l.input_c*l.input_h*l.input_w;
    l.outputs = l.output_c*l.output_h*l.output_w;

    l.loss = calloc(batch*l.inputs, sizeof(float));

    fprintf(stderr, "  softmax          %5d      %4d x%4d         ->  %4d x%4d\n", \
            l.group, l.input_w, l.input_h, l.output_w, l.output_h);
    return l;
}

void softmax_cpu(float *input, int n, int batch, int batch_offset, float *output)
{
    for (int i = 0; i < batch; ++i){
        softmax(input+i*batch_offset, n, output+i*batch_offset);
    }
}

void softmax(float *input, int n, float *output)
{
    float sum = 0;
    float largest = -FLT_MAX;
    for (int i = 0; i < n; ++i){
        if (input[i] > largest) largest = input[i];
    }
    for (int i = 0; i < n; ++i){
        output[i] = exp(input[i] - largest);
        sum += output[i];
    }
    for (int i = 0; i < n; ++i){
        output[i] /= sum;
    }
}

void softmax_x_ent_cpu(int n, float *pred, float *truth, float *delta, float *error)
{
    for(int i = 0; i < n; ++i){
        float t = truth[i];
        float p = pred[i];
        error[i] = (t) ? -log(p) : 0;
        delta[i] = t-p;
    }
}