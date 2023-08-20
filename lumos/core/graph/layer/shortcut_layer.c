#include "shortcut_layer.h"

Layer *make_shortcut_layer(int index, char *active)
{
    Layer *l = malloc(sizeof(Layer));
    l->type = SHORTCUT;
    l->shortcut_index = index;

    l->active_str = active;
    Activation type = load_activate_type(active);
    l->active = type;
    l->gradient = type;

    l->update = NULL;

    fprintf(stderr, "Shortcut        Layer    :    [index=%d, active=%s]\n", l->shortcut_index, l->active_str);
    return l;
}

void init_shortcut_layer(Layer *l, int w, int h, int c, Layer *shortcut)
{
    l->input_h = h;
    l->input_w = w;
    l->input_c = c;
    l->inputs = l->input_h * l->input_w * l->input_c;

    l->output_h = h;
    l->output_w = w;
    l->output_c = c;
    l->outputs = l->output_h * l->output_w * l->output_c;

    l->shortcut = shortcut;
    l->workspace_size = 0;
    l->deltas = l->inputs;

    l->forward = forward_shortcut_layer;
    l->backward = backward_shortcut_layer;

    l->output = calloc(l->outputs, sizeof(float));
    l->delta = calloc(l->inputs, sizeof(float));

    fprintf(stderr, "Shortcut        Layer    %3d*%3d*%3d ==> %3d*%3d*%3d\n",
            l->input_w, l->input_h, l->input_c, l->output_w, l->output_h, l->output_c);
}

void forward_shortcut_layer(Layer l, int num)
{
    Layer *shortcut = l.shortcut;
    for (int i = 0; i < num; ++i){
        int offset_i = i * l.inputs;
        int offset_o = i * l.outputs;
        int offset_c = i * shortcut->outputs;
        float *input = l.input + offset_i;
        float *output = l.output + offset_o;
        float *add = shortcut->output + offset_c;
        shortcut_cpu(add, shortcut->output_w, shortcut->output_h, shortcut->output_c, \
                     input, l.input_w, l.input_h, l.input_c, 1, 1, output);
    }
    activate_list(l.output, num*l.outputs, l.active);
}

void backward_shortcut_layer(Layer l, float rate, int num, float *n_delta)
{
    Layer *shortcut = l.shortcut;
    for (int i = 0; i < num; ++i){
        int offset_i = i * l.inputs;
        int offset_o = i * l.outputs;
        int offset_c = i * shortcut->inputs;
        float *output = l.output + offset_o;
        float *delta_l = l.delta + offset_i;
        float *delta_n = n_delta + offset_o;
        float *out = shortcut->delta + offset_c;
        gradient_list(output, l.outputs, l.active);
        matrix_multiply_cpu(output, delta_n, l.inputs, delta_l);
        shortcut_cpu(delta_l, l.input_w, l.input_h, l.input_c, \
                     out, shortcut->input_w, shortcut->input_h, shortcut->input_c, \
                     1, 1, out);
    }
}
