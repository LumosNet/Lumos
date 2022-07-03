#include "layer.h"

void restore_layer(Layer *l)
{
    l->input_h = -1;
    l->input_w = -1;
    l->input_c = -1;

    l->output_h = -1;
    l->output_w = -1;
    l->output_c = -1;

    l->inputs = -1;
    l->outputs = -1;

    l->kernel_weights_size = -1;
    l->bias_weights_size = -1;

    l->deltas = -1;
    l->workspace_size = 0;
    l->label_num = 0;

    l->input = NULL;
    l->output = NULL;
    l->label = NULL;
    l->delta = NULL;

    l->workspace = NULL;
    l->maxpool_index = NULL;

    l->kernel_weights = NULL;
    l->bias_weights = NULL;

    l->update_kernel_weights = NULL;
    l->update_bias_weights = NULL;
}
