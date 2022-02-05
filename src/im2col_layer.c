#include "im2col_layer.h"

Layer make_im2col_layer(LayerParams *p, int batch, int h, int w, int c)
{
    Layer l = {0};
    l.type = IM2COL;
    l.input_h = h,
    l.input_w = w;
    l.input_c = c;

    l.output_h = 1;
    l.output_w = 1;
    l.output_c = 1;
    Node *n = p->head;
    while (n){
        Params *param = n->val;
        if (0 == strcmp(param->key, "flag")){
            int flag = atoi(param->val);
            if (flag) l.output_h = l.input_h*l.input_w*l.input_c;
            else l.output_w = l.input_h*l.input_w*l.input_c;
        }
        n = n->next;
    }
    l.inputs = l.input_c*l.input_h*l.input_w;
    l.outputs = l.output_c*l.output_h*l.output_w;
    l.forward = forward_im2col_layer;
    l.backward = backward_im2col_layer;
    l.output = calloc(l.outputs, sizeof(float));
    l.delta = calloc(l.inputs, sizeof(float));
    fprintf(stderr, "  im2col                      %4d x%4d x%4d   ->    %2d x%2d\n", \
            l.input_w, l.input_h, l.input_c, l.output_w, l.output_h);
    return l;
}

void forward_im2col_layer(Layer l, Network net)
{
    debug_data(net.fdebug, l.input_h, l.input_w, l.input, "\nim2col_layer input\n");
    memcpy(l.output, l.input, net.batch*l.outputs*sizeof(float));
    debug_data(net.fdebug, l.output_h, l.output_w, l.output, "\nim2col_layer output\n");
}

void backward_im2col_layer(Layer l, Network net)
{
    debug_data(net.bdebug, l.output_h, l.output_w, net.delta, "\nim2col_layer net_delta\n");
    memcpy(l.delta, net.delta, net.batch*l.inputs*sizeof(float));
    debug_data(net.bdebug, l.input_h, l.input_w, l.delta, "\nim2col_layer delta\n");
}