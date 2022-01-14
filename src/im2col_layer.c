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
    fprintf(stderr, "  im2col                      %4d x%4d x%4d   ->    %2d x%2d\n", \
            l.input_w, l.input_h, l.input_c, l.output_w, l.output_h);
    return l;
}