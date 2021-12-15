#include "maxpool_layer.h"

void forward_maxpool_layer(Layer l, Network net)
{
    for (int i = 0; i < net.batch; ++i){
        im2col(l.input[i]->data, l.input_h, l.input_w, l.input_c, l.ksize, l.stride, l.pad, net.workspace);
        for (int c = 0; c < l.input_c; ++c){
            for (int h = 0; h < l.output_h*l.output_w; h++){
                float max = -999;
                int max_index = -1;
                for (int w = 0; w < l.ksize*l.ksize; w++){
                    int mindex = c*l.output_h*l.output_w*l.ksize*l.ksize + l.output_h*l.output_w*w + h;
                    if (net.workspace[mindex] > max){
                        max = net.workspace[mindex];
                        max_index = (c*l.input_h*l.input_w)+(h/l.output_w*l.ksize+w/l.ksize)*l.input_w+(h%l.output_w*l.ksize+w%l.ksize);
                    }
                }
                l.output[i]->data[l.output_h*l.output_w*c + h] = max;
                l.index[i][l.output_h*l.output_w*c + h] = max_index;
            }
        }
    }
}

void backward_maxpool_layer(Layer l, Network net)
{
    for (int i = 0; i < net.batch; ++i){
        for (int j = 0; j < l.output_h*l.output_w*l.output_c; ++j){
            l.delta[i]->data[l.index[i][j]] = net.delta[i]->data[j];
        }
    }
}