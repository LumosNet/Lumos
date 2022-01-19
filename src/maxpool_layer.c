#include "maxpool_layer.h"

void forward_maxpool_layer(Layer l, Network net)
{
    char ip[] = "./data/input";
    char op[] = "./data/output";
    char *inp = link_str(ip, int2str(l.i));
    char *oup = link_str(op, int2str(l.i));
    save_data(l.input, l.input_c, l.input_h, l.input_w, net.batch, inp);
    for (int i = 0; i < net.batch; ++i){
        int offset_i = i*l.input_h*l.input_w*l.input_c;
        int offset_o = i*l.output_h*l.output_w*l.output_c;
        float *output = l.output+offset_o;
        im2col(l.input+offset_i, l.input_h, l.input_w, l.input_c, l.ksize, l.stride, l.pad, net.workspace);
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
                output[l.output_h*l.output_w*c + h] = max;
                l.index[i][l.output_h*l.output_w*c + h] = max_index;
            }
        }
    }
    save_data(l.output, l.output_c, l.output_h, l.output_w, net.batch, oup);
}

void backward_maxpool_layer(Layer l, Network net)
{
    for (int i = 0; i < net.batch; ++i){
        int offset_i = i*l.input_h*l.input_w*l.input_c;
        int offset_o = i*l.output_h*l.output_w*l.output_c;
        float *l_delta = l.delta+offset_i;
        float *n_delta = net.delta+offset_o;
        for (int j = 0; j < l.output_h*l.output_w*l.output_c; ++j){
            l_delta[l.index[i][j]] = n_delta[j];
        }
    }
}