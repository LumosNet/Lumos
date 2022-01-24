#include "avgpool_layer.h"
#include "debug.h"

void forward_avgpool_layer(Layer l, Network net)
{
    char ip[] = "./data/input";
    char op[] = "./data/output";
    char *inp = link_str(ip, int2str(l.i));
    char *oup = link_str(op, int2str(l.i));
    save_data(l.input, l.input_c, l.input_h, l.input_w, net.batch, inp);
    for (int i = 0; i < net.batch; ++i){
        int offset_i = i*l.input_h*l.input_w*l.input_c;
        int offset_o = i*l.output_h*l.output_w*l.output_c;
        im2col(l.input+offset_i, l.input_h, l.input_w, l.input_c, 
            l.ksize, l.stride, l.pad, net.workspace);
        for (int c = 0; c < l.output_c; ++c){
            for (int h = 0; h < l.output_h; ++h){
                for (int w = 0; w < l.output_w; ++w){
                    for (int k = 0; k < l.ksize*l.ksize; ++k){
                        l.output[offset_o+c*l.output_h*l.output_w + h*l.output_w + w] += 
                        net.workspace[(c*l.ksize*l.ksize+k)*l.output_h*l.output_w+h*l.output_w+w] * (float)(1 / (float)(l.ksize*l.ksize));
                    }
                }
            }
        }
    }
    save_data(l.output, l.output_c, l.output_h, l.output_w, net.batch, oup);
}

void backward_avgpool_layer(Layer l, Network net)
{
    for (int i = 0; i < net.batch; ++i){
        int offset_ld = i*l.input_h*l.input_w*l.input_c;
        int offset_nd = i*l.output_h*l.output_w*l.output_c;
        for (int c = 0; c < l.output_c; ++c){
            for (int h = 0; h < l.input_h; ++h){
                for (int w = 0; w < l.input_w; ++w){
                    int height_index = h / l.ksize;
                    int width_index = w / l.ksize;
                    l.delta[offset_ld + l.input_h*l.input_w*c + l.input_w*h + w] = 
                    net.delta[offset_nd + c*l.output_h*l.output_w + height_index*l.output_w + width_index] * (float)(1 / (float)(l.ksize*l.ksize));
                }
            }
        }
    }
}