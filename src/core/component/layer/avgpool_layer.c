#include "avgpool_layer.h"

void forward_avgpool_layer(Layer l, Network net)
{
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
}

void backward_avgpool_layer(Layer l, Network net)
{
    for (int i = 0; i < net.batch; ++i){
        int offset_ld = i*l.inputs;
        int offset_nd = i*l.outputs;
        for (int c = 0; c < l.input_c; ++c){
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