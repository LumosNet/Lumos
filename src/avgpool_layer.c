#include "avgpool_layer.h"

void forward_avgpool_layer(Layer l, Network net)
{
    printf("avgpool\n");
    for (int i = 0; i < net.batch; ++i){
        im2col(l.input[i]->data, l.input_h, l.input_w, l.input_c, 
            l.ksize, l.stride, l.pad, net.workspace);
        for (int c = 0; c < l.output_c; ++c){
            for (int h = 0; h < l.output_h; ++h){
                for (int w = 0; w < l.output_w; ++w){
                    for (int k = 0; k < l.ksize*l.ksize; ++k){
                        l.output[i]->data[c*l.output_h*l.output_w + h*l.output_w + w] += 
                        net.workspace[(c*l.ksize*l.ksize+k)*l.output_h*l.output_w+h*l.output_w+w] * (float)(1 / (float)(l.ksize*l.ksize));
                    }
                }
            }
        }
    }
    printf("pooling\n");
}

void backward_avgpool_layer(Layer l, Network net)
{
    for (int i = 0; i < net.batch; ++i){
        for (int c = 0; c < net.delta[i]->size[2]; ++c){
            for (int h = 0; h < l.input_h; ++h){
                for (int w = 0; w < l.input_w; ++w){
                    int height_index = h / l.ksize;
                    int width_index = w / l.ksize;
                    l.delta[i]->data[l.input_h*l.input_w*c + l.input_w*h + w] = 
                    net.delta[i]->data[c*l.output_h*l.output_w + height_index*l.output_w + width_index] * (float)(1 / (float)(l.ksize*l.ksize));
                }
            }
        }
    }
}