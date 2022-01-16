#include "lumos.h"
#include "network.h"
#include "utils.h"
#include "data.h"

#include "debug.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv)
{
    Network *net = load_network("./cfg/lumos.cfg");
    init_network(net, "./mnist/mnist.data", NULL);
    load_train_data(net, 0);

    // train(net);

    // for (int bt = 0; bt < net->batch; ++bt){
    //     for (int k = 0; k < net->channel; ++k){
    //         for (int i = 0; i < net->height; ++i){
    //             for (int j = 0; j < net->width; ++j){
    //                 printf("%f ", net->input[bt*net->channel*net->width*net->height+k*net->width*net->height+i*net->width+j]);
    //             }
    //             printf("\n");
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
    //     printf("\n");
    // }

    Layer *l = &net->layers[0];
    l->input = net->output;
    l->forward(l[0], net[0]);
    net->output = l->output;
    save_data(l->input, l->input_c, l->input_h, l->input_w, net->batch, "./data/input.txt");
    save_data(l->kernel_weights, 1, l->ksize*l->ksize*l->input_c, l->filters, 1, "./data/cvkw1.txt");
    save_data(l->bias_weights, 1, 1, l->filters, 1, "./data/cvbw1.txt");

    l = &net->layers[1];
    l->input = net->output;
    l->forward(l[0], net[0]);
    net->output = l->output;
    save_data(l->input, l->input_c, l->input_h, l->input_w, net->batch, "./data/convoltion1.txt");

    l = &net->layers[2];
    l->input = net->output;
    l->forward(l[0], net[0]);
    net->output = l->output;
    save_data(l->input, l->input_c, l->input_h, l->input_w, net->batch, "./data/pool1.txt");
    save_data(l->kernel_weights, 1, l->ksize*l->ksize*l->input_c, l->filters, 1, "./data/cvkw2.txt");
    save_data(l->bias_weights, 1, 1, l->filters, 1, "./data/cvbw2.txt");

    l = &net->layers[3];
    l->input = net->output;
    l->forward(l[0], net[0]);
    net->output = l->output;
    save_data(l->input, l->input_c, l->input_h, l->input_w, net->batch, "./data/convolution2.txt");

    l = &net->layers[4];
    l->input = net->output;
    l->forward(l[0], net[0]);
    net->output = l->output;
    save_data(l->input, l->input_c, l->input_h, l->input_w, net->batch, "./data/pool2.txt");
    save_data(l->kernel_weights, 1, l->ksize*l->ksize*l->input_c, l->filters, 1, "./data/cvkw3.txt");
    save_data(l->bias_weights, 1, 1, l->filters, 1, "./data/cvbw3.txt");

    l = &net->layers[6];
    l->input = net->output;
    l->forward(l[0], net[0]);
    net->output = l->output;
    save_data(l->input, l->input_c, l->input_h, l->input_w, net->batch, "./data/convolution3.txt");
    save_data(l->kernel_weights, 1, l->output_h, l->input_h, 1, "./data/cnkw1.txt");
    save_data(l->bias_weights, 1, 1, l->output_h, 1, "./data/cnbw1.txt");

    l = &net->layers[7];
    l->input = net->output;
    l->forward(l[0], net[0]);
    net->output = l->output;
    save_data(l->input, l->input_c, l->input_h, l->input_w, net->batch, "./data/connect1.txt");
    save_data(l->kernel_weights, 1, l->output_h, l->input_h, 1, "./data/cnkw2.txt");
    save_data(l->bias_weights, 1, 1, l->output_h, 1, "./data/cnbw2.txt");

    l = &net->layers[8];
    l->input = net->output;
    l->forward(l[0], net[0]);
    net->output = l->output;
    save_data(l->input, l->input_c, l->input_h, l->input_w, net->batch, "./data/connect2.txt");

    l = &net->layers[9];
    l->input = net->output;
    l->forward(l[0], net[0]);
    net->output = l->output;
    save_data(l->input, l->input_c, l->input_h, l->input_w, net->batch, "./data/softmax.txt");
    // save_data(l->output, l->output_c, l->output_h, l->output_w, net->batch, "/data/loss.txt");

    l = &net->layers[9];
    l->backward(l[0], net[0]);
    net->delta = l->delta;

    l = &net->layers[8];
    l->backward(l[0], net[0]);
    net->delta = l->delta;

    l = &net->layers[7];
    l->backward(l[0], net[0]);
    net->delta = l->delta;

    l = &net->layers[6];
    l->backward(l[0], net[0]);
    net->delta = l->delta;

    // l = &net->layers[4];
    // l->backward(l[0], net[0]);
    // net->delta = l->delta;

    // l = &net->layers[3];
    // l->backward(l[0], net[0]);
    // net->delta = l->delta;

    // l = &net->layers[2];
    // l->backward(l[0], net[0]);
    // net->delta = l->delta;

    // l = &net->layers[1];
    // l->backward(l[0], net[0]);
    // net->delta = l->delta;

    // l = &net->layers[0];
    // l->backward(l[0], net[0]);
    // net->delta = l->delta;

    for (int b = 0; b < net->batch; ++b){
        for (int k = 0; k < l->input_c; ++k){
            for (int i = 0; i < l->input_h; ++i){
                for (int j = 0; j < l->input_w; ++j){
                    printf("%f ", l->delta[b*l->input_c*l->input_h*l->input_w+k*l->input_h*l->input_w+i*l->input_w+j]);
                }
                printf("\n");
            }
            printf("\n");
            printf("\n");
        }
    }

    // for (int b = 0; b < net->batch; ++b){
    //     for (int k = 0; k < l->input_c; ++k){
    //         for (int i = 0; i < l->input_h; ++i){
    //             for (int j = 0; j < l->input_w; ++j){
    //                 printf("%f ", net->delta[b*l->input_c*l->input_h*l->input_w+k*l->input_h*l->input_w+i*l->input_w+j]);
    //             }
    //             printf("\n");
    //         }
    //         printf("\n");
    //         printf("\n");
    //     }
    // }

    // for (int b = 0; b < net->batch; ++b){
    //     for (int k = 0; k < l->output_c; ++k){
    //         for (int i = 0; i < l->output_h; ++i){
    //             for (int j = 0; j < l->output_w; ++j){
    //                 printf("%f ", l->output[b*l->output_c*l->output_h*l->output_w+k*l->output_h*l->output_w+i*l->output_w+j]);
    //             }
    //             printf("\n");
    //         }
    //         printf("\n");
    //         printf("\n");
    //     }
    // }

    // printf("%d %d %d\n", l->filters, l->input_c, l->ksize);
    // for (int i = 0; i < l->filters; ++i){
    //     for (int j = 0; j < l->input_c; ++j){
    //         for (int k = 0; k < l->ksize*l->ksize; ++k){
    //             printf("%f ", l->kernel_weights[i*l->ksize*l->ksize*l->input_c + j*l->ksize*l->ksize + k]);
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
    // }

    // printf("%d %d %d\n", l->input_h, l->input_w, l->input_c);
    // printf("%d %d %d\n", l->output_h, l->output_w, l->output_c);

    // for (int i = 0; i < l->output_h; ++i){
    //     for (int j = 0; j < l->input_h; ++j){
    //         printf("%f ", l->kernel_weights[i*l->input_h+j]);
    //     }
    //     printf("\n");
    // }

    // for (int i = 0; i < l->filters; ++i){
    //     printf("%f ", l->bias_weights[i]);
    // }
    // printf("\n");

    // for (int i = 0; i < l->output_h; ++i){
    //     printf("%f ", l->bias_weights[i]);
    // }

    // for (int k = 0; k < l.input_c; ++k){
    //     for (int i = 0; i < l.input_h; ++i){
    //         for (int j = 0; j < l.input_w; ++j){
    //             printf("%f ", l.delta[k*l.input_h*l.input_w+i*l.input_w+j]);
    //         }
    //         printf("\n");
    //     }
    //     printf("\n");
    // }
    // forward_network(net[0]);
    // train(net);
    return 0;
}