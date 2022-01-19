#include "lumos.h"
#include "network.h"
#include "utils.h"
#include "data.h"
#include "utils.h"

#include "debug.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv)
{
    Network *net = load_network("./cfg/test.cfg");
    // init_network(net, "./mnist/mnist.data", NULL);
    // load_train_data(net, 0);
    // train(net);

    // char *res = int2str(50);
    // printf("%s\n", res);

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

    // Layer *l = &net->layers[0];
    // l->input = net->output;
    // l->forward(l[0], net[0]);
    // net->output = l->output;

    // l = &net->layers[1];
    // l->input = net->output;
    // l->forward(l[0], net[0]);
    // net->output = l->output;

    // l = &net->layers[2];
    // l->input = net->output;
    // l->forward(l[0], net[0]);
    // net->output = l->output;

    // l = &net->layers[3];
    // l->input = net->output;
    // l->forward(l[0], net[0]);
    // net->output = l->output;

    // l = &net->layers[4];
    // l->input = net->output;
    // l->forward(l[0], net[0]);
    // net->output = l->output;

    // l = &net->layers[6];
    // l->input = net->output;
    // l->forward(l[0], net[0]);
    // net->output = l->output;

    // l = &net->layers[7];
    // l->input = net->output;
    // l->forward(l[0], net[0]);
    // net->output = l->output;

    // l = &net->layers[8];
    // l->input = net->output;
    // l->forward(l[0], net[0]);
    // net->output = l->output;

    // l = &net->layers[9];
    // l->input = net->output;
    // l->forward(l[0], net[0]);
    // net->output = l->output;

    // l = &net->layers[9];
    // l->backward(l[0], net[0]);
    // net->delta = l->delta;

    // l = &net->layers[8];
    // l->backward(l[0], net[0]);
    // net->delta = l->delta;

    // l = &net->layers[7];
    // l->backward(l[0], net[0]);
    // net->delta = l->delta;

    // l = &net->layers[6];
    // l->backward(l[0], net[0]);
    // net->delta = l->delta;

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

    // for (int b = 0; b < net->batch; ++b){
    //     for (int k = 0; k < l->input_c; ++k){
    //         for (int i = 0; i < l->input_h; ++i){
    //             for (int j = 0; j < l->input_w; ++j){
    //                 printf("%f ", l->delta[b*l->input_c*l->input_h*l->input_w+k*l->input_h*l->input_w+i*l->input_w+j]);
    //             }
    //             printf("\n");
    //         }
    //         printf("\n");
    //         printf("\n");
    //     }
    // }

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