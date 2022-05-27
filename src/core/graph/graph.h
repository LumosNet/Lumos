#ifndef NETWORK_H
#define NETWORK_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct graph{
    int n;
    int height;
    int channel;
    float *input;
    float *output;
    float *delta;
    Layer *layers;

    int kinds;
    char **label;
    Label *labels;

    int num;
    char **data;
} graph, Graph;

Network *load_network(char *cfg);
void train(Network *net, int x);
void test(Network *net, char *test_png, char *test_label);
void init_network(Network *net, char *data_file, char *weight_file);
void forward_network(Network *net);
void backward_network(Network *net);

Network *create_network(LayerParams *p, int size);

#ifdef __cplusplus
}
#endif

#endif