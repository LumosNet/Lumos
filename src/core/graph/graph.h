#ifndef NETWORK_H
#define NETWORK_H

#ifdef __cplusplus
extern "C" {
#endif

typedef struct graph{
    int layer_num;
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

    CFG *cfg;
} graph, Graph;

Network *load_network(char *cfg);
void train(Network *net, int x);
void test(Network *net, char *test_png, char *test_label);
void init_network(Network *net, char *data_file, char *weight_file);
void forward_network(Network *net);
void backward_network(Network *net);

Network *create_network(LayerParams *p, int size);

Graph *load_graph(char *cfg);
Graph *create_graph(CFGPiece *p, int layer_n);

#ifdef __cplusplus
}
#endif

#endif