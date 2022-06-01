#include "manager.h"

void init_graph(Session sess, Graph g)
{
    Layer l;
    for (int i = 0; i < g.layer_num; ++i){
        l = g.layers[i];
        switch (l.type){
            case AVGPOOL:
                init_avgpool_layer(sess, l); break;
            case CONNECT:
                init_connect_layer(sess, l); break;
            case CONVOLUTIONAL:
                init_convolutional_layer(sess, l); break;
            case IM2COL:
                init_im2col_layer(sess, l); break;
            case MAXPOOL:
                init_maxpool_layer(sess, l); break;
            case MSE:
                init_mse_layer(sess, l); break;
        }
    }
}

void init_avgpool_layer(Session sess, Layer l)
{
    l.input_h = h;
    l.input_w = w;
    l.input_c = c;
}

void init_connect_layer(Session sess, Layer l);
void init_convolutional_layer(Session sess, Layer l);
void init_im2col_layer(Session sess, Layer l);
void init_maxpool_layer(Session sess, Layer l);
void init_mse_layer(Session sess, Layer l);

void restore_graph(Session sess, Layer g)
{
    Layer l;
    for (int i = 0; i < g.layer_num; ++i){
        l = g.layers[i];
        switch (l.type){
            case AVGPOOL:
                restore_avgpool_layer(sess, l); break;
            case CONNECT:
                restore_connect_layer(sess, l); break;
            case CONVOLUTIONAL:
                restore_convolutional_layer(sess, l); break;
            case IM2COL:
                restore_im2col_layer(sess, l); break;
            case MAXPOOL:
                restore_maxpool_layer(sess, l); break;
            case MSE:
                restore_mse_layer(sess, l); break;
        }
    }
}

void restore_avgpool_layer(Session sess, Layer l);
void restore_connect_layer(Session sess, Layer l);
void restore_convolutional_layer(Session sess, Layer l);
void restore_im2col_layer(Session sess, Layer l);
void restore_maxpool_layer(Session sess, Layer l);
void restore_mse_layer(Session sess, Layer l);