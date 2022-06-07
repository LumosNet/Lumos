#include "dispatch.h"

void session_run(Session sess)
{
    Graph graph = sess.graph;
    Layer *layers = graph.layers;
    Layer l;
    for (int i = 0; i < sess.epoch; ++i){
        int sub_epoch = (int)(sess.batch / sess.subdivision);
        for (int j = 0; j < sub_epoch; ++j){
            load_data(sess, i*sess.batch+j*sess.subdivision, sess.subdivision);
            forward_session(sess);
            backward_session(sess);
        }
    }
}

void forward_session(Session sess)
{
    Graph graph = sess.graph;
    Layer *layers = graph.layers;
    Layer l;
    float *input = sess.input;
    for (int i = 0; i < graph.layer_num; ++i){
        l = layers[i];
        l.input = input;
        l.forward(l);
        input = l.output;
    }
}

void backward_session(Session sess)
{
    Graph graph = sess.graph;
    Layer *layers = graph.layers;
    Layer l;
    float *delta = NULL;
    for (int i = graph.layer_num-1; i >= 0; --i){
        l = layers[i];
        l.backward(l, delta);
        delta = l.delta;
    }
}

void update_session(Session sess)
{
    Graph graph = sess.graph;
    Layer *layers = graph.layers;
    Layer l;
    float rate = sess.learning_rate / sess.batch;
    float *delta = NULL;
    for (int i = graph.layer_num-1; i >= 0; --i){
        l = layers[i];
        l.update(l, rate, delta);
        delta = l.delta;
    }
}
