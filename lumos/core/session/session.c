#include "session.h"

Session *create_session()
{
    Session *sess = malloc(sizeof(Session));
    return sess;
}

void set_batch(Session *sess, int batch)
{
    sess->batch = batch;
}

void set_subdivision(Session *sess, int subdivision)
{
    sess->subdivision = subdivision;
}

void set_epoch(Session *sess, int epoch)
{
    sess->epoch = epoch;
}

void set_learning_rate(Session *sess, float learning_rate)
{
    sess->learning_rate = learning_rate;
}

void set_data_paths(Session *sess, char *data_paths)
{
    sess->data_paths = data_paths;
}

void set_truth_paths(Session *sess, char *truth_paths)
{
    sess->truth_paths = truth_paths;
}

void session_execute_command(Session *sess, char *command)
{
    int *index = split(command, ' ');
    if (0 == strcmp(command+index[1], "setbatch")){
        int batch = atoi(command+index[2]);
        set_batch(sess, batch);
    } else if (0 == strcmp(command+index[1], "setsubdivision")){
        int subdivision = atoi(command+index[2]);
        set_subdivision(sess, subdivision);
    } else if (0 == strcmp(command+index[1], "setepoch")){
        int epoch = atoi(command+index[2]);
        set_epoch(sess, epoch);
    } else if (0 == strcmp(command+index[1], "setlearningrate")){
        float learning_rate = atof(command+index[2]);
        set_learning_rate(sess, learning_rate);
    } else if (0 == strcmp(command+index[1], "setdatapaths")){
        set_data_paths(sess, command+index[2]);
    } else if (0 == strcmp(command+index[1], "settruthpaths")){
        set_truth_paths(sess, command+index[2]);
    }
}
