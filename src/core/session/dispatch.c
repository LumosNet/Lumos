#include "dispatch.h"

void session_run(Session sess)
{
    for (int i = 0; i < sess.epoch; ++i){
        int sub_epoch = (int)(sess.batch / sess.subdivision);
        for (int j = 0; j < sub_epoch; ++j){
            float learning_rate = sess.learning_rate / sess.batch;
            load_data(sess, i*sess.batch+j*sess.subdivision, sess.subdivision);
            
        }
    }
}

void forward_session(Session sess);
void backward_session(Session sess);