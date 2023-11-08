#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>

#include "avgpool_layer.h"
#include "connect_layer.h"
#include "convolutional_layer.h"
#include "dropout_layer.h"
#include "im2col.h"
#include "maxpool_layer.h"
#include "normalization_layer.h"
#include "shortcut.h"
#include "softmax.h"
#include "mse_layer.h"
#include "graph.h"
#include "layer.h"
#include "weights_init.h"
#include "dispatch.h"
#include "manager.h"
#include "session.h"

#include "analysis_cmd.h"

#define VERSION "0.1"
#define NUMBER_PTHREAD 1

void *task(void *control)
{
    int *ctrl = (int*)control;
    for (;;){
        if (ctrl[0] == 0){
            fprintf(stderr, "001\n");
            ctrl[0] = 1;
        } else if (ctrl[0] == 2){
            sleep(10);
            break;
        }
    }
    return NULL;
}

// void lumos(int argc, char **argv)
// {
//     if (0 == strcmp(argv[1], "--version") || 0 == strcmp(argv[1], "-v"))
//     {
//         char version[] = VERSION;
//         fprintf(stderr, "Lumos version: v%s\n", version);
//         fprintf(stderr, "This is free software; see the source for copying conditions.  There is NO\n");
//         fprintf(stderr, "warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.\n");
//     }
//     else if (0 == strcmp(argv[1], "--train") || 0 == strcmp(argv[1], "-t"))
//     {
//         if (argc <= 6){
//             fprintf(stderr, "Lumos fatal: Wrong traing option; use option --help for more information\n");
//             return;
//         }
//         char *coretype = argv[2];
//         char *demofile = argv[3];
//         char *datapath = argv[4];
//         char *labelpath = argv[5];
//         char *weightfile = argv[6];
//         Session *sess = load_session_json(demofile, coretype);
//         bind_train_data(sess, datapath);
//         bind_train_label(sess, labelpath);
//         if (0 == strcmp(weightfile, "null")){
//             init_train_scene(sess, NULL);
//         } else {
//             init_train_scene(sess, weightfile);
//         }
//         session_train(sess, "./build/lumos.w");
//     }
//     else if (0 == strcmp(argv[1], "--detect") || 0 == strcmp(argv[1], "-d"))
//     {
//         if (argc <= 6){
//             fprintf(stderr, "Lumos fatal: Wrong traing option; use option --help for more information\n");
//             return;
//         }
//         char *coretype = argv[2];
//         char *demofile = argv[3];
//         char *datapath = argv[4];
//         char *labelpath = argv[5];
//         char *weightfile = argv[6];
//         if (0 == strcmp(weightfile, "null")){
//             fprintf(stderr, "Lumos fatal: NULL weights path; Please input correct weights file path\n");
//             return;
//         }
//         Session *sess = load_session_json(demofile, coretype);
//         bind_test_data(sess, datapath);
//         bind_test_label(sess, labelpath);
//         init_test_scene(sess, weightfile);
//         session_test(sess);
//     }
//     else if (0 == strcmp(argv[1], "--help") || 0 == strcmp(argv[1], "-h"))
//     {
//         fprintf(stderr, "Usage commands:\n");
//         fprintf(stderr, "    --version or -v : To get version\n");
//         fprintf(stderr, "    --path or -p : View The installation path\n");
//         fprintf(stderr, "Thank you for using Lumos deeplearning framework.\n");
//     }
//     else
//     {
//         fprintf(stderr, "Lumos: \e[0;31merror\e[0m: unrecognized command line option '%s'\n", argv[1]);
//         fprintf(stderr, "compilation terminated.\n");
//         fprintf(stderr, "use --help or -h to get available commands.\n");
//     }
// }

int main(int argc, char **argv)
{
    char version[] = VERSION;
    fprintf(stderr, "\e[0;32mThanks For Using The Lighting DeepLearning Framework Lumos\e[0m\n");
    fprintf(stderr, "\e[0;32mLumos version is\e[0m %s\n", version);
    fprintf(stderr, "This is free software; see the source for copying conditions.  There is NO\n");
    fprintf(stderr, "warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.\n");
    pthread_t plumos;
    char *buf = calloc(512, sizeof(char));
    int *control = calloc(1, sizeof(int));
    int status = 0;
    buf[0] = '\0';
    control[0] = 1;
    fprintf(stderr, "\e[0;32m[Lumos Terminal]:\e[0m~lumos$ ");
    for (;;){
        fgets(buf, 512*sizeof(char), stdin);
        if (buf){
            if (0 == strcmp(buf, "version\n")){
                char version[] = VERSION;
                fprintf(stderr, "%s\n", version);
            } else if (0 == strcmp(buf, "help\n")){
                fprintf(stderr, "Usage commands:\n");
                fprintf(stderr, "    version or -v : To get version\n");
                fprintf(stderr, "    help or -h : View The Usage commands\n");
                fprintf(stderr, "    start : To start a Lumos server\n");
                fprintf(stderr, "    exit  : Exit Lumos server safely\n");
                fprintf(stderr, "    timeout: Puase running\n");
                fprintf(stderr, "    stop: Shutdown running\n");
                fprintf(stderr, "    release: release pause\n");
                fprintf(stderr, "    quit: Exit Lumos\n");
                fprintf(stderr, "Thank you for using Lumos deeplearning framework.\n");
            } else if (0 == strcmp(buf, "start\n")){
                if (status == 1){
                    fprintf(stderr, "You have already start a Lumos server!\n");
                } else {
                    fprintf(stderr, "\e[0;32mLumos server starting ....\e[0m ] \n");
                    int st = pthread_create(&plumos, NULL, task, (void*)control);
                    if (st != 0){
                        fprintf(stderr, "\e[0;31mLumos server starting fail\e[0m ] CODE=%d\n", st);
                        exit(-1);
                    }
                    status = 1;
                    fprintf(stderr, "\e[0;32mLumos server start success!\e[0m\n");
                }
            } else if (0 == strcmp(buf, "exit\n")){
                if (status == 1){
                    fprintf(stderr, "\e[0;32mLumos server stop, Please wait ....\e[0m\n");
                    control[0] = 2;
                    pthread_join(plumos, NULL);
                    fprintf(stderr, "\e[0;32mLumos server exits normally\e[0m\n");
                    status = 0;
                } else {
                    fprintf(stderr, "You are not in Lumos server, Use 'start' to get start your Lumos journey.\n");
                }
            } else if (0 == strcmp(buf, "timeout\n")){

            } else if (0 == strcmp(buf, "stop\n")){

            } else if (0 == strcmp(buf, "release\n")){

            } else if (0 == strcmp(buf, "quit\n")){
                int qst = 1;
                if (status == 1){
                    fprintf(stderr, "\e[0;31mLumos server haven't exit, Quit Lumos will lost data!\e[0m\n");
                    fprintf(stderr, "Quit Lumos yes/no: ");
                    buf[0] = '\0';
                    for (;;){
                        fgets(buf, 512*sizeof(char), stdin);
                        if (0 == strcmp(buf, "yes\n")){
                            pthread_cancel(plumos);
                            break;
                        } else if (0 == strcmp(buf, "no\n")){
                            qst = 0;
                            break;
                        } else{
                            fprintf(stderr, "Quit Lumos yes/no: ");
                        }
                    }
                }
                if (qst == 1) break;
            } else {
                if (status == 0){
                    fprintf(stderr, "\e[0;31mWrong option; use option 'help' for more information\e[0m\n");
                } else {
                    int *index = split(buf, ' ');
                    if (index[0] <= 6){
                        fprintf(stderr, "\e[0;31mLumos fatal: Wrong option; use option --help for more information\e[0m\n");
                        continue;
                    }
                    char *type = buf+index[1];
                    char *coretype = buf+index[2];
                    char *demofile = buf+index[3];
                    char *datapath = buf+index[4];
                    char *labelpath = buf+index[5];
                    char *weightfile = buf+index[6];
                    if (0 == strcmp(type, "train")){
                        Session *sess = load_session_json(demofile, coretype);
                        bind_train_data(sess, datapath);
                        bind_train_label(sess, labelpath);
                        if (0 == strcmp(weightfile, "null")){
                            init_train_scene(sess, NULL);
                        } else {
                            init_train_scene(sess, weightfile);
                        }
                        session_train(sess, "./build/lumos.w");
                    } else if (0 == strcmp(type, "detect")){
                        if (0 == strcmp(weightfile, "null")){
                            fprintf(stderr, "Lumos fatal: NULL weights path; Please input correct weights file path\n");
                            continue;
                        }
                        Session *sess = load_session_json(demofile, coretype);
                        bind_test_data(sess, datapath);
                        bind_test_label(sess, labelpath);
                        init_test_scene(sess, weightfile);
                        session_test(sess);
                    }
                }
            }
            if (status == 0) fprintf(stderr, "\e[0;32m[Lumos Terminal]:\e[0m~lumos$ ");
            else if (status == 1) fprintf(stderr, "\e[0;32m[Lumos Terminal]:\e[0m~server$ ");
            buf[0] = '\0';
        }
    }
    return 0;
}
