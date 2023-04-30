#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "lumos.h"

#define VERSION "0.1"

void lumos(int argc, char **argv)
{
    if (argc <= 1){
        fprintf(stderr, "Lumos fatal: No input option; use option --help for more information");
        return;
    }
    if (0 == strcmp(argv[1], "--version") || 0 == strcmp(argv[1], "-v"))
    {
        char version[] = VERSION;
        fprintf(stderr, "Lumos version: v%s\n", version);
        fprintf(stderr, "This is free software; see the source for copying conditions.  There is NO\n");
        fprintf(stderr, "warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.\n");
    }
    else if (0 == strcmp(argv[1], "--train") || 0 == strcmp(argv[1], "-t"))
    {
        if (argc <= 6){
            fprintf(stderr, "Lumos fatal: Wrong traing option; use option --help for more information\n");
            return;
        }
        char *coretype = argv[2];
        char *demofile = argv[3];
        char *datapath = argv[4];
        char *labelpath = argv[5];
        char *weightfile = argv[6];
        Session *sess = load_session_json(demofile, coretype);
        bind_train_data(sess, datapath);
        bind_train_label(sess, labelpath);
        if (0 == strcmp(weightfile, "null")){
            init_train_scene(sess, NULL);
        } else {
            init_train_scene(sess, weightfile);
        }
        session_train(sess, "./build/lumos.w");
    }
    else if (0 == strcmp(argv[1], "--detect") || 0 == strcmp(argv[1], "-d"))
    {
        if (argc <= 6){
            fprintf(stderr, "Lumos fatal: Wrong traing option; use option --help for more information\n");
            return;
        }
        char *coretype = argv[2];
        char *demofile = argv[3];
        char *datapath = argv[4];
        char *labelpath = argv[5];
        char *weightfile = argv[6];
        if (0 == strcmp(weightfile, "null")){
            fprintf(stderr, "Lumos fatal: NULL weights path; Please input correct weights file path\n");
            return;
        }
        Session *sess = load_session_json(demofile, coretype);
        bind_test_data(sess, datapath);
        bind_test_label(sess, labelpath);
        init_test_scene(sess, weightfile);
        session_test(sess);
    }
    else if (0 == strcmp(argv[1], "--help") || 0 == strcmp(argv[1], "-h"))
    {
        fprintf(stderr, "Usage commands:\n");
        fprintf(stderr, "    --version or -v : To get version\n");
        fprintf(stderr, "    --path or -p : View The installation path\n");
        fprintf(stderr, "Thank you for using Lumos deeplearning framework.\n");
    }
    else
    {
        fprintf(stderr, "Lumos: \e[0;31merror\e[0m: unrecognized command line option '%s'\n", argv[1]);
        fprintf(stderr, "compilation terminated.\n");
        fprintf(stderr, "use --help or -h to get available commands.\n");
    }
}

int main(int argc, char **argv)
{
    lumos(argc, argv);
    return 0;
}
