#include "test.h"

void test_run(char *msg)
{
    fprintf(stderr, "[ RUN       ] %s\n", msg);
}

void test_res(int flag, char *msg)
{
    switch (flag){
        case SUCCESS:
            fprintf(stderr, "[        \e[0;32mOK\e[0m ] %s\n", msg); break;
        case ERROR_NUM:
            fprintf(stderr, "[      \e[0;31mFAIL\e[0m ] %s\n", msg); break;
        case ERROR_MEM_APPLY:
            fprintf(stderr, "[      \e[0;31mFAIL\e[0m ] %s\n", msg); break;
        case ERROR_MEM_SET:
            fprintf(stderr, "[      \e[0;31mFAIL\e[0m ] %s\n", msg); break;
        case ERROR_PARAM_INIT:
            fprintf(stderr, "[      \e[0;31mFAIL\e[0m ] %s\n", msg); break;
        default:
            break;
    }
}

void test_msg(char *msg)
{
    fprintf(stderr, "  %s\n", msg);
}