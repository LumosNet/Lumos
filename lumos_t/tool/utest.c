#include "utest.h"

void test_run(char *msg, int coretype)
{
    char *type = "CPU";
    if (coretype == 1){
        type = "GPU";
    }
    fprintf(stderr, "[ RUN    %s] -----%s-----\n", type, msg);
}

void test_res(int flag, char *msg)
{
    switch (flag)
    {
    case 1:
        fprintf(stderr, "[        \e[0;32mOK\e[0m ] -----%s-----\n\n", msg);
        break;
    case 0:
        fprintf(stderr, "[      \e[0;31mFAIL\e[0m ] -----%s-----\n\n", msg);
        break;
    default:
        break;
    }
}

void test_msg(char *msg)
{
    fprintf(stderr, "  %s\n", msg);
}

void test_msg_pass(char *msg)
{
    fprintf(stderr, "   %s \e[0;32mPASS\e[0m\n", msg);
}

void test_msg_error(char *msg)
{
    fprintf(stderr, "  %s \e[0;31mFAIL\e[0m\n", msg);
}
