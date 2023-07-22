#ifndef TEST_MSG_H
#define TEST_MSG_H

#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tmsg TMsg;

struct tmsg{
    char *interface;
    TMsg *next;
};

TMsg *add_t_msg(char *msg, TMsg *t_msg);

#ifdef __cplusplus
}
#endif

#endif