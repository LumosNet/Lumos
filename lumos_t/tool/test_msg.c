#include "test_msg.h"

TMsg *add_t_msg(char *msg, TMsg *t_msg)
{
    TMsg *t = malloc(sizeof(TMsg));
    t_msg->next = t;
    t->interface = msg;
    t->next = NULL;
    return t;
}
