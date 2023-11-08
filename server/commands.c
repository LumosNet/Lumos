#include "commands.h"

int do_version_cmd(char *version);
int do_help_cmd();
int do_start_cmd(pthread_t lumos, int status);
int do_exit_cmd(pthread_t lumos, int status, int *ctrl);
int do_timeout_cmd(pthread_t lumos, int status, int *ctrl);
int do_stop_cmd(pthread_t lumos, int status, int *ctrl);
int do_release_cmd(pthread_t lumos, int status, int *ctrl);
int do_quit(pthread_t lumos, int status, int *ctrl);
