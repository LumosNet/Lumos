#ifndef COMMANDS_H
#define COMMANDS_H

#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif

#define VERSIONCMD  0
#define HELPCMD     1
#define STARTCMD    2
#define EXITCMD     3
#define TIMEOUTCMD  4
#define STOPCMD     5
#define RELEASECMD  6
#define QUITCMD     7

int do_version_cmd(char *version);
int do_help_cmd();
int do_start_cmd(pthread_t lumos, int status);
int do_exit_cmd(pthread_t lumos, int status, int *ctrl);
int do_timeout_cmd(pthread_t lumos, int status, int *ctrl);
int do_stop_cmd(pthread_t lumos, int status, int *ctrl);
int do_release_cmd(pthread_t lumos, int status, int *ctrl);
int do_quit(pthread_t lumos, int status, int *ctrl);

#ifdef __cplusplus
}
#endif
#endif
