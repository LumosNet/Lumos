#include "log.h"

void logging(FILE *buf, int level, char *msg, int prt)
{
    if (!buf){
        fprintf(stderr, "\e[0;32mLogging Error, Logging Path Not Found!\e[0m\n");
    }
    switch (level){
        case DEBUG:
            logging_debug(buf, msg, prt);
            break;
        case INFO:
            logging_info(buf, msg, prt);
            break;
        case WARNING:
            logging_warning(buf, msg, prt);
            break;
        case ERROR:
            logging_error(buf, msg, prt);
            break;
        case CRITICAL:
            logging_critical(buf, msg, prt);
            break;
        default:
            logging_warning(buf, "Unknown Logging Level, MSG Logging As INFO", 1);
            logging_info(buf, msg, prt);
            break;
    }
}

void logging_debug(FILE *buf, char *msg, int prt)
{
    fprintf(buf, "[%s %s %s   ] %s\n", __DATE__, __TIME__, "DEBUG", msg);
    if (prt){
        fprintf(stderr, "[%s %s \e[0;34m%s\e[0m   ] \e[0;34m%s\e[0m\n", __DATE__, __TIME__, "DEBUG", msg);
    }
}

void logging_info(FILE *buf, char *msg, int prt)
{
    fprintf(buf, "[%s %s %s    ] %s\n", __DATE__, __TIME__, "INFO", msg);
    if (prt){
        fprintf(stderr, "[%s %s %s    ] %s\n", __DATE__, __TIME__, "INFO", msg);
    }
}

void logging_warning(FILE *buf, char *msg, int prt)
{
    fprintf(buf, "[%s %s %s ] %s\n", __DATE__, __TIME__, "WARNING", msg);
    if (prt){
        fprintf(stderr, "[%s %s \e[0;33m%s\e[0m ] \e[0;33m%s\e[0m\n", __DATE__, __TIME__, "WARNING", msg);
    }
}

void logging_error(FILE *buf, char *msg, int prt)
{
    fprintf(buf, "[%s %s %s   ] %s\n", __DATE__, __TIME__, "ERROR", msg);
    if (prt){
        fprintf(stderr, "[%s %s \e[0;31m%s\e[0m   ] \e[0;31m%s\e[0m\n", __DATE__, __TIME__, "ERROR", msg);
    }
}

void logging_critical(FILE *buf, char *msg, int prt)
{
    fprintf(buf, "[%s %s %s] %s\n", __DATE__, __TIME__, "CRITICAL", msg);
    if (prt){
        fprintf(stderr, "[%s %s \e[1;31m%s\e[0m] \e[1;31m%s\e[0m\n", __DATE__, __TIME__, "CRITICAL", msg);
    }
}