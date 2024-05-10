#include "logging.h"

char* getDateTime()
{
	static char nowtime[20];
	time_t rawtime;
	struct tm* ltime;
	time(&rawtime);
	ltime = localtime(&rawtime);
	strftime(nowtime, 20, "%Y-%m-%d %H:%M:%S", ltime);
	return nowtime;
}

void logging_type(int type, FILE *buffer)
{
    switch (type){
        case DEBUG:
            fprintf(buffer, "[DEBUG ");
            break;
        case INFO:
            fprintf(buffer, "[INFO ");
            break;
        case WARNING:
            fprintf(buffer, "[WARNING ");
            break;
        case ERROR:
            fprintf(buffer, "[ERROR ");
            break;
        case FATAL:
            fprintf(buffer, "[FATAL ");
            break;
        default:
            break;
    }
    char* nowtime = getDateTime();
    fprintf(buffer, "%s] ", nowtime);
}

void logging_msg(int type, char *msg, FILE *buffer)
{
    logging_type(type, buffer);
    fprintf(buffer, "%s\n", msg);
}

void logging_data(char *type, void *data, int h, int w, int c, FILE *buffer)
{
    if (0 == strcmp(type, "int")){
        logging_int_data(h, w, c, (int*)data, buffer);
    } else if (0 == strcmp(type, "float")){
        logging_float_data(h, w, c, (float*)data, buffer);
    }
}

void logging_int_data(int h, int w, int c, int *data, FILE *buffer)
{
    for (int k = 0; k < c; ++k){
        for (int i = 0; i < h; ++i){
            for (int j = 0; j < w; ++j){
                fprintf(buffer, "%d ", data[k*h*w+i*w+j]);
            }
            fprintf(buffer, "\n");
        }
        fprintf(buffer, "\n");
    }
}

void logging_float_data(int h, int w, int c, float *data, FILE *buffer)
{
    for (int k = 0; k < c; ++k){
        for (int i = 0; i < h; ++i){
            for (int j = 0; j < w; ++j){
                fprintf(buffer, "%f ", data[k*h*w+i*w+j]);
            }
            fprintf(buffer, "\n");
        }
        fprintf(buffer, "\n");
    }
}
