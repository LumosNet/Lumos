#include "utils.h"

char *fgetl(FILE *fp)
{
    if(feof(fp)) return 0;
    char *line = malloc(512*sizeof(char));
    fgets(line, 512, fp);
    int len = strlen(line);
    line[len-1] = '\0';
    return line;
}

void strip(char *line)
{
    int len = strlen(line);
    int i;
    int offset = 0;
    for (i = 0; i < len; ++i){
        char c = line[i];
        if (c == ' ' || c == '\t' || c == '\n') ++offset;
        else line[i-offset] = c;
    }
    line[len-offset] = '\0';
}

char **split(char *line, char c, int *num)
{
    int len = strlen(line);
    int n = 0;
    int head = 0;
    int i = 0;
    for (i = 0; i < len; ++i){
        if (line[i] == c){
            if (i != head) n += 1;
            head = i+1;
        }
    }
    if (i != head) n += 1;
    char **res = malloc(n*sizeof(char *));
    num[0] = n;
    head = 0;
    n = 0;
    for (i = 0; i < len; ++i){
        if (line[i] == c){
            if (i != head) {
                char *A = malloc((i-head+1)*sizeof(char));
                memcpy(A, line+head, (i-head)*sizeof(char));
                A[i-head] = '\0';
                res[n] = A;
                n += 1;
            }
            head = i+1;
        }
    }
    if (i != head) {
        char *A = malloc((i-head+1)*sizeof(char));
        memcpy(A, line+head, (i-head)*sizeof(char));
        A[i-head] = '\0';
        res[n] = A;
    }
    return res;
}
