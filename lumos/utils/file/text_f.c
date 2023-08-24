#include "text_f.h"

void fputl(FILE *fp, char *line)
{
    fputs(line, fp);
    fputc('\n', fp);
}

void fputls(FILE *fp, char **lines, int n)
{
    for (int i = 0; i < n; ++i)
    {
        fputl(fp, lines[i]);
    }
}

char *fget(char *file)
{
    FILE *fp = fopen(file, "r");
    if (fp == NULL) {
        fprintf(stderr, "\nfopen error: %s is not exist\n", file);
        abort();
    }
    fseek(fp, 0, SEEK_END);
    int file_size = ftell(fp);
    char *tmp = (char*)malloc(file_size * sizeof(char));
    memset(tmp, '\0', file_size * sizeof(char));
    fseek(fp, 0, SEEK_SET);
    fread(tmp, sizeof(char), file_size, fp);
    fclose(fp);
    return tmp;
}

int lines_num(char *tmp)
{
    int res = 0;
    int len = strlen(tmp);
    int flag = 0;
    for (int i = 0; i < len; ++i){
        if (tmp[i] == '\r' || tmp[i] == '\n'){
            if (flag) res += 1;
            flag = 0;
        } else if (tmp[i] == ' ') {
            continue;
        } else {
            flag = 1;
        }
    }
    return res;
}

int format_str_line(char *tmp)
{
    int res = 0;
    int len = strlen(tmp);
    int flag = 0;
    int meet = 0;
    int offset = 0;
    int n = len;
    for (int i = 0; i < n; ++i){
        if (tmp[i] == '\r' || tmp[i] == '\n') flag = 1;
        if (tmp[i] == '\r' || tmp[i] == '\n' || tmp[i] == '\t' || tmp[i] == ' '){
            tmp[i] = '\0';
            offset += 1;
        } else {
            if (offset != 0){
                if (flag && meet){
                    char *head = tmp+i-offset;
                    head[0] = '\0';
                    res += 1;
                    offset -= 1;
                }
                memcpy(tmp+i-offset, tmp+i, (len-i)*sizeof(char));
                n -= offset;
                i -= offset;
            }
            offset = 0;
            flag = 0;
            meet = 1;
        }
    }
    if (meet) res += 1;
    return res;
}

int format_str_space(char *tmp)
{
    int res = 0;
    int len = strlen(tmp);
    int offset = 0;
    int meet = 0;
    int flag = 0;
    int n = len;
    for (int i = 0; i < n; ++i){
        if (tmp[i] == ' ' || tmp[i] == '\n' || tmp[i] == '\t' || tmp[i] == '\r'){
            tmp[i] = '\0';
            flag = 1;
            offset += 1;
        } else {
            if (offset != 0){
                if (flag && meet) {
                    char *head = tmp+i-offset;
                    head[0] = '\0';
                    res += 1;
                    offset -= 1;
                }
                memcpy(tmp+i-offset, tmp+i, (len-i)*sizeof(char));
                n -= offset;
                i -= offset;
            }
            meet = 1;
            offset = 0;
            flag = 0;
        }
    }
    if (meet) res += 1;
    return res;
}
