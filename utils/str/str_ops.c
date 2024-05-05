#include "str_ops.h"

void strip(char *line, char c)
{
    int len = strlen(line);
    int i;
    int offset = 0;
    for (i = 0; i < len; ++i)
    {
        char x = line[i];
        if (x == ' ' || x == '\t' || x == '\n' || x == '\r' || x == c)
            ++offset;
        else
            line[i - offset] = x;
    }
    line[len - offset] = '\0';
}

int *split(char *line, char c)
{
    int len = strlen(line);
    int n = 0;
    int head = 0;
    int i = 0;
    for (i = 0; i < len; ++i)
    {
        if (line[i] == c)
        {
            if (i != head)
                n += 1;
            head = i + 1;
        }
    }
    if (i != head)
        n += 1;
    int *index = malloc((n+1)*sizeof(int));
    index[0] = n;
    head = 0;
    n = 0;
    for (i = 0; i < len; ++i)
    {
        if (line[i] == c)
        {
            if (i != head)
            {
                index[n+1] = head;
                n += 1;
            }
            head = i + 1;
            line[i] = '\0';
        }
    }
    if (i != head)
    {
        index[n+1] = head;
    }
    return index;
}

void padding_string(char *space, char *str, int index)
{
    strcpy(space+index, str);
    space[index+strlen(str)] = '\0';
}

char *int2str(int x)
{
    char *res;
    int a = x;
    int b;
    int n = 1;
    while (1)
    {
        if (a != 0)
            n += 1;
        else
            break;
        a = a / 10;
    }
    a = x;
    if (x < 0)
    {
        a = -x;
        n += 1;
    }
    res = malloc(n * sizeof(char));
    for (int i = 0; i < n; ++i)
    {
        b = a % 10;
        a = a / 10;
        res[n - i - 2] = inten2str(b)[0];
    }
    if (x < 0)
        res[0] = '-';
    res[n - 1] = '\0';
    return res;
}

char *inten2str(int x)
{
    char *res = malloc(2 * sizeof(char));
    if (x == 0)
        res[0] = '0';
    else if (x == 1)
        res[0] = '1';
    else if (x == 2)
        res[0] = '2';
    else if (x == 3)
        res[0] = '3';
    else if (x == 4)
        res[0] = '4';
    else if (x == 5)
        res[0] = '5';
    else if (x == 6)
        res[0] = '6';
    else if (x == 7)
        res[0] = '7';
    else if (x == 8)
        res[0] = '8';
    else if (x == 9)
        res[0] = '9';
    res[1] = '\0';
    return res;
}