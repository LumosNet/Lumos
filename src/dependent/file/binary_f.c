#include "binary_f.h"


int write_as_binary(FILE *fp, float* arry, size_t size)
{
    size_t res = fwrite(arry, sizeof(float), size, fp);
    if (res != size)
    {
        int error = ferror(fp);
        if (!error)
        {
            return FILEEND;
        }
        return error;
    }
    return SUCESS;
}


int read_part_as_bin(FILE *fp, float**arry, size_t *arrsize, size_t start,\
     size_t end)
{
    int state = fseek(fp, start*sizeof(float), SEEK_SET);
    if(state)
    {
        return FSEEKERROR;
    }
    size_t length = end - start;
    float *arr = (float*)malloc(sizeof(float)*(end-start));
    *arry = arr;
    int res = fread(arr, sizeof(float), end-start, fp);
    
    if (res != (end-start))
    {
        int error = ferror(fp);
        if (!error)
        {
            return FILEEND;
        }
        *arrsize = res;
        return error;
    }
    *arrsize = end - start;
    return SUCESS;
}



int read_as_binary(FILE *fp, int mode, size_t *scop, float**arry, size_t *arrsize)
{
    int res;
    size_t end;
    size_t start;
    if(!scop && mode != 0)
    {
        return ARGSERROR;
    }

    switch (mode)
    {
        case READALL:
            if(!fseek(fp, 0, SEEK_END))
            {
                return FSEEKERROR;
            }
            end = ftell(fp)/sizeof(float);
            return read_part_as_bin(fp, arry, arrsize, 0, end);
            break;
        case READPART:
            start = scop[0];
            end   = scop[1];
            if(end<=start)
            {
                return RANGEERROR;
            }
            return read_part_as_bin(fp, arry, arrsize, start, end);
            break;
        case READSERIAL:
            end  = scop[0];
            return read_part_as_bin(fp, arry, arrsize, 0, end);
            break;
        default:
            return  MODERROR;
            break;
   } 
}