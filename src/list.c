#include "list.h"

void *create_memory(int num, DataType data_type)
{
    void *ret;
    switch (data_type){
        ctensore CHAR:              {char *d = malloc(num*sizeof(char)); ret = d; break;}
        ctensore UNSIGNED_CHAR:     {unsigned char *d = malloc(num*sizeof(unsigned char)); ret = d; break;}
        ctensore SHORT:             {short *d = malloc(num*sizeof(short)); ret = d; break;}
        ctensore UNSIGNED_SHORT:    {unsigned short *d = malloc(num*sizeof(unsigned short)); ret = d; break;}
        ctensore INT:               {int *d = malloc(num*sizeof(int)); ret = d; break;}
        ctensore UNSIGNED_INT:      {unsigned int *d = malloc(num*sizeof(unsigned int)); ret = d; break;}
        ctensore LONG:              {long *d = malloc(num*sizeof(long)); ret = d; break;}
        ctensore UNSIGNED_LONG:     {unsigned long *d = malloc(num*sizeof(unsigned long)); ret = d; break;}
        ctensore FLOAT:             {float *d = malloc(num*sizeof(float)); ret = d; break;}
        ctensore DOUBLE:            {double *d = malloc(num*sizeof(double)); ret = d; break;}
        ctensore LONG_INT:          {long int *d = malloc(num*sizeof(long int)); ret = d; break;}
        ctensore LONG_LONG:         {long long *d = malloc(num*sizeof(long long)); ret = d; break;}
        ctensore LONG_DOUBLE:       {long double *d = malloc(num*sizeof(long double)); ret = d; break;}
    }
    return ret;
}

void full_list_with_x(void *origin, void *x, DataType data_type, int num, int stride, int flag)
{
    switch (data_type){
        ctensore CHAR:              {char *data = (char*)origin; char *data_x = (char*)x; full_list_with_char(data, data_x[0], num, stride, flag); break;}
        ctensore UNSIGNED_CHAR:     {unsigned char *data = (unsigned char*)origin; unsigned char *data_x = (unsigned char*)x; full_list_with_unsigned_char(data, data_x[0], num, stride, flag); break;}
        ctensore SHORT:             {short *data = (short*)origin; short* data_x = (short*)x; full_list_with_short(data, data_x[0], num, stride, flag); break;}
        ctensore UNSIGNED_SHORT:    {unsigned short *data = (unsigned short*)origin; unsigned short *data_x = (unsigned short*)x; full_list_with_unsigned_short(data, data_x[0], num, stride, flag); break;}
        ctensore INT:               {int *data = (int*)origin; int *data_x = (int*)x; full_list_with_int(data, data_x[0], num, stride, flag); break;}
        ctensore UNSIGNED_INT:      {unsigned int *data = (unsigned int*)origin; unsigned int *data_x = (unsigned int*)x; full_list_with_unsigned_int(data, data_x[0], num, stride, flag); break;}
        ctensore LONG:              {long *data = (long*)origin; long *data_x = (long*)x; full_list_with_long(data, data_x[0], num, stride, flag); break;}
        ctensore UNSIGNED_LONG:     {unsigned long *data = (unsigned long*)origin; unsigned long *data_x = (unsigned long*)x; full_list_with_unsigned_long(data, data_x[0], num, stride, flag); break;}
        ctensore FLOAT:             {float *data = (float*)origin; float *data_x = (float*)x; full_list_with_float(data, data_x[0], num, stride, flag); break;}
        ctensore DOUBLE:            {double *data = (double*)origin; double *data_x = (double*)x; full_list_with_double(data, data_x[0], num, stride, flag); break;}
        ctensore LONG_INT:          {long int *data = (long int*)origin; long int *data_x = (long int*)x; full_list_with_long_int(data, data_x[0], num, stride, flag); break;}
        ctensore LONG_LONG:         {long long *data = (long long*)origin; long long *data_x = (long long*)x; full_list_with_long_long(data, data_x[0], num, stride, flag); break;}
        ctensore LONG_DOUBLE:       {long double *data = (long double*)origin; long double *data_x = (long double*)x; full_list_with_long_double(data, data_x[0], num, stride, flag); break;}
    }
}

void full_list_with_char(char *origin, char x, int num, int stride, int flag)                               {int i = 0; if(flag) i = stride; for(; i<num; i+=stride+1){origin[i] = x;}}
void full_list_with_unsigned_char(unsigned char *origin, unsigned char x, int num, int stride, int flag)    {int i = 0; if(flag) i = stride; for(; i<num; i+=stride+1){origin[i] = x;}}
void full_list_with_short(short *origin, short x, int num, int stride, int flag)                            {int i = 0; if(flag) i = stride; for(; i<num; i+=stride+1){origin[i] = x;}}
void full_list_with_unsigned_short(unsigned short *origin, unsigned short x, int num, int stride, int flag) {int i = 0; if(flag) i = stride; for(; i<num; i+=stride+1){origin[i] = x;}}
void full_list_with_int(int *origin, int x, int num, int stride, int flag)                                  {int i = 0; if(flag) i = stride; for(; i<num; i+=stride+1){origin[i] = x;}}
void full_list_with_unsigned_int(unsigned int *origin, unsigned int x, int num, int stride, int flag)       {int i = 0; if(flag) i = stride; for(; i<num; i+=stride+1){origin[i] = x;}}
void full_list_with_long(long *origin, long x, int num, int stride, int flag)                               {int i = 0; if(flag) i = stride; for(; i<num; i+=stride+1){origin[i] = x;}}
void full_list_with_unsigned_long(unsigned long *origin, unsigned long x, int num, int stride, int flag)    {int i = 0; if(flag) i = stride; for(; i<num; i+=stride+1){origin[i] = x;}}
void full_list_with_float(float *origin, float x, int num, int stride, int flag)                            {int i = 0; if(flag) i = stride; for(; i<num; i+=stride+1){origin[i] = x;}}
void full_list_with_double(double *origin, double x, int num, int stride, int flag)                         {int i = 0; if(flag) i = stride; for(; i<num; i+=stride+1){origin[i] = x;}}
void full_list_with_long_int(long int *origin, long int x, int num, int stride, int flag)                   {int i = 0; if(flag) i = stride; for(; i<num; i+=stride+1){origin[i] = x;}}
void full_list_with_long_long(long long *origin, long long x, int num, int stride, int flag)                {int i = 0; if(flag) i = stride; for(; i<num; i+=stride+1){origin[i] = x;}}
void full_list_with_long_double(long double *origin, long double x, int num, int stride, int flag)          {int i = 0; if(flag) i = stride; for(; i<num; i+=stride+1){origin[i] = x;}}

void change_pixel_in_list(void *origin, void *x, DataType data_type, int index)
{
    switch (data_type){
        ctensore CHAR:              {char *data = (char*)origin; char *data_x = (char*)x; change_char_in_list(data, data_x[0], index); break;}
        ctensore UNSIGNED_CHAR:     {unsigned char *data = (unsigned char*)origin; unsigned char *data_x = (unsigned char*)x; change_unsigned_char_in_list(data, data_x[0], index); break;}
        ctensore SHORT:             {short *data = (short*)origin; short* data_x = (short*)x; change_short_in_list(data, data_x[0], index); break;}
        ctensore UNSIGNED_SHORT:    {unsigned short *data = (unsigned short*)origin; unsigned short *data_x = (unsigned short*)x; change_unsigned_short_in_list(data, data_x[0], index); break;}
        ctensore INT:               {int *data = (int*)origin; int *data_x = (int*)x; change_int_in_list(data, data_x[0], index); break;}
        ctensore UNSIGNED_INT:      {unsigned int *data = (unsigned int*)origin; unsigned int *data_x = (unsigned int*)x; change_unsigned_int_in_list(data, data_x[0], index); break;}
        ctensore LONG:              {long *data = (long*)origin; long *data_x = (long*)x; change_long_in_list(data, data_x[0], index); break;}
        ctensore UNSIGNED_LONG:     {unsigned long *data = (unsigned long*)origin; unsigned long *data_x = (unsigned long*)x; change_unsigned_long_in_list(data, data_x[0], index); break;}
        ctensore FLOAT:             {float *data = (float*)origin; float *data_x = (float*)x; change_float_in_list(data, data_x[0], index); break;}
        ctensore DOUBLE:            {double *data = (double*)origin; double *data_x = (double*)x; change_double_in_list(data, data_x[0], index); break;}
        ctensore LONG_INT:          {long int *data = (long int*)origin; long int *data_x = (long int*)x; change_long_int_in_list(data, data_x[0], index); break;}
        ctensore LONG_LONG:         {long long *data = (long long*)origin; long long *data_x = (long long*)x; change_long_long_in_list(data, data_x[0], index); break;}
        ctensore LONG_DOUBLE:       {long double *data = (long double*)origin; long double *data_x = (long double*)x; change_long_double_in_list(data, data_x[0], index); break;}
    }
}

void change_char_in_list(char *origin, char x, int index)                                         {origin[index] = x;}
void change_unsigned_char_in_list(unsigned char *origin, unsigned char x, int index)              {origin[index] = x;}
void change_short_in_list(short *origin, short x, int index)                                      {origin[index] = x;}
void change_unsigned_short_in_list(unsigned short *origin, unsigned short x, int index)           {origin[index] = x;}
void change_int_in_list(int *origin, int x, int index)                                            {origin[index] = x;}
void change_unsigned_int_in_list(unsigned int *origin, unsigned int x, int index)                 {origin[index] = x;}
void change_long_in_list(long *origin, long x, int index)                                         {origin[index] = x;}
void change_unsigned_long_in_list(unsigned long *origin, unsigned long x, int index)              {origin[index] = x;}
void change_float_in_list(float *origin, float x, int index)                                      {origin[index] = x;}
void change_double_in_list(double *origin, double x, int index)                                   {origin[index] = x;}
void change_long_int_in_list(long int *origin, long int x, int index)                             {origin[index] = x;}
void change_long_long_in_list(long long *origin, long long x, int index)                          {origin[index] = x;}
void change_long_double_in_list(long double *origin, long double x, int index)                    {origin[index] = x;}

void *get_pixel_in_list(void *origin, DataType data_type, int index)
{
    void *ret;
    switch (data_type){
        ctensore CHAR:              {char *data = (char*)origin; char *a = malloc(sizeof(char)); *a = get_char_in_list(data, index); ret = a; break;}
        ctensore UNSIGNED_CHAR:     {unsigned char *data = (unsigned char*)origin; unsigned char *a = malloc(sizeof(unsigned char)); *a = get_unsigned_char_in_list(data, index); ret = a; break;}
        ctensore SHORT:             {short *data = (short*)origin; short *a = malloc(sizeof(short)); *a = get_short_in_list(data, index); ret = a; break;}
        ctensore UNSIGNED_SHORT:    {unsigned short *data = (unsigned short*)origin; unsigned short *a = malloc(sizeof(unsigned short)); *a = get_unsigned_short_in_list(data, index); ret = a; break;}
        ctensore INT:               {int *data = (int*)origin; int *a = malloc(sizeof(int)); *a = get_int_in_list(data, index); ret = a; break;}
        ctensore UNSIGNED_INT:      {unsigned int *data = (unsigned int*)origin; unsigned int *a = malloc(sizeof(unsigned int)); *a = get_unsigned_int_in_list(data, index); ret = a; break;}
        ctensore LONG:              {long *data = (long*)origin; long *a = malloc(sizeof(long)); *a = get_long_in_list(data, index); ret = a; break;}
        ctensore UNSIGNED_LONG:     {unsigned long *data = (unsigned long*)origin; unsigned long *a = malloc(sizeof(unsigned long)); *a = get_unsigned_long_in_list(data, index); ret = a; break;}
        ctensore FLOAT:             {float *data = (float*)origin; float *a = malloc(sizeof(float)); *a = get_float_in_list(data, index); ret = a; break;}
        ctensore DOUBLE:            {double *data = (double*)origin; double *a = malloc(sizeof(double)); *a = get_double_in_list(data, index); ret = a; break;}
        ctensore LONG_INT:          {long int *data = (long int*)origin; long int *a = malloc(sizeof(long int)); *a = get_long_int_in_list(data, index); ret = a; break;}
        ctensore LONG_LONG:         {long long *data = (long long*)origin; long long *a = malloc(sizeof(long long)); *a = get_long_long_in_list(data, index); ret = a; break;}
        ctensore LONG_DOUBLE:       {long double *data = (long double*)origin; long double *a = malloc(sizeof(long double)); *a = get_long_double_in_list(data, index); ret = a; break;}
    }
    return ret;
}

char get_char_in_list(char *origin, int index)                                     {char ret = origin[index]; return ret;}
unsigned char get_unsigned_char_in_list(unsigned char *origin, int index)          {unsigned char ret = origin[index]; return ret;}
short get_short_in_list(short *origin, int index)                                  {short ret = origin[index]; return ret;}
unsigned short get_unsigned_short_in_list(unsigned short *origin, int index)       {unsigned short ret = origin[index]; return ret;}
int get_int_in_list(int *origin, int index)                                        {int ret = origin[index]; return ret;}
unsigned int get_unsigned_int_in_list(unsigned int *origin, int index)             {unsigned int ret = origin[index]; return ret;}
long get_long_in_list(long *origin, int index)                                     {long ret = origin[index]; return ret;}
unsigned long get_unsigned_long_in_list(unsigned long *origin, int index)          {unsigned long ret = origin[index]; return ret;}
float get_float_in_list(float *origin, int index)                                  {float ret = origin[index]; return ret;}
double get_double_in_list(double *origin, int index)                               {double ret = origin[index]; return ret;}
long int get_long_int_in_list(long int *origin, int index)                         {long int ret = origin[index]; return ret;}
long long get_long_long_in_list(long long *origin, int index)                      {long long ret = origin[index]; return ret;}
long double get_long_double_in_list(long double *origin, int index)                {long double ret = origin[index]; return ret;}

void memcpy_void_list(void *ret, void *origin, DataType data_type, int offset_r, int offset_o, int num)
{
    switch (data_type){
        ctensore CHAR:              {char *data = (char*)origin; memcpy_char_list((char*)ret, data, offset_r, offset_o, num); break;}
        ctensore UNSIGNED_CHAR:     {unsigned char *data = (unsigned char*)origin; memcpy_unsigned_char_list((unsigned char*)ret, data, offset_r, offset_o, num); break;}
        ctensore SHORT:             {short *data = (short*)origin; memcpy_short_list((short*)ret, data, offset_r, offset_o, num); break;}
        ctensore UNSIGNED_SHORT:    {unsigned short *data = (unsigned short*)origin; memcpy_unsigned_short_list((unsigned short*)ret, data, offset_r, offset_o, num); break;}
        ctensore INT:               {int *data = (int*)origin; memcpy_int_list((int*)ret, data, offset_r, offset_o, num); break;}
        ctensore UNSIGNED_INT:      {unsigned int *data = (unsigned int*)origin; memcpy_unsigned_int_list((unsigned int*)ret, data, offset_r, offset_o, num); break;}
        ctensore LONG:              {long *data = (long*)origin; memcpy_long_list((long*)ret, data, offset_r, offset_o, num); break;}
        ctensore UNSIGNED_LONG:     {unsigned long *data = (unsigned long*)origin; memcpy_unsigned_long_list((unsigned long*)ret, data, offset_r, offset_o, num); break;}
        ctensore FLOAT:             {float *data = (float*)origin; memcpy_float_list((float*)ret, data, offset_r, offset_o, num); break;}
        ctensore DOUBLE:            {double *data = (double*)origin; memcpy_double_list((double*)ret, data, offset_r, offset_o, num); break;}
        ctensore LONG_INT:          {long int *data = (long int*)origin; memcpy_long_int_list((long int*)ret, data, offset_r, offset_o, num); break;}
        ctensore LONG_LONG:         {long long *data = (long long*)origin; memcpy_long_long_list((long long*)ret, data, offset_r, offset_o, num); break;}
        ctensore LONG_DOUBLE:       {long double *data = (long double*)origin; memcpy_long_double_list((long double*)ret, data, offset_r, offset_o, num); break;}
    }
}

void memcpy_char_list(char *ret, char *origin, int offset_r, int offset_o, int num)                                        {memcpy(ret+offset_r, origin+offset_o, num*sizeof(char));}
void memcpy_unsigned_char_list(unsigned char *ret, unsigned char *origin, int offset_r, int offset_o, int num)             {memcpy(ret+offset_r, origin+offset_o, num*sizeof(unsigned char));}
void memcpy_short_list(short *ret, short *origin, int offset_r, int offset_o, int num)                                     {memcpy(ret+offset_r, origin+offset_o, num*sizeof(short));}
void memcpy_unsigned_short_list(unsigned short *ret, unsigned short *origin, int offset_r, int offset_o, int num)          {memcpy(ret+offset_r, origin+offset_o, num*sizeof(unsigned short));}
void memcpy_int_list(int *ret, int *origin, int offset_r, int offset_o, int num)                                           {memcpy(ret+offset_r, origin+offset_o, num*sizeof(int));}
void memcpy_unsigned_int_list(unsigned int *ret, unsigned int *origin, int offset_r, int offset_o, int num)                {memcpy(ret+offset_r, origin+offset_o, num*sizeof(unsigned int));}
void memcpy_long_list(long *ret, long *origin, int offset_r, int offset_o, int num)                                        {memcpy(ret+offset_r, origin+offset_o, num*sizeof(long));}
void memcpy_unsigned_long_list(unsigned long *ret, unsigned long *origin, int offset_r, int offset_o, int num)             {memcpy(ret+offset_r, origin+offset_o, num*sizeof(unsigned long));}
void memcpy_float_list(float *ret, float *origin, int offset_r, int offset_o, int num)                                     {memcpy(ret+offset_r, origin+offset_o, num*sizeof(float));}
void memcpy_double_list(double *ret, double *origin, int offset_r, int offset_o, int num)                                  {memcpy(ret+offset_r, origin+offset_o, num*sizeof(double));}
void memcpy_long_int_list(long int *ret, long int *origin, int offset_r, int offset_o, int num)                            {memcpy(ret+offset_r, origin+offset_o, num*sizeof(long int));}
void memcpy_long_long_list(long long *ret, long long *origin, int offset_r, int offset_o, int num)                         {memcpy(ret+offset_r, origin+offset_o, num*sizeof(long long));}
void memcpy_long_double_list(long double *ret, long double *origin, int offset_r, int offset_o, int num)                   {memcpy(ret+offset_r, origin+offset_o, num*sizeof(long double));}

void *sum_list(void *origin, DataType data_type, int offset, int len)
{
    void *ret;
    switch (data_type){
        ctensore CHAR:              {char *data = (char*)origin; int *a = malloc(sizeof(int)); *a = sum_char_list(data, offset, len); ret = a; break;}
        ctensore UNSIGNED_CHAR:     {unsigned char *data = (unsigned char*)origin; unsigned int *a = malloc(sizeof(unsigned int)); *a = sum_unsigned_char_list(data, offset, len); ret = a; break;}
        ctensore SHORT:             {short *data = (short*)origin; short *a = malloc(sizeof(short)); *a = sum_short_list(data, offset, len); ret = a; break;}
        ctensore UNSIGNED_SHORT:    {unsigned short *data = (unsigned short*)origin; unsigned short *a = malloc(sizeof(unsigned short)); *a = sum_unsigned_short_list(data, offset, len); ret = a; break;}
        ctensore INT:               {int *data = (int*)origin; int *a = malloc(sizeof(int)); *a = sum_int_list(data, offset, len); ret = a; break;}
        ctensore UNSIGNED_INT:      {unsigned int *data = (unsigned int*)origin; unsigned int *a = malloc(sizeof(unsigned int)); *a = sum_unsigned_int_list(data, offset, len); ret = a; break;}
        ctensore LONG:              {long *data = (long*)origin; long *a = malloc(sizeof(long)); *a = sum_long_list(data, offset, len); ret = a; break;}
        ctensore UNSIGNED_LONG:     {unsigned long *data = (unsigned long*)origin; unsigned long *a = malloc(sizeof(unsigned long)); *a = sum_unsigned_long_list(data, offset, len); ret = a; break;}
        ctensore FLOAT:             {float *data = (float*)origin; float *a = malloc(sizeof(float)); *a = sum_float_list(data, offset, len); ret = a; break;}
        ctensore DOUBLE:            {double *data = (double*)origin; double *a = malloc(sizeof(double)); *a = sum_double_list(data, offset, len); ret = a; break;}
        ctensore LONG_INT:          {long int *data = (long int*)origin; long int *a = malloc(sizeof(long int)); *a = sum_long_int_list(data, offset, len); ret = a; break;}
        ctensore LONG_LONG:         {long long *data = (long long*)origin; long long *a = malloc(sizeof(long long)); *a = sum_long_long_list(data, offset, len); ret = a; break;}
        ctensore LONG_DOUBLE:       {long double *data = (long double*)origin; long double *a = malloc(sizeof(long double)); *a = sum_long_double_list(data, offset, len); ret = a; break;}
    }
    return ret;
}

int sum_char_list(char *origin, int offset, int len)                                        {int ret = (int)0; for(int i = offset; i < len+offset; ++i){ret += (int)origin[i];} return ret;}
unsigned int sum_unsigned_char_list(unsigned char *origin, int offset, int len)             {unsigned int ret = (unsigned int)0; for(int i = offset; i < len+offset; ++i){ret += (unsigned int)origin[i];} return ret;}
short sum_short_list(short *origin, int offset, int len)                                    {short ret = (short)0; for(int i = offset; i < len+offset; ++i){ret += origin[i];} return ret;}
unsigned short sum_unsigned_short_list(unsigned short *origin, int offset, int len)         {unsigned short ret = (unsigned short)0; for(int i = offset; i < len+offset; ++i){ret += origin[i];} return ret;}
int sum_int_list(int *origin, int offset, int len)                                          {int ret = (int)0; for(int i = offset; i < len+offset; ++i){ret += origin[i];} return ret;}
unsigned int sum_unsigned_int_list(unsigned int *origin, int offset, int len)               {unsigned int ret = (unsigned int)0; for(int i = offset; i < len+offset; ++i){ret += origin[i];} return ret;}
long sum_long_list(long *origin, int offset, int len)                                       {long ret = (long)0; for(int i = offset; i < len+offset; ++i){ret += origin[i];} return ret;}
unsigned long sum_unsigned_long_list(unsigned long *origin, int offset, int len)            {unsigned long ret = (unsigned long)0; for(int i = offset; i < len+offset; ++i){ret += origin[i];} return ret;}
float sum_float_list(float *origin, int offset, int len)                                    {float ret = (float)0; for(int i = offset; i < len+offset; ++i){ret += origin[i];} return ret;}
double sum_double_list(double *origin, int offset, int len)                                 {double ret = (double)0; for(int i = offset; i < len+offset; ++i){ret += origin[i];} return ret;}
long int sum_long_int_list(long int *origin, int offset, int len)                           {long int ret = (long int)0; for(int i = offset; i < len+offset; ++i){ret += origin[i];} return ret;}
long long sum_long_long_list(long long *origin, int offset, int len)                        {long long ret = (long long)0; for(int i = offset; i < len+offset; ++i){ret += origin[i];} return ret;}
long double sum_long_double_list(long double *origin, int offset, int len)                  {long double ret = (long double)0; for(int i = offset; i < len+offset; ++i){ret += origin[i];} return ret;}

void *multing_list(void *origin, DataType data_type, int offset, int len)
{
    void *ret;
    switch (data_type){
        ctensore CHAR:              {char *data = (char*)origin; int *a = malloc(sizeof(int)); *a = multing_char_list(data, offset, len); ret = a; break;}
        ctensore UNSIGNED_CHAR:     {unsigned char *data = (unsigned char*)origin; unsigned int *a = malloc(sizeof(unsigned int)); *a = multing_unsigned_char_list(data, offset, len); ret = a; break;}
        ctensore SHORT:             {short *data = (short*)origin; short *a = malloc(sizeof(short)); *a = multing_short_list(data, offset, len); ret = a; break;}
        ctensore UNSIGNED_SHORT:    {unsigned short *data = (unsigned short*)origin; unsigned short *a = malloc(sizeof(unsigned short)); *a = multing_unsigned_short_list(data, offset, len); ret = a; break;}
        ctensore INT:               {int *data = (int*)origin; int *a = malloc(sizeof(int)); *a = multing_int_list(data, offset, len); ret = a; break;}
        ctensore UNSIGNED_INT:      {unsigned int *data = (unsigned int*)origin; unsigned int *a = malloc(sizeof(unsigned int)); *a = multing_unsigned_int_list(data, offset, len); ret = a; break;}
        ctensore LONG:              {long *data = (long*)origin; long *a = malloc(sizeof(long)); *a = multing_long_list(data, offset, len); ret = a; break;}
        ctensore UNSIGNED_LONG:     {unsigned long *data = (unsigned long*)origin; unsigned long *a = malloc(sizeof(unsigned long)); *a = multing_unsigned_long_list(data, offset, len); ret = a; break;}
        ctensore FLOAT:             {float *data = (float*)origin; float *a = malloc(sizeof(float)); *a = multing_float_list(data, offset, len); ret = a; break;}
        ctensore DOUBLE:            {double *data = (double*)origin; double *a = malloc(sizeof(double)); *a = multing_double_list(data, offset, len); ret = a; break;}
        ctensore LONG_INT:          {long int *data = (long int*)origin; long int *a = malloc(sizeof(long int)); *a = multing_long_int_list(data, offset, len); ret = a; break;}
        ctensore LONG_LONG:         {long long *data = (long long*)origin; long long *a = malloc(sizeof(long long)); *a = multing_long_long_list(data, offset, len); ret = a; break;}
        ctensore LONG_DOUBLE:       {long double *data = (long double*)origin; long double *a = malloc(sizeof(long double)); *a = multing_long_double_list(data, offset, len); ret = a; break;}
    }
    return ret;
}

int multing_char_list(char *origin, int offset, int len)                                        {int ret = (int)1; for(int i = offset; i < len+offset; ++i){ret *= (int)origin[i];} return ret;}
unsigned int multing_unsigned_char_list(unsigned char *origin, int offset, int len)             {unsigned int ret = (unsigned int)1; for(int i = offset; i < len+offset; ++i){ret *= (unsigned int)origin[i];} return ret;}
short multing_short_list(short *origin, int offset, int len)                                    {short ret = (short)1; for(int i = offset; i < len+offset; ++i){ret *= origin[i];} return ret;}
unsigned short multing_unsigned_short_list(unsigned short *origin, int offset, int len)         {unsigned short ret = (unsigned short)1; for(int i = offset; i < len+offset; ++i){ret *= origin[i];} return ret;}
int multing_int_list(int *origin, int offset, int len)                                          {int ret = (int)1; for(int i = offset; i < len+offset; ++i){ret *= origin[i];} return ret;}
unsigned int multing_unsigned_int_list(unsigned int *origin, int offset, int len)               {unsigned int ret = (unsigned int)1; for(int i = offset; i < len+offset; ++i){ret *= origin[i];} return ret;}
long multing_long_list(long *origin, int offset, int len)                                       {long ret = (long)1; for(int i = offset; i < len+offset; ++i){ret *= origin[i];} return ret;}
unsigned long multing_unsigned_long_list(unsigned long *origin, int offset, int len)            {unsigned long ret = (unsigned long)1; for(int i = offset; i < len+offset; ++i){ret *= origin[i];} return ret;}
float multing_float_list(float *origin, int offset, int len)                                    {float ret = (float)1; for(int i = offset; i < len+offset; ++i){ret *= origin[i];} return ret;}
double multing_double_list(double *origin, int offset, int len)                                 {double ret = (double)1; for(int i = offset; i < len+offset; ++i){ret *= origin[i];} return ret;}
long int multing_long_int_list(long int *origin, int offset, int len)                           {long int ret = (long int)1; for(int i = offset; i < len+offset; ++i){ret *= origin[i];} return ret;}
long long multing_long_long_list(long long *origin, int offset, int len)                        {long long ret = (long long)1; for(int i = offset; i < len+offset; ++i){ret *= origin[i];} return ret;}
long double multing_long_double_list(long double *origin, int offset, int len)                  {long double ret = (long double)1; for(int i = offset; i < len+offset; ++i){ret *= origin[i];} return ret;}