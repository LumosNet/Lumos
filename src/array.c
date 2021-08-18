#include "array.h"

/*************************************************************************************************\
 * 描述
       创建矩阵，初始化内存空间为x
 * 参数
       row:行数
       col:列数
       x:空间填充值
 * 返回
       Array*
       返回创建好的Array对象，您可以使用该Array对象进行一切与矩阵相关的操作
\*************************************************************************************************/
Array *create_array(int row, int col, float x)
{
    int *size = malloc(2*sizeof(int));
    size[0] = col;
    size[1] = row;
    Array *ret = create(2, size, x);
    free(size);
    return ret;
}

/*************************************************************************************************\
 * 描述
       将列表元素转换为矩阵，以行优先，逐一填充
       当列表元素无法完全填充矩阵时，待填充元素不会进行初始化
       当矩阵无法完全容纳列表元素时，多余元素将会被抛弃
 * 参数
       row:行数
       col:列数
       list:待转换列表
 * 返回
       Array*
       返回转换好的Array对象，您可以使用该Array对象进行一切与矩阵相关的操作
 * 补充
       1.
          row:2
          col:2
          list:[1, 2, 3, 4]
          ret:
               1, 2
               3, 4

       2.
          row:2
          col:2
          list:[1, 2, 3, 4, 5]
          ret:
               1, 2
               3, 4

       3.
          row: 3
          col: 3
          list:[1, 2, 3, 4, 5]
          ret:
               1, 2, 3
               4, 5, x
               x, x, x
\*************************************************************************************************/
Array *list_to_array(int row, int col, float *list)
{
    int *size = malloc(2*sizeof(int));
    size[0] = col;
    size[1] = row;
    Array *a = list_to_AS(2, size, list);
    free(size);
    return a;
}

/*************************************************************************************************\
 * 描述
       创建一个矩阵，不过会将主对角线或者斜对角线置为x
 * 参数
       row:行数
       col:列数
       x:待填充值
       flag:1代表主对角线，0代表斜对角线
 * 返回
       Array*
       返回创建好的矩阵对象Array，您可以使用该Array对象进行一切与矩阵相关的操作
 * 补充
       若矩阵不是一个方阵，我们会寻找行数、列数中的最小值k = min(row, col)，以该值
       作为基准对k*k的矩阵进行操作
       1.
          row:3
          col:4
          flag:1        flag:0
          ret:
               x o o          o o x
               o x o          o x o
               o o x          x o o
               o o o          o o o

       2.
          row:4
          col:3
          flag:1        flag:0
          ret:
               x o o o        o o o x
               o x o o        o o x o
               o o x o        o x o o
\*************************************************************************************************/
Array *create_unit_array(int row, int col, float x, int flag)
{
    int *size = malloc(2*sizeof(int));
    size[0] = col;
    size[1] = row;
    Array *ret = create_array(row, col, 0);
    int min_row_col = (row <= col) ? row : col;
    for (int i = 0; i < min_row_col; ++i){
        if (flag) change_float_in_list(ret->data, x, i*col+i);
        else change_float_in_list(ret->data, x, i*col+(col-i-1));
    }
    return ret;
}

/*************************************************************************************************\
 * 描述
       复制一个矩阵对象
 * 参数
       a:待复制矩阵对象
 * 返回
       Array*
       返回复制后的Array对象
\*************************************************************************************************/
Array *copy_array(Array *a)
{
    return copy(a);
}

/*************************************************************************************************\
 * 描述
       将二维索引转换为一维索引
 * 参数
       size:int类型列表，大小为2(size[0], size[1])
             表示一个二维空间的大小(行数，列数)
       row:行索引
       col:列索引
 * 返回
       int
       返回二维索引(row，col)对应的一维索引值
 * 补充
       设一个二维空间大小为m*n
       一个二维索引为x、y，即索引第x行，第y列
       如果以行优先将二维空间展开为一维空间
       那么二维索引对应的一维索引为
       x*n + y
\*************************************************************************************************/
int get_array_lindex(Array *a, int row, int col)
{
    int index[] = {col, row};
    return get_lindex(a, index);
}

/*************************************************************************************************\
 * 描述
       将一维索引转换为二维索引
 * 参数
       size:int类型列表，大小为2(size[0], size[1])
             表示一个二维空间的大小(行数，列数)
       index:一维索引值
 * 返回
       int*
       返回一维索引对应的二维索引
 * 补充
       设一个二维空间大小为m*n
       一个一维索引为index
       那么一维索引对应的二维索引为
       index / n, index % n
\*************************************************************************************************/
int *get_array_aindex(Array *a, int index)
{
    int *lindex = get_mindex(a, index);
    int x = lindex[0];
    lindex[0] = lindex[1];
    lindex[1] = x;
    return lindex;
}

/*************************************************************************************************\
 * 描述
       以二维索引方式，索引矩阵元素
 * 参数
       a:待索引矩阵
       row:行索引
       col:列索引
 * 返回
       float
       返回索引到的元素值
\*************************************************************************************************/
float get_array_pixel(Array *a, int row, int col)
{
    int index[] = {col, row};
    return get_pixel(a, index);
}

/*************************************************************************************************\
 * 描述
       获取矩阵中最小元素值
 * 参数
       a:待获取矩阵
 * 返回
       float
       矩阵中的最小元素值
\*************************************************************************************************/
float get_array_min(Array *a)
{
    return get_min(a);
}

/*************************************************************************************************\
 * 描述
       获取矩阵中最大元素值
 * 参数
       a:待获取矩阵
 * 返回
       float
       矩阵中的最大元素值
\*************************************************************************************************/
float get_array_max(Array *a)
{
    return get_max(a);
}

/*************************************************************************************************\
 * 描述
       计算矩阵元素的算数平均值
 * 参数
       a:待计算矩阵
 * 返回
       float
       矩阵的算数平均值
\*************************************************************************************************/
float get_array_mean(Array *a)
{
    return get_mean(a);
}

/*************************************************************************************************\
 * 描述
       获取矩阵中指定数值元素个数
 * 参数
       a:待统计的矩阵
       x:指定的数值
 * 返回
       int
       返回矩阵中指定数值元素个数
\*************************************************************************************************/
int pixel_num_array(Array *a, float x)
{
    return get_num(a, x);
}

/*************************************************************************************************\
 * 描述
       修改矩阵中某元素值
       以二维索引方式
 * 参数
       a:待操作矩阵对象
       row:行索引
       col:列索引
       x:指定修改值
 * 返回
       void
\*************************************************************************************************/
void change_array_pixel(Array *a, int row, int col, float x)
{
    int index[] = {col, row};
    change_pixel(a, index, x);
}

/*************************************************************************************************\
 * 描述
       重置矩阵大小
 * 参数
       a:待转换矩阵
       row:新矩阵大小（行数）
       col:新矩阵大小（列数）
 * 返回
       void
\*************************************************************************************************/
void resize_array(Array *a, int row, int col)
{
    int *size = malloc(2*sizeof(int));
    size[0] = col;
    size[1] = row;
    resize(a, 2, size);
    free(size);
}

/*************************************************************************************************\
 * 描述
       获取矩阵某一行（以列表形式返回）
 * 参数
       a:待获取矩阵
       row:待获取的行（1开始）
 * 返回
       float*
       获取的行数据列表
\*************************************************************************************************/
float *get_row_in_array(Array *a, int row)
{
    int size = a->size[0];
    int offset = (row-1)*size;
    float *data = malloc(size*sizeof(float));
    memcpy_float_list(data, a->data, 0, offset, size);
    return data;
}

/*************************************************************************************************\
 * 描述
       获取矩阵某一列（以列表形式返回）
 * 参数
       a:待获取矩阵
       col:待获取的列（1开始）
 * 返回
       float*
       获取的列数据列表
\*************************************************************************************************/
float *get_col_in_array(Array *a, int col)
{
    int x = a->size[0];
    int y = a->size[1];
    float *data = malloc(y*sizeof(float));
    for (int i = 0; i < x; ++i){
        data[i] = get_array_pixel(a, i+1, col);
    }
    return data;
}

/*************************************************************************************************\
 * 描述
       获取矩阵对角线元素（以列表形式返回）
 * 参数
       a:待获取矩阵
       flag:1代表主对角线，0代表斜对角线
 * 返回
       float*
       获取的对角线数据列表
\*************************************************************************************************/
float *get_diagonal_in_array(Array *a, int flag)
{
    int x = a->size[0];
    int y = a->size[1];
    int min = (x<y) ? x : y;
    float *data = malloc(min*sizeof(float));
    for (int i = 0; i < min; ++i){
        if (flag) data[i] = get_array_pixel(a, i+1, i+1);
        else data[i] = get_array_pixel(a, i+1, min-i);
    }
    return data;
}

/*************************************************************************************************\
 * 描述
       获取矩阵某一行（以向量形式返回）
 * 参数
       a:待获取矩阵
       row:待获取的行（1开始）
 * 返回
       Victor*
       获取的行数据组成的向量
\*************************************************************************************************/
Victor *row2Victor(Array *a, int row)
{
    int size = a->size[0];
    int offset = (row-1)*size;
    float *data = malloc(size*sizeof(float));
    memcpy_float_list(data, a->data, 0, offset, size);
    return list_to_Victor(size, 1, data);
}

/*************************************************************************************************\
 * 描述
       获取矩阵某一列（一向量形式返回）
 * 参数
       a:待获取矩阵
       col:待获取的列（1开始）
 * 返回
       Victor*
       获取的列数据组成的向量
\*************************************************************************************************/
Victor *col2Victor(Array *a, int col)
{
    int x = a->size[0];
    int y = a->size[1];
    float *data = malloc(y*sizeof(float));
    for (int i = 0; i < x; ++i){
        data[i] = get_array_pixel(a, i+1, col);
    }
    return list_to_Victor(y, 0, data);
}

/*************************************************************************************************\
 * 描述
       获取矩阵对角线元素（以列表形式返回）
 * 参数
       a:待获取矩阵
       flag:1代表主对角线，0代表斜对角线
 * 返回
       Victor*
       获取的对角线数据向量
\*************************************************************************************************/
Victor *diagonal2Victor(Array *a, int flag)
{
    int x = a->size[0];
    int y = a->size[1];
    int min = (x<y) ? x : y;
    float *data = malloc(min*sizeof(float));
    for (int i = 0; i < min; ++i){
        if (flag) data[i] = get_array_pixel(a, i+1, i+1);
        else data[i] = get_array_pixel(a, i+1, min-i);
    }
    return list_to_Victor(min, flag, data);
}

/*************************************************************************************************\
 * 描述
       替换矩阵中某一行元素
 * 参数
       a:待处理矩阵
       row:待替换行
       list:替换的元素
 * 返回
       void
\*************************************************************************************************/
void replace_array_row2list(Array *a, int row, float *list)
{
    int size = a->size[0];
    int index = (row-1)*size;
    memcpy_float_list(a->data, list, index, 0, size);
}

/*************************************************************************************************\
 * 描述
       替换矩阵中某一列
 * 参数
       a:待处理矩阵
       col:待替换列
       list:替换的元素
 * 返回
       void
\*************************************************************************************************/
void replace_array_col2list(Array *a, int col, float *list)
{
    for (int i = 0; i < a->size[1]; ++i){
        change_array_pixel(a, i+1, col, list[i]);
    }
}

/*************************************************************************************************\
 * 描述
       替换矩阵的对角线元素
 * 参数
       a:待处理元素
       list:替换的元素
       flag:1代表主对角线，0代表斜对角线
 * 返回
       void
\*************************************************************************************************/
void replace_array_diagonal2list(Array *a, float *list, int flag)
{
    int min = (a->size[0] < a->size[1]) ? a->size[0] : a->size[1];
    for (int i = 0; i < min; ++i){
        if (flag) change_array_pixel(a, i+1, i+1, list[i]);
        else change_array_pixel(a, i+1, a->size[0]-i, list[i]);
    }
}

/*************************************************************************************************\
 * 描述
       用元素x替换某一行所有元素
 * 参数
       a:待处理矩阵
       row:待替换行
       x:替换的元素
 * 返回
       void
\*************************************************************************************************/
void replace_array_row_with_x(Array *a, int row, float x)
{
    for (int i = 0; i < a->size[0]; ++i){
        change_array_pixel(a, row, i+1, x);
    }
}

/*************************************************************************************************\
 * 描述
       用元素x替换某一列所有元素
 * 参数
       a:待处理矩阵‘
       col:待替换列
       x:替换的元素
 * 返回
       void
\*************************************************************************************************/
void replace_array_col_with_x(Array *a, int col, float x)
{
    for (int i = 0; i < a->size[1]; ++i){
        change_array_pixel(a, i+1, col, x);
    }
}

/*************************************************************************************************\
 * 描述
       用元素x替换矩阵对角线元素
 * 参数
       a:待处理矩阵
       x:替换的元素
       flag:1代表主对角线，0代表斜对角线
 * 返回
       void
\*************************************************************************************************/
void replace_diagonal_with_x(Array *a, float x, int flag)
{
    int min = (a->size[0] < a->size[1]) ? a->size[0] : a->size[1];
    for (int i = 0; i < min; ++i){
        if (flag) change_array_pixel(a, i+1, i+1, x);
        else change_array_pixel(a, i+1, a->size[0]-i, x);
    }
}

/*************************************************************************************************\
 * 描述
       删除矩阵某一行
 * 参数
       a:待处理矩阵
       row:待删除行（1开始）
 * 返回
       void
\*************************************************************************************************/
void del_row_in_array(Array *a, int row)
{
    a->num -= a->size[0];
    float *data = malloc((a->num)*sizeof(float));
    memcpy_float_list(data, a->data, 0, 0, (row-1)*a->size[0]);
    memcpy_float_list(data, a->data, (row-1)*a->size[0], row*a->size[0], (a->size[1]-row)*a->size[0]);
    free(a->data);
    a->data = data;
    a->size[1] -= 1;
}

/*************************************************************************************************\
 * 描述
       删除矩阵某一列
 * 参数
       a:待处理矩阵
       col:待删除列（1开始）
 * 返回
       void
\*************************************************************************************************/
void del_col_in_array(Array *a, int col)
{
    a->num -= a->size[1];
    float *data = malloc((a->num)*sizeof(float));
    for (int i = 0; i < a->size[1]; ++i){
        int offset = i*a->size[0];
        memcpy_float_list(data, a->data, i*(a->size[0]-1), offset, col-1);
        memcpy_float_list(data, a->data, i*(a->size[0]-1)+col-1, offset+col, (a->size[0]-col));
    }
    free(a->data);
    a->data = data;
    a->size[0] -= 1;
}

/*************************************************************************************************\
 * 描述
       向矩阵中插入一行元素
       前向插入，即将新行插入指定行索引的前面
 * 参数
       a:待插入矩阵
       index:待插入位置索引
       data:待插入数据
 * 返回
       void
 * 补充
       设data = [x1, x2, x3, x4]
       向矩阵A插入:
            a11  a12  a13  a14
            a21  a22  a23  a24
            a31  a32  a33  a34
            a41  a42  a43  a44

       index:1（插入最前）                      index:5（插入最后）
            x1   x2   x3   x4                         a11  a12  a13  a14
            a11  a12  a13  a14                        a21  a22  a23  a24
            a21  a22  a23  a24                        a31  a32  a33  a34
            a31  a32  a33  a34                        a41  a42  a43  a44
            a41  a42  a43  a44                        x1   x2   x3   x4
\*************************************************************************************************/
void insert_row_in_array(Array *a, int index, float *data)
{
    a->num += a->size[0];
    float *new_data = malloc(a->num*sizeof(float));
    memcpy_float_list(new_data, a->data, 0, 0, (index-1)*a->size[0]);
    memcpy_float_list(new_data, data, (index-1)*a->size[0], 0, a->size[0]);
    memcpy_float_list(new_data, a->data, index*a->size[0], (index-1)*a->size[0], (a->size[1]-index+1)*a->size[0]);
    free(a->data);
    a->data = new_data;
    a->size[1] += 1;
}

/*************************************************************************************************\
 * 描述
       向矩阵中插入一列元素
       前向插入，即将新列插入指定列索引的前面
 * 参数
       a:待插入矩阵
       index:待插入位置索引
       data:待插入数据
 * 返回
       void
 * 补充
       设data = [x1, x2, x3, x4]
       向矩阵A插入:
            a11  a12  a13  a14
            a21  a22  a23  a24
            a31  a32  a33  a34
            a41  a42  a43  a44

       index:1（插入最前）                      index:5（插入最后）
            x1  a11  a12  a13  a14                    a11  a12  a13  a14  x1
            x2  a21  a22  a23  a24                    a21  a22  a23  a24  x2
            x3  a31  a32  a33  a34                    a31  a32  a33  a34  x3
            x4  a41  a42  a43  a44                    a41  a42  a43  a44  x4
\*************************************************************************************************/
void insert_col_in_array(Array *a, int index, float *data)
{
    a->num += a->size[1];
    float *new_data = malloc((a->num)*sizeof(float));
    for (int i = 0; i < a->size[1]; ++i){
        int offset = i*a->size[0];
        memcpy_float_list(new_data, a->data, i*(a->size[0]+1), offset, index-1);
        new_data[i*(a->size[0]+1)+index-1] = data[i];
        memcpy_float_list(new_data, a->data, i*(a->size[0]+1)+index, offset+index-1, (a->size[0]-index+1));
    }
    free(a->data);
    a->data = new_data;
    a->size[0] += 1;
}

void replace_part_array(Array *a, Array *b, int row, int col)
{
    int *index = malloc(2*sizeof(int));
    index[0] = row;
    index[1] = col;
    replace_part(a, b, index);
}

/*************************************************************************************************\
 * 描述
       合并两个矩阵，支持向不同维不同位置索引进行合并
       参数中第一个矩阵为目标矩阵，第二个矩阵向第一个矩阵中合并
       合并时，采取前向合并
 * 参数
       a:目标矩阵
       b:待合并矩阵
       dim:向dim维进行合并
       index:合并位置索引
 * 返回
       Array*
       合并后的新矩阵
 * 补充
       一:
       a:                                       b:
            a11  a12  a13                             b11  b12
            a21  a22  a23                             b21  b22
      
       dim:1
       由于a、b行数相同，所以只能向列（第二维）合并

       index:1                                  index:4
            b11  b12  a11  a12  a13                   a11  a12  a13  b11  b12
            b21  b22  a21  a22  a23                   a21  a22  a23  b21  b22

       二:
       a:                                       b:
            a11  a12  a13                             b11  b12  b13
            a21  a22  a23                             b21  b22  b23
                                                      b31  b32  b33
       dim:2
       由于a、b列数相同，所以只能向行（第一维）合并

       index:1                                  index:3
            b11  b12  b13                             a11  a12  a13
            b21  b22  b23                             a21  a22  a23
            b31  b32  b33                             b11  b12  b13
            a11  a12  a13                             b21  b22  b23
            a21  a22  a23                             b31  b32  b33
\*************************************************************************************************/
Array *merge_array(Array *a, Array *b, int flag, int index)
{
    int dim = -1;
    int size[] = {a->size[0], a->size[1]};
    if (flag){
        size[1] += b->size[1];
        dim = 2;
    }
    else{
        size[0] += b->size[0];
        dim = 1;
    }
    Array *ret = create_array(size[1], size[0], 0);
    merge(a, b, dim, index, ret->data);
    return ret;
}

/*************************************************************************************************\
 * 描述
       对矩阵进行切片操作，可以类比python numpy或python list的相关操作a[: ; :]
       选取行索引范围以及列索引范围，将提取索引范围内的元素
       提取后的元素组成一个新的矩阵
 * 参数
       a：待切片矩阵
       rowu：上侧行索引
       rowd：下侧行索引
       coll：左侧列索引
       colr：右侧列索引
 * 返回
       Array*
       切片结果
 * 补充
       a：
            a11  a12  a13  a14  a15
            a21  a22  a23  a24  a25
            a31  a32  a33  a34  a35
            a41  a42  a43  a44  a45
            a51  a52  a53  a54  a55

       rowu:2  rowd:4  coll:2  colr:5

       结果：
            a22  a23  a24  a25
            a32  a33  a34  a35
            a42  a43  a44  a45
\*************************************************************************************************/
Array *slice_array(Array *a, int rowu, int rowd, int coll, int colr)
{
    Array *ret = create_array(rowd-rowu+1, colr-coll+1, 0);
    int offset_row = coll-1;
    int size = colr-coll+1;
    for (int i = rowu-1, n = 0; i < rowd; ++i, ++n){
        int offset_a = i*a->size[0];
        int offset_r = n*ret->size[0];
        memcpy_float_list(ret->data, a->data, offset_r, offset_a+offset_row, size);
    }
    return ret;
}

/*************************************************************************************************\
 * 描述
       删除矩阵，回收内存
 * 参数
       a：待删除矩阵
 * 返回
       void
\*************************************************************************************************/
void del_array(Array *a)
{
    del(a);
}

/*************************************************************************************************\
 * 描述
       对矩阵进行左右翻转
 * 参数
       a：待处理矩阵
 * 返回
       void
 * 补充
       a：
            a11  a12  a13
            a21  a22  a23
            a31  a32  a33

       结果：
            a13  a12  a11
            a23  a22  a21
            a33  a32  a31
\*************************************************************************************************/
void array_overturn_lr(Array *a)
{
    int x = a->size[1];
    int y = a->size[0];
    float *data = malloc(a->num*sizeof(float));
    for (int i = 0; i < x; ++i){
        for (int j = 0; j < y; ++j){
            int col_index = y-1-j;
            data[i*y+j] = a->data[i*y+col_index];
        }
    }
    free(a->data);
    a->data = data;
}

/*************************************************************************************************\
 * 描述
       对矩阵进行上下翻转
 * 参数
       a：待处理矩阵
 * 返回
       void
 * 补充
       a：
            a11  a12  a13
            a21  a22  a23
            a31  a32  a33

       结果：
            a31  a32  a33
            a21  a22  a23
            a11  a12  a13
\*************************************************************************************************/
void array_overturn_ud(Array *a)
{
    int x = a->size[1];
    int y = a->size[0];
    float *data = malloc(a->num*sizeof(float));
    for (int i = 0; i < x; ++i){
        int row_index = x-1-i;
        for (int j = 0; j < y; ++j){
            data[i*y+j] = a->data[row_index*y+j];
        }
    }
    free(a->data);
    a->data = data;
}

/*************************************************************************************************\
 * 描述
       矩阵沿对角线翻转，flag=1沿主对角线，flag=0沿斜对角线
 * 参数
       a:待处理矩阵
       flag：1沿主对角线，0沿斜对角线
 * 返回
       void
\*************************************************************************************************/
void array_overturn_diagonal(Array *a, int flag)
{
    float *data = malloc(a->num*sizeof(float));
    for (int i = 0; i < a->size[1]; ++i){
        for (int j = 0; j < a->size[0]; ++j){
            if (flag) data[j*a->size[1]+i] = a->data[i*a->size[0]+j];
            else data[(a->size[0]-j-1)*a->size[1]+a->size[1]-i-1] = a->data[i*a->size[0]+j];
        }
    }
    int x = a->size[0];
    a->size[0] = a->size[1];
    a->size[1] = x;
    free(a->data);
    a->data = data;
}

/*************************************************************************************************\
 * 描述
       向左旋转矩阵，k为旋转次数
 * 参数
       a：待处理矩阵
       k：旋转次数
 * 返回
       void
\*************************************************************************************************/
void array_rotate_left(Array *a, int k)
{
    k %= 4;
    if (k == 0) return;
    if (k == 1){
        int x = a->size[1];
        int y = a->size[0];
        float *data = malloc(x*y*sizeof(float));
        for (int i = 0; i < x; ++i){
            for (int j = 0; j < y; ++j){
                data[(y-j-1)*x+i] = a->data[i*y+j];
            }
        }
        a->size[1] = y;
        a->size[0] = x;
        free(a->data);
        a->data = data;
    }
    if (k == 2){
        array_overturn_ud(a);
        array_overturn_lr(a);
    }
    if (k == 3){
        int x = a->size[1];
        int y = a->size[0];
        float *data = malloc(x*y*sizeof(float));
        for (int i = 0; i < x; ++i){
            for (int j = 0; j < y; ++j){
                data[j*x+x-i-1] = a->data[i*y+j];
            }
        }
        a->size[1] = y;
        a->size[0] = x;
        free(a->data);
        a->data = data;
    }
}

/*************************************************************************************************\
 * 描述
       向右旋转矩阵，k为旋转次数
 * 参数
       a：待处理矩阵
       k：旋转次数
 * 返回
       void
\*************************************************************************************************/
void array_rotate_right(Array *a, int k)
{
    k %= 4;
    if (k == 0) return;
    if (k == 1){
        int x = a->size[1];
        int y = a->size[0];
        float *data = malloc(x*y*sizeof(float));
        for (int i = 0; i < x; ++i){
            for (int j = 0; j < y; ++j){
                data[j*x+x-i-1] = a->data[i*y+j];
            }
        }
        a->size[1] = y;
        a->size[0] = x;
        free(a->data);
        a->data = data;
    }
    if (k == 2){
        array_overturn_ud(a);
        array_overturn_lr(a);
    }
    if (k == 3){
        int x = a->size[1];
        int y = a->size[0];
        float *data = malloc(x*y*sizeof(float));
        for (int i = 0; i < x; ++i){
            for (int j = 0; j < y; ++j){
                data[(y-j-1)*x+i] = a->data[i*y+j];
            }
        }
        a->size[1] = y;
        a->size[0] = x;
        free(a->data);
        a->data = data;
    }
}

/*************************************************************************************************\
 * 描述
       对调矩阵两行
 * 参数
       a：待处理矩阵
       rowx，rowy：待对调行的索引
 * 返回
       void
\*************************************************************************************************/
void exchange2row_in_array(Array *a, int rowx, int rowy)
{
    int y = a->size[0];
    for (int i = 0; i < y; ++i){
        int n = a->data[(rowx-1)*y+i];
        a->data[(rowx-1)*y+i] = a->data[(rowy-1)*y+i];
        a->data[(rowy-1)*y+i] = n;
    }
}

/*************************************************************************************************\
 * 描述
       对调矩阵两列
 * 参数
       a：待处理矩阵
       colx，coly：待对调列的索引
 * 返回
       void
\*************************************************************************************************/
void exchange2col_in_array(Array *a, int colx, int coly)
{
    int x = a->size[1];
    int y = a->size[0];
    for (int i = 0; i < x; ++i){
        int n = a->data[i*y+colx-1];
        a->data[i*y+colx-1] = a->data[i*y+coly-1];
        a->data[i*y+coly-1] = n;
    }
}

/*************************************************************************************************\
 * 描述
       矩阵转置
 * 参数
       a：待处理矩阵
 * 返回
       void
\*************************************************************************************************/
void transposition(Array *a)
{
    array_overturn_diagonal(a, 1);
}

/*************************************************************************************************\
 * 描述
       高斯消元求矩阵的逆
 * 参数
       a：待求逆矩阵
 * 返回
       Array*
       矩阵a的逆矩阵
\*************************************************************************************************/
Array *array_inverse(Array *a)
{
    int x = a->size[1];
    int y = a->size[0];
    int key = -1;
    Array *res = create_unit_array(x, x, 1, 1);
    for (int i = 0; i < x; ++i) {
        if (a->data[i*y+i] == 0.0) {
            //交换行获得一个非零的对角元素
            for (int j = i + 1; j < x; ++j) {
                if (a->data[j*y+i] != 0.0){
                    key = j;
                    break;
                }
            }
            if (key == -1) {
                return res;
            }
            // 交换矩阵两行
            exchange2row_in_array(a, i+1, key+1);
            exchange2row_in_array(res, i+1, key+1);
        }
        float scalar = 1.0 / a->data[i*y+i];
        array_multiplication_row_x(a, i, scalar);
        array_multiplication_row_x(res, i, scalar);
        for (int k = 0; k < x; ++k) {
            if (i == k) {
                continue;
            }
            float shear_needed = -a->data[k*y+i];
            // 行乘以一个系数加到另一行
            array_rowmulti_add_to_row(a, i, k, shear_needed);
            array_rowmulti_add_to_row(res, i, k, shear_needed);
        }
        key = -1;
    }
    return res;
}

/*************************************************************************************************\
 * 描述
       求矩阵中所有元素之和
 * 参数
       a：待求矩阵
 * 返回
       float
       元素之和
\*************************************************************************************************/
float sum_array(Array *a)
{
    return sum_float_list(a->data, 0, a->num);
}

/*************************************************************************************************\
 * 描述
       求矩阵的迹
 * 参数
       a：待求矩阵
 * 返回
       float
       矩阵的迹
\*************************************************************************************************/
float get_trace(Array *a)
{
    int x = a->size[1];
    int y = a->size[0];
    int min = (x < y) ? x : y;
    float *diagonal = get_diagonal_in_array(a, 1);
    float res = 0;
    for (int i = 0; i < min; ++i){
        res += diagonal[i];
    }
    return res;
}

/*************************************************************************************************\
 * 描述
       矩阵加法，对应元素相加
 * 参数
       a，b：待相加的两个矩阵
 * 返回
       Array*
       相加后的新矩阵
\*************************************************************************************************/
Array *array_add(Array *a, Array *b)
{
    int x = a->size[1];
    int y = a->size[0];
    Array *res = create_array(x, y, 0);
    for (int i = 0; i < x; ++i){
        for (int j = 0; j < y; ++j){
            res->data[i*y+j] = a->data[i*y+j] + b->data[i*y+j];
        }
    }
    return res;
}

/*************************************************************************************************\
 * 描述
       矩阵相减，对应元素相减
       a为被减数，b为减数
 * 参数
       a，b：待相减的两个矩阵
 * 返回
       Array*
       相减后的新矩阵
\*************************************************************************************************/
Array *array_subtract(Array *a, Array *b)
{
    int x = a->size[1];
    int y = a->size[0];
    Array *res = create_array(x, y, 0);
    for (int i = 0; i < x; ++i){
        for (int j = 0; j < y; ++j){
            res->data[i*y+j] = a->data[i*y+j] - b->data[i*y+j];
        }
    }
    return res;
}

/*************************************************************************************************\
 * 描述
       矩阵相除，对应位置相除
       a为分母，b为分子
 * 参数
       a，b：待相除的两个矩阵
 * 返回
       Array*
       相除后的新矩阵
\*************************************************************************************************/
Array *array_divide(Array *a, Array *b)
{
    int x = a->size[1];
    int y = a->size[0];
    Array *res = create_array(x, y, 0);
    for (int i = 0; i < x; ++i){
        for (int j = 0; j < y; ++j){
            res->data[i*y+j] = a->data[i*y+j] / b->data[i*y+j];
        }
    }
    return res;
}

/*************************************************************************************************\
 * 描述
       矩阵乘法（叉乘），对应位置相乘
 * 参数
       a，b：待相乘的两个矩阵
 * 返回
       Array*
       相乘后的新矩阵
\*************************************************************************************************/
Array *array_x_multiplication(Array *a, Array *b)
{
    int x = a->size[1];
    int y = a->size[0];
    Array *res = create_array(x, y, 0);
    for (int i = 0; i < x; ++i){
        for (int j = 0; j < y; ++j){
            res->data[i*y+j] = a->data[i*y+j] * b->data[i*y+j];
        }
    }
    return res;
}

/*************************************************************************************************\
 * 描述
       矩阵中各个元素累加一个指定数值x
 * 参数
       a：待处理矩阵
       x：指定的累加数值
 * 返回
       void
\*************************************************************************************************/
void array_add_x(Array *a, float x)
{
    int m = a->size[1];
    int n = a->size[0];
    for (int i = 0; i < m; ++i){
        for (int j = 0; j < n; ++j){
            a->data[i*n+j] += x;
        }
    }
}

/*************************************************************************************************\
 * 描述
       矩阵中某一行元素累加一个指定数值x
 * 参数
       a：待操作矩阵
       row：待累加行索引
       x：指定累加数值
 * 返回
       void
\*************************************************************************************************/
void array_add_row_x(Array *a, int row, float x)
{
    int y = a->size[0];
    for (int i = 0; i < y; ++i){
        a->data[row*y+i] += x;
    }
}

/*************************************************************************************************\
 * 描述
       矩阵中某一列元素累加一个指定数值x
 * 参数
       a：待操作矩阵
       col：待累加列索引
       x：指定累加数值
 * 返回
       void
\*************************************************************************************************/
void array_add_col_x(Array *a, int col, float x)
{
    int m = a->size[1];
    int n = a->size[0];
    for (int i = 0; i < m; ++i){
        a->data[i*n+col] += x;
    }
}

void array_subtract_x(Array *a, float x)
{
    int m = a->size[1];
    int n = a->size[0];
    for (int i = 0; i < m; ++i){
        for (int j = 0; j < n; ++j){
            a->data[i*n+j] -= x;
        }
    }
}

void array_subtract_row_x(Array *a, int row, float x)
{
    int y = a->size[0];
    for (int i = 0; i < y; ++i){
        a->data[row*y+i] -= x;
    }
}

void array_subtract_col_x(Array *a, int col, float x)
{
    int m = a->size[1];
    int n = a->size[0];
    for (int i = 0; i < m; ++i){
        a->data[i*n+col] -= x;
    }
}

/*************************************************************************************************\
 * 描述
       矩阵中每一个元素与一个指定数值相乘
 * 参数
       a：待处理矩阵
       x：指定数值
 * 返回
       void
\*************************************************************************************************/
void array_multiplication_x(Array *a, float x)
{
    int m = a->size[1];
    int n = a->size[0];
    for (int i = 0; i < m; ++i){
        for (int j = 0; j < n; ++j){
            a->data[i*n+j] *= x;
        }
    }
}

/*************************************************************************************************\
 * 描述
       矩阵中某一行元素依次与指定数值相乘
 * 参数
       a：待处理矩阵
       row：待处理行索引
       x：指定数值
 * 返回
       void
\*************************************************************************************************/
void array_multiplication_row_x(Array *a, int row, float x)
{
    int y = a->size[0];
    for (int i = 0; i < y; ++i){
        a->data[row*y+i] *= x;
    }
}

/*************************************************************************************************\
 * 描述
       矩阵中某一列元素依次与指定数值相乘
 * 参数
       a：待处理矩阵
       col：待处理列索引
       x：指定数值
 * 返回
       void
\*************************************************************************************************/
void array_multiplication_col_x(Array *a, int col, float x)
{
    int m = a->size[1];
    int n = a->size[0];
    for (int i = 0; i < m; ++i){
        a->data[i*n+col] *= x;
    }
}

void array_divide_x(Array *a, float x)
{
    if (x == 0) printf("警告，被除数为0\n");
    int m = a->size[1];
    int n = a->size[0];
    for (int i = 0; i < m; ++i){
        for (int j = 0; j < n; ++j){
            a->data[i*n+j] /= x;
        }
    }
}

void array_divide_row_x(Array *a, int row, float x)
{
    if (x == 0) printf("警告，被除数为0\n");
    int y = a->size[0];
    for (int i = 0; i < y; ++i){
        a->data[row*y+i] /= x;
    }
}

void array_divide_col_x(Array *a, int col, float x)
{
    if (x == 0) printf("警告，被除数为0\n");
    int m = a->size[1];
    int n = a->size[0];
    for (int i = 0; i < m; ++i){
        a->data[i*n+col] /= x;
    }
}

/*************************************************************************************************\
 * 描述
       矩阵某一行元素加到另一行
 * 参数
       a：待处理矩阵
       row1，row2：待处理行索引，row2为被加行
 * 返回
       void
\*************************************************************************************************/
void array_add_row_to_row(Array *a, int row1, int row2)
{
    int y = a->size[0];
    for (int i = 0; i < y; ++i){
        a->data[row2*y+i] += a->data[row1*y+i];
    }
}

/*************************************************************************************************\
 * 描述
       矩阵某一列元素加到另一列
 * 参数
       a：待处理矩阵
       col1，col2：待处理列索引，col2为被加列
 * 返回
       void
\*************************************************************************************************/
void array_add_col_to_col(Array *a, int col1, int col2)
{
    int m = a->size[1];
    int n = a->size[0];
    for (int i = 0; i < m; ++i){
        a->data[i*n+col2] += a->data[i*n+col1];
    }
}

/*************************************************************************************************\
 * 描述
       矩阵某一行乘一个数加到另一行
 * 参数
       a：待处理矩阵
       row1，row2：待处理行索引，row2为被加行
       x：指定数值
 * 返回
       void
\*************************************************************************************************/
void array_rowmulti_add_to_row(Array *a, int row1, int row2, float x)
{
    int y = a->size[0];
    for (int i = 0; i < y; ++i){
        a->data[row2*y+i] += a->data[row1*y+i] * x;
    }
}

/*************************************************************************************************\
 * 描述
       矩阵某一列乘一个数加到另一列
 * 参数
       a：待处理矩阵
       col1，col2：待处理列索引，row2为被加列
       x：指定数值
 * 返回
       void
\*************************************************************************************************/
void array_colmulti_add_to_col(Array *a, int col1, int col2, float x)
{
    int m = a->size[1];
    int n = a->size[0];
    for (int i = 0; i < m; ++i){
        a->data[i*n+col2] += a->data[i*n+col1] * x;
    }
}

/*************************************************************************************************\
 * 描述
       实现矩阵之间的乘法
 * 参数
       a，b：矩阵乘的矩阵
 * 返回
       Array*
       矩阵乘的结果
\*************************************************************************************************/
Array *gemm(Array *a, Array *b)
{
    int x = a->size[1];
    int y = a->size[0];
    int z = b->size[0];
    Array *res = create_array(x, z, 0);
    #pragma omp parallel for
    for (int i = 0; i < x; ++i){
        for (int k = 0; k < y; ++k){
            register float temp = a->data[i*y+k];
            for (int j = 0; j < z; ++j){
                res->data[i*z+j] += temp * b->data[k*z+j];
            }
        }
    }
    return res;
}

/*************************************************************************************************\
 * 描述
       矩阵（向量）外积修正
 * 参数
       ax：修正对象
       ay：被修正对象
       x：指定的外积元素
 * 返回
       void
\*************************************************************************************************/
void array_saxpy(Array *ax, Array *ay, float x)
{
    saxpy(ax, ay, x);
}

/*************************************************************************************************\
 * 描述
       矩阵1-范数，列和范数
 * 参数
       a：待求对象
 * 返回
       float
       求得的结果
\*************************************************************************************************/
float array_1norm(Array *a)
{
    float res = -9999;
    for (int i = 0; i < a->size[1]; ++i){
        float sum = 0;
        for (int j = 0; j < a->size[0]; ++j){
            sum += get_array_pixel(a, j, i);
        }
        if (res < sum){
            res = sum;
        }
    }
    return res;
}

/*************************************************************************************************\
 * 描述
       矩阵2-范数，谱范数
 * 参数
       a：待求对象
 * 返回
       float
       求得的结果
\*************************************************************************************************/
float array_2norm(Array *a)
{
    return 0;
}

/*************************************************************************************************\
 * 描述
       矩阵∞-范数，行和范数
 * 参数
       a：待求对象
 * 返回
       float
       求得的结果
\*************************************************************************************************/
float array_infinite_norm(Array *a)
{
    float res = -9999;
    for (int i = 0; i < a->size[0]; ++i){
        float sum = 0;
        for (int j = 0; j < a->size[1]; ++j){
            sum += get_array_pixel(a, i, j);
        }
        if (res < sum){
            res = sum;
        }
    }
    return res;
}

/*************************************************************************************************\
 * 描述
       矩阵F-范数，Frobenius范数
 * 参数
       a：待求对象
 * 返回
       float
       求得的结果
\*************************************************************************************************/
float array_frobenius_norm(Array *a)
{
    float res = 0;
    for (int i = 0; i < a->num; ++i){
        res += a->data[i] * a->data[i];
    }
    res = sqrt(res);
    return res;
}

/*************************************************************************************************\
 * 描述
       求householder向量
 * 参数
       x：求给定向量x对应的householder变换分解式中的householder向量以及β值
 * 返回
       Victor*
       返回householder向量
\*************************************************************************************************/
Victor *__householder_v(Victor *x, float *beta)
{
    Victor *m = slice_Victor(x, 1, x->size[0]);
    Victor *mt = copy_victor(m);
    transposition(mt);
    float theta = Victor_x_multiplication(mt, m)->data[0];
    Victor *v = copy_victor(x);
    m->data[0] = 1;
    beta[0] = 0;
    if (theta != 0){
        float u = sqrt(x->data[0]*x->data[0] + theta);
        if (x->data[0] <= 0) v->data[0] = x->data[0] - u;
        else v->data[0] = -theta / (x->data[0] + u);
        beta[0] = (2*v->data[0]*v->data[0]) / (theta + v->data[0]*v->data[0]);
        Victor_divide_x(v, v->data[0]);
    }
    del_Victor(m);
    del_Victor(mt);
    return v;
}

Array *__householder_a(Victor *v, float beta)
{
    Array *In = create_unit_array(v->size[0], v->size[0], 1, 1);
    Array *vt = copy_array(v);
    transposition(vt);
    Array *k = array_x_multiplication(v, vt);
    array_multiplication_x(k, beta);
    Array *P = array_subtract(In, k);
    del_array(In);
    del_array(vt);
    del_array(k);
    return P;
}

Array *householder(Victor *x, float *beta)
{
    Victor *v = __householder_v(x, beta);
    Array *res = __householder_a(v, beta[0]);
    return res;
}

float *givens(float a, float b)
{
    float *res = malloc(2*sizeof(float));
    if (b == 0){
        res[0] = 1;
        res[1] = 0;
    }
    else{
        if (fabs(b) > fabs(a)){
            float x = -a / b;
            res[1] = 1 / (sqrt(1+x*x));
            res[0] = res[1] * x;
        }
        else{
            float x = -b /a;
            res[0] = 1 / (sqrt(1+x*x));
            res[1] = res[0] * x;
        }
    }
    return res;
}

Array *givens_rotate(Array *a, int i, int k, float c, float s)
{
    Array *res = copy_array(a);
    for (int j = 0; j < res->size[0]; ++j){
        float x = get_array_pixel(res, j, i);
        float y = get_array_pixel(res, j, k);
        change_array_pixel(res, j, i, c*x - s*y);
        change_array_pixel(res, j, k, s*x + c*y);
    }
    return res;
}

Array *__householder_QR(Array *a, Array *r, Array *q)
{
    Array *res = copy_array(a);
    for (int i = 0; i < res->size[1]; ++i){
        Array *x = slice_array(res, i+1, res->size[0], i+1, i+1);
        Victor *y = list_to_Victor(x->num, 1, x->data);
        float *beta = malloc(sizeof(float));
        Victor *v = __householder_v(y, beta);
        Array *house = __householder_a(v, beta[0]);

        Array *apart = slice_array(res, i, res->size[0], i, res->size[1]);
        Array *rpart = array_x_multiplication(house, apart);
        replace_part_array(res, rpart, i, i);
        if (i < res->size[0]){
            Victor *vpart = slice_Victor(v, 2, res->size[0]-i+1);
            for (int j = i+1; j < res->size[0]; ++j){
                change_array_pixel(res, j, i, vpart->data[j-i-1]);
            }
        }
    }
    return res;
}