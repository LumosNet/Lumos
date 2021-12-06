#include "array.h"

Tensor *array_d(int row, int col)
{
    int size[] = {col, row};
    Tensor *ret = tensor_d(2, size);
    ret->type = ARRAY;
    return ret;
}

/*************************************************************************************************\
 * 描述
       创建矩阵，初始化内存空间为x
 * 参数
       row:行数
       col:列数
       x:空间填充值
 * 返回
       Tensor*
       返回创建好的Tensor对象，您可以使用该Tensor对象进行一切与矩阵相关的操作
\*************************************************************************************************/
Tensor *array_x(int row, int col, float x)
{
    int size[] = {col, row};
    Tensor *ret = tensor_x(2, size, x);
    ret->type = ARRAY;
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
       Tensor*
       返回转换好的Tensor对象，您可以使用该Tensor对象进行一切与矩阵相关的操作
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
Tensor *array_list(int row, int col, float *list)
{
    int size[] = {col, row};
    Tensor *ret = tensor_list(2, size, list);
    ret->type = ARRAY;
    return ret;
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
       Tensor*
       返回创建好的矩阵对象Tensor，您可以使用该Tensor对象进行一切与矩阵相关的操作
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
Tensor *array_unit(int row, int col, float x, int flag)
{
    Tensor *ret = array_d(row, col);
    int min_row_col = (row <= col) ? row : col;
    for (int i = 0; i < min_row_col; ++i){
        if (flag) change_float_in_list(ret->data, x, i*col+i);
        else change_float_in_list(ret->data, x, i*col+(col-i-1));
    }
    ret->type = ARRAY;
    return ret;
}

Tensor *array_sparse(int row, int col, int **index, float *list, int n)
{
    int size[] = {col, row};
    Tensor *ret = tensor_sparse(2, size, index, list, n);
    return ret;
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
float ar_get_pixel(Tensor *ts, int row, int col)
{
    int index[] = {col, row};
    return ts_get_pixel(ts, index);
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
void ar_change_pixel(Tensor *ts, int row, int col, float x)
{
    int index[] = {col, row};
    ts_change_pixel(ts, index, x);
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
void resize_ar(Tensor *ts, int row, int col)
{
    int size[] = {col, row};
    resize_ts(ts, 2, size);
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
void replace_rowlist(Tensor *ts, int row, float *list)
{
    int size = ts->size[0];
    int index = row*size;
    memcpy_float_list(ts->data, list, index, 0, size);
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
void replace_collist(Tensor *ts, int col, float *list)
{
    for (int i = 0; i < ts->size[1]; ++i){
        ar_change_pixel(ts, i, col, list[i]);
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
void replace_diagonalist(Tensor *ts, float *list, int flag)
{
    int min = MIN(ts->size[0], ts->size[1]);
    for (int i = 0; i < min; ++i){
        if (flag) ar_change_pixel(ts, i, i, list[i]);
        else ar_change_pixel(ts, i, ts->size[0]-i-1, list[i]);
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
void replace_rowx(Tensor *ts, int row, float x)
{
    for (int i = 0; i < ts->size[0]; ++i){
        ar_change_pixel(ts, row, i, x);
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
void replace_colx(Tensor *ts, int col, float x)
{
    for (int i = 0; i < ts->size[1]; ++i){
        ar_change_pixel(ts, i, col, x);
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
void replace_diagonalx(Tensor *ts, float x, int flag)
{
    int min = MIN(ts->size[0], ts->size[1]);
    for (int i = 0; i < min; ++i){
        if (flag) ar_change_pixel(ts, i, i, x);
        else ar_change_pixel(ts, i, ts->size[0]-i-1, x);
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
void del_row(Tensor *ts, int row)
{
    ts->num -= ts->size[0];
    float *data = malloc((ts->num)*sizeof(float));
    memcpy_float_list(data, ts->data, 0, 0, row*ts->size[0]);
    memcpy_float_list(data, ts->data, row*ts->size[0], (row+1)*ts->size[0], (ts->size[1]-row-1)*ts->size[0]);
    free(ts->data);
    ts->data = data;
    ts->size[1] -= 1;
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
void del_col(Tensor *ts, int col)
{
    ts->num -= ts->size[1];
    float *data = malloc((ts->num)*sizeof(float));
    for (int i = 0; i < ts->size[1]; ++i){
        int offset = i*ts->size[0];
        memcpy_float_list(data, ts->data, i*(ts->size[0]-1), offset, col);
        memcpy_float_list(data, ts->data, i*(ts->size[0]-1)+col, offset+col+1, ts->size[0]-col-1);
    }
    free(ts->data);
    ts->data = data;
    ts->size[0] -= 1;
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
void insert_row(Tensor *ts, int row, float *data)
{
    ts->num += ts->size[0];
    float *new_data = malloc(ts->num*sizeof(float));
    memcpy_float_list(new_data, ts->data, 0, 0, row*ts->size[0]);
    memcpy_float_list(new_data, data, row*ts->size[0], 0, ts->size[0]);
    memcpy_float_list(new_data, ts->data, (row+1)*ts->size[0], row*ts->size[0], (ts->size[1]-row)*ts->size[0]);
    free(ts->data);
    ts->data = new_data;
    ts->size[1] += 1;
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
void insert_col(Tensor *ts, int col, float *data)
{
    ts->num += ts->size[1];
    float *new_data = malloc((ts->num)*sizeof(float));
    for (int i = 0; i < ts->size[1]; ++i){
        int offset = i*ts->size[0];
        memcpy_float_list(new_data, ts->data, i*(ts->size[0]+1), offset, col);
        new_data[i*(ts->size[0]+1)+col] = data[i];
        memcpy_float_list(new_data, ts->data, i*(ts->size[0]+1)+col+1, offset+col, (ts->size[0]-col));
    }
    free(ts->data);
    ts->data = new_data;
    ts->size[0] += 1;
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
void overturn_lr(Tensor *ts)
{
    int x = ts->size[1];
    int y = ts->size[0];
    float *data = malloc(ts->num*sizeof(float));
    for (int i = 0; i < x; ++i){
        for (int j = 0; j < y; ++j){
            int col_index = y-1-j;
            data[i*y+j] = ts->data[i*y+col_index];
        }
    }
    free(ts->data);
    ts->data = data;
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
void overturn_ud(Tensor *ts)
{
    int x = ts->size[1];
    int y = ts->size[0];
    float *data = malloc(ts->num*sizeof(float));
    for (int i = 0; i < x; ++i){
        int row_index = x-1-i;
        for (int j = 0; j < y; ++j){
            data[i*y+j] = ts->data[row_index*y+j];
        }
    }
    free(ts->data);
    ts->data = data;
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
void overturn_diagonal(Tensor *ts, int flag)
{
    float *data = malloc(ts->num*sizeof(float));
    for (int i = 0; i < ts->size[1]; ++i){
        for (int j = 0; j < ts->size[0]; ++j){
            if (flag) data[j*ts->size[1]+i] = ts->data[i*ts->size[0]+j];
            else data[(ts->size[0]-j-1)*ts->size[1]+ts->size[1]-i-1] = ts->data[i*ts->size[0]+j];
        }
    }
    int x = ts->size[0];
    ts->size[0] = ts->size[1];
    ts->size[1] = x;
    free(ts->data);
    ts->data = data;
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
void rotate_left(Tensor *ts, int k)
{
    k %= 4;
    if (k == 0) return;
    if (k == 1){
        int x = ts->size[1];
        int y = ts->size[0];
        float *data = malloc(x*y*sizeof(float));
        for (int i = 0; i < x; ++i){
            for (int j = 0; j < y; ++j){
                data[(y-j-1)*x+i] = ts->data[i*y+j];
            }
        }
        ts->size[1] = y;
        ts->size[0] = x;
        free(ts->data);
        ts->data = data;
    }
    if (k == 2){
        overturn_ud(ts);
        overturn_lr(ts);
    }
    if (k == 3){
        int x = ts->size[1];
        int y = ts->size[0];
        float *data = malloc(x*y*sizeof(float));
        for (int i = 0; i < x; ++i){
            for (int j = 0; j < y; ++j){
                data[j*x+x-i-1] = ts->data[i*y+j];
            }
        }
        ts->size[1] = y;
        ts->size[0] = x;
        free(ts->data);
        ts->data = data;
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
void rotate_right(Tensor *ts, int k)
{
    k %= 4;
    if (k == 0) return;
    if (k == 1){
        int x = ts->size[1];
        int y = ts->size[0];
        float *data = malloc(x*y*sizeof(float));
        for (int i = 0; i < x; ++i){
            for (int j = 0; j < y; ++j){
                data[j*x+x-i-1] = ts->data[i*y+j];
            }
        }
        ts->size[1] = y;
        ts->size[0] = x;
        free(ts->data);
        ts->data = data;
    }
    if (k == 2){
        overturn_ud(ts);
        overturn_lr(ts);
    }
    if (k == 3){
        int x = ts->size[1];
        int y = ts->size[0];
        float *data = malloc(x*y*sizeof(float));
        for (int i = 0; i < x; ++i){
            for (int j = 0; j < y; ++j){
                data[(y-j-1)*x+i] = ts->data[i*y+j];
            }
        }
        ts->size[1] = y;
        ts->size[0] = x;
        free(ts->data);
        ts->data = data;
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
void exchange2row(Tensor *ts, int rowx, int rowy)
{
    int y = ts->size[0];
    for (int i = 0; i < y; ++i){
        int n = ts->data[rowx*y+i];
        ts->data[rowx*y+i] = ts->data[rowy*y+i];
        ts->data[rowy*y+i] = n;
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
void exchange2col(Tensor *ts, int colx, int coly)
{
    int x = ts->size[1];
    int y = ts->size[0];
    for (int i = 0; i < x; ++i){
        int n = ts->data[i*y+colx];
        ts->data[i*y+colx] = ts->data[i*y+coly];
        ts->data[i*y+coly] = n;
    }
}

/*************************************************************************************************\
 * 描述
       获取矩阵某一行（以向量形式返回）
 * 参数
       a:待获取矩阵
       row:待获取的行（1开始）
 * 返回
       Tensor*
       获取的行数据组成的向量
\*************************************************************************************************/
void row2list(Tensor *ts, int row, float *space)
{
    int offset = row*ts->size[0];
    memcpy_float_list(space, ts->data, 0, offset, ts->size[0]);
}

/*************************************************************************************************\
 * 描述
       获取矩阵某一列（一向量形式返回）
 * 参数
       a:待获取矩阵
       col:待获取的列（1开始）
 * 返回
       Tensor*
       获取的列数据组成的向量
\*************************************************************************************************/
void col2list(Tensor *ts, int col, float *space)
{
    for (int i = 0; i < ts->size[0]; ++i){
        space[i] = ar_get_pixel(ts, i, col);
    }
}

/*************************************************************************************************\
 * 描述
       获取矩阵对角线元素（以列表形式返回）
 * 参数
       a:待获取矩阵
       flag:1代表主对角线，0代表斜对角线
 * 返回
       Tensor*
       获取的对角线数据向量
\*************************************************************************************************/
void diagonal2list(Tensor *ts, int flag, float *space)
{
    int min = MIN(ts->size[0], ts->size[1]);
    for (int i = 0; i < min; ++i){
        if (flag) space[i] = ar_get_pixel(ts, i, i);
        else space[i] = ar_get_pixel(ts, i, min-i-1);
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
void transposition(Tensor *ts)
{
    overturn_diagonal(ts, 1);
}

/*************************************************************************************************\
 * 描述
       高斯消元求矩阵的逆
 * 参数
       a：待求逆矩阵
 * 返回
       Tensor*
       矩阵a的逆矩阵
\*************************************************************************************************/
Tensor *inverse(Tensor *ts)
{
    int x = ts->size[1];
    int y = ts->size[0];
    int key = -1;
    Tensor *res = array_unit(x, x, 1, 1);
    for (int i = 0; i < x; ++i) {
        if (ts->data[i*y+i] == 0.0) {
            //交换行获得一个非零的对角元素
            for (int j = i + 1; j < x; ++j) {
                if (ts->data[j*y+i] != 0.0){
                    key = j;
                    break;
                }
            }
            if (key == -1) {
                return res;
            }
            // 交换矩阵两行
            exchange2row(ts, i, key);
            exchange2row(res, i, key);
        }
        float scalar = 1.0 / ts->data[i*y+i];
        row_multx(ts, i-1, scalar);
        row_multx(res, i-1, scalar);
        for (int k = 0; k < x; ++k) {
            if (i == k) {
                continue;
            }
            float shear_needed = -ts->data[k*y+i];
            // 行乘以一个系数加到另一行
            add_multrow2r(ts, i-1, k-1, shear_needed);
            add_multrow2r(res, i-1, k-1, shear_needed);
        }
        key = -1;
    }
    return res;
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
float trace(Tensor *ts)
{
    float res = (float)0;
    int min = MIN(ts->size[0], ts->size[1]);
    for (int i = 0; i < min; ++i){
        res += ar_get_pixel(ts, i, i);
    }
    return res;
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
void row_addx(Tensor *ts, int row, float x)
{
    int y = ts->size[0];
    for (int i = 0; i < y; ++i){
        ts->data[row*y+i] += x;
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
void col_addx(Tensor *ts, int col, float x)
{
    int m = ts->size[1];
    int n = ts->size[0];
    for (int i = 0; i < m; ++i){
        ts->data[i*n+col] += x;
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
void row_multx(Tensor *ts, int row, float x)
{
    int y = ts->size[0];
    for (int i = 0; i < y; ++i){
        ts->data[row*y+i] *= x;
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
void col_multx(Tensor *ts, int col, float x)
{
    int m = ts->size[1];
    int n = ts->size[0];
    for (int i = 0; i < m; ++i){
        ts->data[i*n+col] *= x;
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
void add_row2r(Tensor *ts, int row1, int row2)
{
    int y = ts->size[0];
    for (int i = 0; i < y; ++i){
        ts->data[row2*y+i] += ts->data[row1*y+i];
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
void add_col2c(Tensor *ts, int col1, int col2)
{
    int m = ts->size[1];
    int n = ts->size[0];
    for (int i = 0; i < m; ++i){
        ts->data[i*n+col2] += ts->data[i*n+col1];
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
void add_multrow2r(Tensor *ts, int row1, int row2, float x)
{
    int y = ts->size[0];
    for (int i = 0; i < y; ++i){
        ts->data[row2*y+i] += ts->data[row1*y+i] * x;
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
void add_multcol2c(Tensor *ts, int col1, int col2, float x)
{
    int m = ts->size[1];
    int n = ts->size[0];
    for (int i = 0; i < m; ++i){
        ts->data[i*n+col2] += ts->data[i*n+col1] * x;
    }
}

/*************************************************************************************************\
 * 描述
       实现矩阵之间的乘法
 * 参数
       a，b：矩阵乘的矩阵
 * 返回
       Tensor*
       矩阵乘的结果
\*************************************************************************************************/
void gemm(Tensor *ts_a, Tensor *ts_b, float *space)
{
    int x = ts_a->size[1];
    int y = ts_a->size[0];
    int z = ts_b->size[0];
    #pragma omp parallel for
    for (int i = 0; i < x; ++i){
        for (int k = 0; k < y; ++k){
            register float temp = ts_a->data[i*y+k];
            for (int j = 0; j < z; ++j){
                space[i*z+j] += temp * ts_b->data[k*z+j];
            }
        }
    }
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
float norm1_ar(Tensor *ts)
{
    float res = -9999;
    for (int i = 0; i < ts->size[0]; ++i){
        float sum = 0;
        for (int j = 0; j < ts->size[1]; ++j){
            sum += ar_get_pixel(ts, j, i);
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
float norm2_ar(Tensor *ts)
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
float infnorm_ar(Tensor *ts)
{
    float res = -9999;
    for (int i = 0; i < ts->size[1]; ++i){
        float sum = 0;
        for (int j = 0; j < ts->size[0]; ++j){
            sum += ar_get_pixel(ts, i, j);
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
float fronorm_ar(Tensor *ts)
{
    float res = 0;
    for (int i = 0; i < ts->num; ++i){
        res += ts->data[i] * ts->data[i];
    }
    res = sqrt(res);
    return res;
}

#ifdef LINEAR
/*************************************************************************************************\
 * 描述
       求householder向量
 * 参数
       x：求给定向量x对应的householder变换分解式中的householder向量以及β值
 * 返回
       Tensor*
       返回householder向量
\*************************************************************************************************/
Tensor *__householder_v(Tensor *x, float *beta)
{
    Tensor *m = slice_vt(x, 1, x->num);
    Tensor *mt = tensor_copy(m);
    transposition(mt);
	float theta = multiply_ar(mt, m)->data[0];
    Tensor *v = tensor_copy(x);
    // ts->data[0] = 1;
    beta[0] = 0;
    if (theta != 0){
        float u = sqrt(x->data[0]*x->data[0] + theta);
        if (x->data[0] <= 0) v->data[0] = x->data[0] - u;
        else v->data[0] = -theta / (x->data[0] + u);
        beta[0] = (2*v->data[0]*v->data[0]) / (theta + v->data[0]*v->data[0]);
        array_multx(v, -v->data[0]);
    }
    free_tensor(m);
    free_tensor(mt);
    return v;
}

Tensor *__householder_a(Tensor *v, float beta)
{
    Tensor *In = array_unit(v->num, v->num, 1, 1);
    Tensor *vt = tensor_copy(v);
    transposition(vt);
    Tensor *k = multiply_ar(v, vt);
    array_multx(k, beta);
    Tensor *P = subtract_ar(In, k);
    free_tensor(In);
    free_tensor(vt);
    free_tensor(k);
    return P;
}

Tensor *householder(Tensor *x, float *beta)
{
    Tensor *v = __householder_v(x, beta);
    Tensor *res = __householder_a(v, beta[0]);
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

Tensor *givens_rotate(Tensor *ts, int i, int k, float c, float s)
{
    Tensor *res = tensor_copy(a);
    for (int j = 0; j < res->size[0]; ++j){
        float x = ts_get_pixel_ar(res, j, i);
        float y = ts_get_pixel_ar(res, j, k);
        ts_change_pixel_ar(res, j, i, c*x - s*y);
        ts_change_pixel_ar(res, j, k, s*x + c*y);
    }
    return res;
}

Tensor *__householder_QR(Tensor *ts, Tensor *r, Tensor *q)
{
    Tensor *res = tensor_copy(a);
    for (int i = 0; i < res->size[1]; ++i){
        Tensor *x = slice_array(res, i+1, res->size[0], i+1, i+1);
        Tensor *y = Tensor_list(x->num, 1, x->data);
        float *beta = malloc(sizeof(float));
        Tensor *v = __householder_v(y, beta);
        Tensor *house = __householder_a(v, beta[0]);

        Tensor *apart = slice_array(res, i, res->size[0], i, res->size[1]);
        Tensor *rpart = multiply_ar(house, apart);
        replace_part(res, rpart, i, i);
        if (i < res->size[0]){
            Tensor *vpart = slice_vt(v, 2, res->size[0]-i+1);
            for (int j = i+1; j < res->size[0]; ++j){
                ts_change_pixel_ar(res, j, i, vpart->data[j-i-1]);
            }
        }
    }
    return res;
}
#endif

#ifdef NUMPY
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
       Tensor*
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
Tensor *merge_array(Tensor *ts, Tensor *b, int flag, int index)
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
    Tensor *ret = array_x(size[1], size[0], 0);
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
       Tensor*
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
Tensor *slice_array(Tensor *ts, int rowu, int rowd, int coll, int colr)
{
    Tensor *ret = array_x(rowd-rowu+1, colr-coll+1, 0);
    int offset_row = coll-1;
    int size = colr-coll+1;
    for (int i = rowu-1, n = 0; i < rowd; ++i, ++n){
        int offset_a = i*a->size[0];
        int offset_r = n*ret->size[0];
        memcpy_float_list(ret->data, a->data, offset_r, offset_a+offset_row, size);
    }
    return ret;
}
#endif