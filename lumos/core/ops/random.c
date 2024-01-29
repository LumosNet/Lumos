#include "random.h"

float uniform_data(float a, float b, int *seed)
{
	float t;
	*seed = 2045.0 * (*seed) + 1;
	*seed = *seed - (*seed / 1048576) * 1048576;
	t = (*seed) / 1048576.0;
	t = a + (b - a) * t;
	return t;
}

float guass_data(float mean, float sigma, int *seed)
{
	int i;
	float x, y;
	for (x = 0, i = 0; i < 12; i++){
		x += uniform_data(0.0, 1.0, seed);
	}
	x = x - 6;
	y = mean + x * sigma;
	return y;
}

// void uniform_list(float a, float b, int seed, int num, float *space)
// {
// 	for (int i = 0; i < num; ++i){
// 		space[i] = uniform_data(a, b, &seed);
// 	}
// }

void guass_list(float mean, float sigma, int seed, int num, float *space)
{
    int sed[1] = {seed};
    for (int i = 0; i < num; ++i){
        space[i] = guass_data(mean, sigma, sed);
    }
}

void normal_list(int num, float *space)
{
	for (int i = 0; i < num; ++i){
		space[i] = rand_normal();
	}
}

void uniform_int_list(int a, int b, int num, float *space)
{
    for (int i = 0; i < num; ++i){
        space[i] = (int)rand_uniform(a, b);
    }
}

void uniform_list(float a, float b, int num, float *space)
{
	for (int i = 0; i < num; ++i){
		space[i] = rand_uniform(a, b);
	}
}

// From http://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
float rand_normal()
{
    static int haveSpare = 0;
    static double rand1, rand2;

    if(haveSpare)
    {
        haveSpare = 0;
        return sqrt(rand1) * sin(rand2);
    }

    haveSpare = 1;

    rand1 = rand() / ((double) RAND_MAX);
    if(rand1 < 1e-100) rand1 = 1e-100;
    rand1 = -2 * log(rand1);
    rand2 = (rand() / ((double) RAND_MAX)) * TWO_PI;

    return sqrt(rand1) * cos(rand2);
}

float rand_uniform(float min, float max)
{
    if(max < min){
        float swap = min;
        min = max;
        max = swap;
    }
    return ((float)rand()/RAND_MAX * (max - min)) + min;
}
