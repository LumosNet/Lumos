#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "cJSON.h"
#include "benchmark_json.h"

char *message = 
"{                              \
    \"origin\": {               \
        \"type\": \"float\",    \
        \"value\": [0.1, 0.2]   \
    },                          \
    \"bias\": {                 \
        \"type\": \"int\",      \
        \"value\": [1]          \
    }                           \
}";

int main(void)
{
    cJSON* cjson_test = NULL;
    // cJSON* cjson_name = NULL;
    // cJSON* cjson_age = NULL;
    // cJSON* cjson_weight = NULL;
    // cJSON* cjson_address = NULL;
    // cJSON* cjson_address_country = NULL;
    // cJSON* cjson_address_zipcode = NULL;
    // cJSON* cjson_array = NULL;
    // cJSON* cjson_skill = NULL;
    // cJSON* cjson_student = NULL;
    // int    skill_array_size = 0, i = 0;
    // cJSON* cjson_skill_item = NULL;
    // cJSON* cjson_array_item = NULL;

    // /* 解析整段JSO数据 */
    cjson_test = cJSON_Parse(message);
    if(cjson_test == NULL)
    {
        printf("parse fail.\n");
        return -1;
    }

    // /* 依次根据名称提取JSON数据（键值对） */
    // cjson_name = cJSON_GetObjectItem(cjson_test, "name");
    // cjson_age = cJSON_GetObjectItem(cjson_test, "age");
    // cjson_weight = cJSON_GetObjectItem(cjson_test, "weight");

    // printf("name: %s\n", cjson_name->valuestring);
    // printf("age:%d\n", cjson_age->valueint);
    // printf("weight:%.1f\n", cjson_weight->valuedouble);

    // /* 解析嵌套json数据 */
    // cjson_address = cJSON_GetObjectItem(cjson_test, "address");
    // cjson_address_country = cJSON_GetObjectItem(cjson_address, "country");
    // cjson_address_zipcode = cJSON_GetObjectItem(cjson_address, "zip-code");
    // printf("address-country:%s\naddress-zipcode:%d\n", cjson_address_country->valuestring, cjson_address_zipcode->valueint);

    // /* 解析数组 */
    // cjson_skill = cJSON_GetObjectItem(cjson_test, "skill");
    // skill_array_size = cJSON_GetArraySize(cjson_skill);
    // printf("skill:[");
    // for(i = 0; i < skill_array_size; i++)
    // {
    //     cjson_skill_item = cJSON_GetArrayItem(cjson_skill, i);
    //     printf("%s,", cjson_skill_item->valuestring);
    // }
    // printf("\b]\n");

    // /* 解析布尔型数据 */
    // cjson_student = cJSON_GetObjectItem(cjson_test, "student");
    // if(cjson_student->valueint == 0)
    // {
    //     printf("student: false\n");
    // }
    // else
    // {
    //     printf("student:error\n");
    // }

    // cjson_array = cJSON_GetObjectItem(cjson_test, "array");
    // int float_array_size = cJSON_GetArraySize(cjson_array);
    // printf("array:[");
    // for(i = 0; i < float_array_size; i++)
    // {
    //     cjson_array_item = cJSON_GetArrayItem(cjson_array, i);
    //     double x = cjson_array_item->valuedouble;
    //     printf("%f,", x);
    // }
    // printf("\b]\n");
    void **params = malloc(2*sizeof(void*));
    load_param(cjson_test, "origin", params, 0);
    load_param(cjson_test, "bias", params, 1);
    float *a = (float*)params[0];
    int *b = (int*)params[1];
    printf("%f %f\n", a[0], a[1]);
    printf("%d\n", b[0]);
    return 0;
}
