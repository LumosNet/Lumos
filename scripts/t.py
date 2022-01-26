import math

a = [32.391186,
    -2457.479004,
    2475.894775,
    6627.136719,
    -5105.113770,
    461.495148,
    923.064880,
    3308.588379,
    1552.206665,
    1556.081543]

res = []
sum = 0

for i in a:
    b = math.e**i
    res.append(b)
    sum += b

for i in res:
    print(i/sum)