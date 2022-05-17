LINUX=1
GPU=0
DEBUG=1

ARCH= 	-gencode arch=compute_35,code=sm_35 \
      	-gencode arch=compute_50,code=[sm_50,compute_50] \
      	-gencode arch=compute_52,code=[sm_52,compute_52] \
		-gencode arch=compute_61,code=[sm_61,compute_61]

# 源代码所在目录（包括所有子目录）
VPATH=./src/: \
      ./src/core/: \
      ./src/core/component/: \
      ./src/core/component/layer/: \
      ./src/core/component/network/: \
      ./src/core/ops/: \
      ./src/dependent/: \
      ./src/dependent/file/: \
      ./src/dependent/str/: \
      ./demo/: \
      ./

COMMON=-Isrc \
       -Isrc/core \
	   -Isrc/core/component \
	   -Isrc/core/component/layer \
	   -Isrc/core/component/network \
	   -Isrc/core/ops \
	   -Isrc/dependent \
	   -Isrc/dependent/file \
	   -Isrc/dependent/str \

EXEC=main.exe
OBJDIR=./obj/

CC=gcc
CPP=g++
NVCC=nvcc

LDFLAGS= -lm -pthread
COMMON+= -Ilib/ -Iinclude/ -Iscripts -std=c99 -fopenmp
CFLAGS=-Wall -Wno-unused-result -Wno-unknown-pragmtensor -Wfatal-errors

ifeq ($(GPU), 1)
COMMON+= -DGPU -I/usr/local/cuda/include/
CFLAGS+= -DGPU
LDFLAGS+= -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand
endif

ifeq ($(DEBUG), 1)
COMMON+= -g
endif

OBJ=	avgpool_layer.o connect_layer.o convolutional_layer.o im2col_layer.o \
		maxpool_layer.o mse_layer.o pooling_layer.o softmax_layer.o \
		network.o \
		active.o bias.o cpu.o gemm.o im2col.o image.o \
		read_f.o \
		parser.o str_ops.o \
		data.o umath.o
EXECOBJA=xor.o

ifeq ($(GPU), 1)
LDFLAGS+= -lstdc++
endif

ifeq ($(LINUX),1)
CFLAGS += -fPIC
endif

EXECOBJ = $(addprefix $(OBJDIR), $(EXECOBJA))
OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard src/*.h) makefile include/lumos.h

all: obj backup $(EXEC)

$(EXEC): $(OBJS) $(EXECOBJ)
	$(CC) $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(OBJDIR)%.o: %.cpp $(DEPS)
	$(CPP) $(COMMON) $(CFLAGS) -c $< -o $@

$(OBJDIR)%.o: %.c $(DEPS)
	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@

$(OBJDIR)%.o: %.cu $(DEPS)
	$(NVCC) $(ARCH) $(COMMON) --compiler-options "$(CFLAGS)" -c $< -o $@

obj:
	mkdir obj
backup:
	mkdir backup

.PHONY: clean

clean:
	python clean.py
