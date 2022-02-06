LINUX=1
GPU=0
DEBUG=1

ARCH= 	-gencode arch=compute_35,code=sm_35 \
      	-gencode arch=compute_50,code=[sm_50,compute_50] \
      	-gencode arch=compute_52,code=[sm_52,compute_52] \
		-gencode arch=compute_61,code=[sm_61,compute_61]

VPATH=./src/:./test/:./scripts
EXEC=main.exe
OBJDIR=./obj/

CC=gcc
CPP=g++
NVCC=nvcc

LDFLAGS= -lm -pthread
COMMON= -Iinclude/ -Isrc -Iscripts -std=c99 -fopenmp
CFLAGS=-Wall -Wno-unused-result -Wno-unknown-pragmtensor -Wfatal-errors

ifeq ($(GPU), 1)
COMMON+= -DGPU -I/usr/local/cuda/include/
CFLAGS+= -DGPU
LDFLAGS+= -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand
endif

ifeq ($(DEBUG), 1)
COMMON+= -g
endif

OBJ=active.o array.o avgpool_layer.o bias.o cluster.o connect_layer.o \
	convolutional_layer.o data.o gemm.o gray_process.o im2col.o image.o list.o loss.o maxpool_layer.o \
	network.o parser.o pooling_layer.o softmax_layer.o tensor.o umath.o utils.o vector.o \
	weights.o im2col_layer.o debug.o mse_layer.o
EXECOBJA=main.o

ifeq ($(GPU), 1)
LDFLAGS+= -lstdc++
endif

ifeq ($(LINUX),1)
CFLAGS += -fPIC
endif

EXECOBJ = $(addprefix $(OBJDIR), $(EXECOBJA))
OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard src/*.h) makefile include/lumos.h

all: obj results $(EXEC)
#all: obj $(EXEC)

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
results:
	mkdir results

.PHONY: clean

clean:
	python clean.py
