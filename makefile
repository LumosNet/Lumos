LINUX=1
GPU=1
DEBUG=0

ARCH= 	-gencode arch=compute_35,code=sm_35 \
      	-gencode arch=compute_50,code=[sm_50,compute_50] \
      	-gencode arch=compute_52,code=[sm_52,compute_52] \
		-gencode arch=compute_61,code=[sm_61,compute_61]

VPATH=./src/:./test
EXEC=main.exe
OBJDIR=./obj/

DEBUG = 0

CC=gcc
CPP=g++
NVCC=nvcc

LDFLAGS= -lm -pthread
COMMON= -Iinclude/ -Isrc -std=c99 -fopenmp
CFLAGS=-Wall -Wno-unused-result -Wno-unknown-pragmtensor -Wfatal-errors

ifeq ($(GPU), 1)
COMMON+= -DGPU -I/usr/local/cuda/include/
CFLAGS+= -DGPU
LDFLAGS+= -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand
endif

OBJ=array.o list.o tensor.o vector.o umath.o session.o active.o cluster.o loss.o gray_process.o image.o \
	im2col.o pooling.o network.o convolutional_layer.o pooling_layer.o activation_layer.o \
	umath.o parser.o bias.o softmax_layer.o connect_layer.o utils.o weights.o
EXECOBJA=tensor_test.o

ifeq ($(GPU), 1)
LDFLAGS+= -lstdc++
endif

ifeq ($(LINUX),1)
CFLAGS += -fPIC
endif

ifeq ($(DEBUG),1)
COMMON += -DDEBUG
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
