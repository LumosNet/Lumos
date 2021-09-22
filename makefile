LINUX = 0
GPU=0
DEBUG=0

ARCH= 	-gencode arch=compute_35,code=sm_35 \
      	-gencode arch=compute_50,code=[sm_50,compute_50] \
      	-gencode arch=compute_52,code=[sm_52,compute_52] \
		-gencode arch=compute_61,code=[sm_61,compute_61]

VPATH=./src/tensor:./src/operator:./src/utils:./src/graph:./include/:./test
EXEC=main.exe
OBJDIR=./obj/

DEBUG = 0

CC=gcc
CPP=g++

LDFLAGS= -lm -pthread
COMMON= -Iinclude/ -Isrc/tensor -Isrc/operator -Isrc/utils -Isrc/graph -std=c99 -fopenmp
CFLAGS=-Wall -Wno-unused-result -Wno-unknown-pragmtensor -Wfatal-errors

ifeq ($(GPU), 1)
COMMON+= -DGPU -I/usr/local/cuda/include/
CFLAGS+= -DGPU
LDFLAGS+= -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand
endif

TENSOR=array.o list.o tensor.o victor.o session.o
OPERATOR=active.o cluster.o loss.o
GRAPH=gray_process.o image.o im2col.o
NETWORK=convolutional_layer.o network.o
UTIL=umath.o parser.o utils.o
EXECOBJA=main.o

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
TENSORS = $(addprefix $(OBJDIR), $(TENSOR))
OPERATORS = $(addprefix $(OBJDIR), $(OPERATOR))
GRAPHS = $(addprefix $(OBJDIR), $(GRAPH))
NETWORKS = $(addprefix $(OBJDIR), $(NETWORK))
UTILS = $(addprefix $(OBJDIR), $(UTIL))
DEPS = $(wildcard src/*.h) makefile

all: obj $(EXEC)
#all: obj $(EXEC)

$(EXEC): $(TENSORS) $(OPERATORS) $(GRAPHS) $(NETWORK) $(UTILS) $(EXECOBJ)
	$(CC) $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(OBJDIR)%.o: %.cpp $(DEPS)
	$(CPP) $(COMMON) $(CFLAGS) -c $< -o $@

$(OBJDIR)%.o: %.c $(DEPS)
	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@

$(OBJDIR)%.o: %.cu $(DEPS)
	$(NVCC) $(ARCH) $(COMMON) --compiler-options "$(CFLAGS)" -c $< -o $@

obj:
	mkdir obj

.PHONY: clean

clean:
	python clean.py