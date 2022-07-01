LINUX=1
GPU=0
DEBUG=0
TEST=1
AST=0

ARCH= 	-gencode arch=compute_35,code=sm_35 \
      	-gencode arch=compute_50,code=[sm_50,compute_50] \
      	-gencode arch=compute_52,code=[sm_52,compute_52] \
		-gencode arch=compute_61,code=[sm_61,compute_61]

# 源代码所在目录（包括所有子目录）
VPATH=./src/: \
	  ./src/core/: \
	  ./src/core/cu_ops/: \
	  ./src/core/graph/: \
	  ./src/core/graph/layer/: \
	  ./src/core/graph/loss_layer/: \
	  ./src/core/ops/: \
	  ./src/core/session/: \
	  ./src/dependent/: \
	  ./src/dependent/cmd/: \
      ./src/dependent/file/: \
      ./src/dependent/str/: \
      ./ \
	  ./test/

COMMON=-Isrc/core/graph \
	   -Isrc/core/graph/layer \
	   -Isrc/core/graph/loss_layer \
	   -Isrc/core/ops \
	   -Isrc/core/session \
	   -Isrc/dependent/cmd \
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

ifeq ($(TEST), 1)
COMMON+= -Itest
endif

OBJ=	avgpool_layer.o connect_layer.o convolutional_layer.o graph.o im2col_layer.o layer.o maxpool_layer.o \
		mse_layer.o \
		active.o bias.o cpu.o gemm.o im2col.o image.o \
		session.o manager.o dispatch.o\
		progress_bar.o \
		binary_f.o cfg_f.o text_f.o \
		str_ops.o \

EXECOBJA=main.o

ifeq ($(TEST), 1)
OBJ+= test.o
endif

ifeq ($(GPU), 1)
LDFLAGS+= -lstdc++
endif

ifeq ($(LINUX),1)
CFLAGS += -fPIC
endif

ifeq ($(AST),1)
COMMON+= -fdump-tree-all-graph
endif

EXECOBJ = $(addprefix $(OBJDIR), $(EXECOBJA))
OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = makefile

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
	python3 clean.py
