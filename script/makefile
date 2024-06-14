CUDAINCLUDE=/usr/local/cuda/include/
CUDALIB=/usr/local/cuda/lib

ARCH=	-gencode arch=compute_52,code=[sm_52,compute_52] \
		-gencode arch=compute_61,code=[sm_61,compute_61]

VPATH=	./build/lib/: \
		./build/lumos/: \
		./build/lumos/core/: \
		./build/lumos/core/graph/: \
		./build/lumos/core/graph/layer/: \
		./build/lumos/core/graph/loss_layer/: \
		./build/lumos/core/ops/: \
		./build/lumos/core/session/: \
		./build/utils/: \
		./build/utils/cmd/: \
		./build/utils/file/: \
		./build/utils/str/: \
		./build/utils/logging/: \
		./build/: \
		./build/lumos/core_cu/: \
		./build/lumos/core_cu/graph_cu/: \
		./build/lumos/core_cu/ops_cu/: \
		./build/lumos/core_cu/graph_cu/layer_cu/: \
		./build/lumos/core_cu/graph_cu/loss_layer_cu/

COMMON=	-Ibuild/lib \
		-Ibuild/lumos/core/graph \
		-Ibuild/lumos/core/graph/layer \
		-Ibuild/lumos/core/graph/loss_layer \
		-Ibuild/lumos/core/ops \
		-Ibuild/lumos/core/session \
		-Ibuild/utils/cmd \
		-Ibuild/utils/file \
		-Ibuild/utils/str \
		-Ibuild/utils/logging \
		-Ibuild/lumos/core_cu \
		-Ibuild/lumos/core_cu/graph_cu \
		-Ibuild/lumos/core_cu/ops_cu \
		-Ibuild/lumos/core_cu/graph_cu/layer_cu \
		-Ibuild/lumos/core_cu/graph_cu/loss_layer_cu

CC=gcc
NVCC=nvcc

EXEC=./build/bin/lumos
LIBLUMOS=./build/lulib/liblumos.so

BINDIR=./build/bin/
OBJDIR=./build/obj/
LIBDIR=./build/lulib/

LDFLAGS= -lm -pthread
COMMON+= -Iscripts
CFLAGS=-fopenmp -Wall -Wno-unused-result -Wno-unknown-pragmtensor -Wfatal-errors -m64

COMMON+= -DGPU -I$(CUDAINCLUDE)
CFLAGS+= -DGPU -Wno-deprecated-gpu-targets
LDFLAGS+= -L$(CUDALIB) -lcudart -lcublas -lcurand

EXECOBJ= lumos.o

OBJ=	avgpool_layer.o connect_layer.o convolutional_layer.o graph.o im2col_layer.o maxpool_layer.o \
		softmax_layer.o \
		mse_layer.o \
		active.o bias.o cpu.o gemm.o im2col.o image.o pooling.o random.o softmax.o shortcut.o normalize.o \
		session.o \
		progress_bar.o \
		binary_f.o text_f.o \
		str_ops.o logging.o \
		active_gpu.o bias_gpu.o cpu_gpu.o gemm_gpu.o im2col_gpu.o pooling_gpu.o softmax_gpu.o shortcut_gpu.o normalize_gpu.o \
		avgpool_layer_gpu.o maxpool_layer_gpu.o connect_layer_gpu.o convolutional_layer_gpu.o im2col_layer_gpu.o \
		softmax_layer_gpu.o mse_layer_gpu.o

CFLAGS += -fPIC
COMMONGPU = $(COMMON)
COMMONGPU += --maxrregcount=64
LDFLAGS+= -lstdc++

OBJS = $(addprefix $(OBJDIR), $(OBJ))
EXECOBJS = $(addprefix $(OBJDIR), $(EXECOBJ))
DEPS = makefile

all: $(LIBLUMOS) $(EXEC)

$(EXEC): $(EXECOBJS)
	$(CC) $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(LIBLUMOS): $(OBJS)
	$(CC) $(CFLAGS) -shared $^ -o $@ $(LDFLAGS)

$(OBJDIR)%.o: %.c $(DEPS)
	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@

$(OBJDIR)%.o: %.cu $(DEPS)
	$(NVCC) $(ARCH) $(COMMONGPU) --compiler-options "$(CFLAGS)" -c $< -o $@
