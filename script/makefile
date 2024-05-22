CUDAINCLUDE=/usr/local/cuda/include/
CUDALIB=/usr/local/cuda/lib

ARCH=	-gencode arch=compute_52,code=[sm_52,compute_52] \
		-gencode arch=compute_61,code=[sm_61,compute_61]

VPATH=	./lib/: \
		./lumos/core/: \
		./lumos/core/graph/: \
		./lumos/core/graph/layer/: \
		./lumos/core/graph/loss_layer/: \
		./lumos/core/ops/: \
		./lumos/core/session/: \
		./utils/: \
		./utils/cmd/: \
		./utils/file/: \
		./utils/str/: \
		./utils/logging/: \
		./: \
		./lumos/core_cu/: \
		./lumos/core_cu/graph_cu/: \
		./lumos/core_cu/ops_cu/: \
		./lumos/core_cu/graph_cu/layer_cu/: \
		./lumos/core_cu/graph_cu/loss_layer_cu/

COMMON=	-Ilib \
		-Ilumos/core/graph \
		-Ilumos/core/graph/layer \
		-Ilumos/core/graph/loss_layer \
		-Ilumos/core/ops \
		-Ilumos/core/session \
		-Iutils/cmd \
		-Iutils/file \
		-Iutils/str \
		-Iutils/logging \
		-Ilumos/core_cu \
		-Ilumos/core_cu/graph_cu \
		-Ilumos/core_cu/ops_cu \
		-Ilumos/core_cu/graph_cu/layer_cu \
		-Ilumos/core_cu/graph_cu/loss_layer_cu

CC=gcc
NVCC=nvcc

LIBLUMOS=./build/lib/liblumos.so

OBJDIR=./build/obj/
LIBDIR=./build/lib/

LDFLAGS= -lm -pthread
COMMON+= -Iscripts
CFLAGS=-fopenmp -Wall -Wno-unused-result -Wno-unknown-pragmtensor -Wfatal-errors -m64

COMMON+= -DGPU -I$(CUDAINCLUDE)
CFLAGS+= -DGPU -Wno-deprecated-gpu-targets
LDFLAGS+= -L$(CUDALIB) -lcudart -lcublas -lcurand

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
DEPS = makefile

all: $(LIBLUMOS) $(LIBLUMOSCU)

$(LIBLUMOS): $(OBJS)
	$(CC) $(CFLAGS) -shared $^ -o $@ $(LDFLAGS)

$(OBJDIR)%.o: %.c $(DEPS)
	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@

$(OBJDIR)%.o: %.cu $(DEPS)
	$(NVCC) $(ARCH) $(COMMONGPU) --compiler-options "$(CFLAGS)" -c $< -o $@
