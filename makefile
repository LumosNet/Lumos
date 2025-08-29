LINUX=1
TEST=0
DEBUG=0
MEMDEBUG=0

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
		./utils/debug/: \
		./: \
		./lumos/core_cu/: \
		./lumos/core_cu/graph_cu/: \
		./lumos/core_cu/ops_cu/: \
		./lumos/core_cu/graph_cu/layer_cu/: \
		./lumos/core_cu/graph_cu/loss_layer_cu/ \

COMMON=	-Ilib \
		-Iinclude \
		-Ilumos/core/graph \
		-Ilumos/core/graph/layer \
		-Ilumos/core/graph/loss_layer \
		-Ilumos/core/ops \
		-Ilumos/core/session \
		-Iutils/cmd \
		-Iutils/file \
		-Iutils/str \
		-Iutils/logging \
		-Iutils/debug \
		-Ilumos/core_cu \
		-Ilumos/core_cu/graph_cu \
		-Ilumos/core_cu/ops_cu \
		-Ilumos/core_cu/graph_cu/layer_cu \
		-Ilumos/core_cu/graph_cu/loss_layer_cu \

EXEC=lumos.exe
OBJDIR=./obj/

CC=gcc
NVCC=nvcc

LDFLAGS= -lm -pthread
COMMON+= -Iscripts
CFLAGS=-fopenmp -Wall -Wno-unused-result -Wno-unknown-pragmtensor -Wfatal-errors -m64

COMMON+= -DGPU -I/usr/local/cuda/include/
CFLAGS+= -DGPU -Wno-deprecated-gpu-targets
LDFLAGS+= -L/usr/local/cuda-12.4/targets/x86_64-linux/lib -lcudart -lcublas -lcurand

ifeq ($(DEBUG), 1)
CFLAGS+= -g
endif

ifeq ($(MEMDEBUG),1)
CFLAGS+= -fsanitize=address
endif

ifeq ($(TEST), 1)
COMMON+= -Ilumos_t \
		 -Ilumos_t/core/ops \
		 -Ilumos_t/core/graph \
		 -Ilumos_t/tool \
		 -Ilumos_t/core_cu/ops \
		 -Ilumos_t/core_cu/graph \
		 -Ilumos_t/memory \
		 -Ilumos_t/memory_cu \
		 -Iutils/logging

VPATH+=	./lumos_t \
		./lumos_t/tool \
		./lumos_t/core \
		./lumos_t/core/graph \
		./lumos_t/core/ops \
		./lumos_t/memory \
		./lumos_t/core_cu/ops \
		./lumos_t/core_cu/graph \
		./lumos_t/memory_cu \
		./utils/logging
endif

ifeq ($(TEST), 0)
COMMON+= -Idemo

VPATH+= ./demo
endif

OBJ=	avgpool_layer.o connect_layer.o convolutional_layer.o graph.o maxpool_layer.o \
		softmax_layer.o dropout_layer.o normalization_layer.o \
		mse_layer.o mae_layer.o ce_layer.o\
		active.o bias.o cpu.o gemm.o im2col.o image.o pooling.o random.o softmax.o shortcut.o normalize.o \
		session.o optimize.o \
		progress_bar.o \
		binary_f.o text_f.o debug_data.o\
		str_ops.o logging.o

OBJ+= 	active_gpu.o bias_gpu.o cpu_gpu.o gemm_gpu.o im2col_gpu.o pooling_gpu.o softmax_gpu.o shortcut_gpu.o normalize_gpu.o \
	  	avgpool_layer_gpu.o maxpool_layer_gpu.o connect_layer_gpu.o convolutional_layer_gpu.o \
	  	softmax_layer_gpu.o dropout_layer_gpu.o normalization_layer_gpu.o \
		mse_layer_gpu.o mae_layer_gpu.o ce_layer_gpu.o

EXECOBJA=lumos.o

ifeq ($(TEST), 1)
OBJ+=   analysis_benchmark_file.o compare.o run_test.o test_msg.o utest.o cJSON.o cJSON_Utils.o logging.o

OBJ+= 	avgpool_layer_call.o connect_layer_call.o mse_layer_call.o

OBJ+= 	avgpool_layer_gpu_call.o connect_layer_gpu_call.o mse_layer_gpu_call.o

OBJ+=	im2col_call.o pooling_call.o

OBJ+=	layer_delta_call.o
endif

ifeq ($(TEST), 0)
OBJ+=	xor.o lenet5_mnist.o
endif

ifeq ($(TEST),1)
COMMON+= -DLUMOST
CFLAGS+= -DLUMOST
endif

ifeq ($(LINUX),1)
CFLAGS += -fPIC
endif

COMMONGPU = $(COMMON)
COMMONGPU += --maxrregcount=64

ifeq ($(DEBUG),1)
COMMONGPU += -g -G
endif

LDFLAGS+= -lstdc++

EXECOBJ = $(addprefix $(OBJDIR), $(EXECOBJA))
OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = makefile

all: obj $(EXEC)

$(EXEC): $(OBJS) $(EXECOBJ)
	$(CC) $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(OBJDIR)%.o: %.c $(DEPS)
	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@

$(OBJDIR)%.o: %.cu $(DEPS)
	$(NVCC) $(ARCH) $(COMMONGPU) --compiler-options "$(CFLAGS)" -c $< -o $@

obj:
	mkdir obj

.PHONY: clean

clean:
	rm -rf obj
	rm -rf $(EXEC)