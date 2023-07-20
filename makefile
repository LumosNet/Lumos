LINUX=1
TEST=0

ARCH=	-gencode arch=compute_50,code=[sm_50,compute_50] \
      	-gencode arch=compute_52,code=[sm_52,compute_52] \
		-gencode arch=compute_61,code=[sm_61,compute_61]

VPATH=	./lib/: \
		./lumos/core/: \
		./lumos/core/graph/: \
		./lumos/core/graph/layer/: \
		./lumos/core/graph/loss_layer/: \
		./lumos/core/ops/: \
		./lumos/core/session/: \
		./lumos/utils/: \
		./lumos/utils/cmd/: \
		./lumos/utils/file/: \
		./lumos/utils/str/: \
		./: \
		./lumos/core_cu/: \
		./lumos/core_cu/graph_cu/: \
		./lumos/core_cu/ops_cu/: \
		./lumos/core_cu/graph_cu/layer_cu/: \
		./lumos/core_cu/graph_cu/loss_layer_cu/

COMMON=	-Ilib \
		-Iinclude \
		-Ilumos/core/graph \
		-Ilumos/core/graph/layer \
		-Ilumos/core/graph/loss_layer \
		-Ilumos/core/ops \
		-Ilumos/core/session \
		-Ilumos/utils/cmd \
		-Ilumos/utils/file \
		-Ilumos/utils/str \
		-Ilumos/core_cu \
		-Ilumos/core_cu/graph_cu \
		-Ilumos/core_cu/ops_cu \
		-Ilumos/core_cu/graph_cu/layer_cu \
		-Ilumos/core_cu/graph_cu/loss_layer_cu

ifeq ($(TEST), 0)
COMMON += -Ilumos/lumos

VPATH += ./lumos/lumos
endif

EXEC=lumos.exe
OBJDIR=./obj/

CC=gcc
NVCC=nvcc

LDFLAGS= -lm -pthread
COMMON+= -Iscripts
CFLAGS=-fopenmp -Wall -Wno-unused-result -Wno-unknown-pragmtensor -Wfatal-errors -m64

COMMON+= -DGPU -I/usr/local/cuda/include/
CFLAGS+= -DGPU -Wno-deprecated-gpu-targets
LDFLAGS+= -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand

ifeq ($(TEST), 1)
COMMON+= -Ilumos_t \
		 -Ilumos_t/core/ops \
		 -Ilumos_t/core/graph \
		 -Ilumos_t/tool \
		 -Ilumos_t/core_cu/ops \
		 -Ilumos_t/core_cu/graph

VPATH+=	./lumos_t \
		./lumos_t/tool \
		./lumos_t/core \
		./lumos_t/core/graph \
		./lumos_t/core/ops \
		./lumos_t/core_cu/ops \
		./lumos_t/core_cu/graph
endif

OBJ=	avgpool_layer.o connect_layer.o convolutional_layer.o graph.o im2col_layer.o maxpool_layer.o softmax_layer.o dropout_layer.o shortcut_layer.o \
		mse_layer.o normalization_layer.o weights_init.o \
		active.o bias.o cpu.o gemm.o im2col.o image.o pooling.o random.o softmax.o shortcut.o normalize.o \
		session.o manager.o dispatch.o \
		progress_bar.o \
		binary_f.o text_f.o \
		str_ops.o \
		cJSON_Utils.o cJSON.o

OBJ+= 	gpu.o active_gpu.o bias_gpu.o cpu_gpu.o gemm_gpu.o im2col_gpu.o pooling_gpu.o softmax_gpu.o shortcut_gpu.o normalize_gpu.o \
	  	avgpool_layer_gpu.o maxpool_layer_gpu.o connect_layer_gpu.o convolutional_layer_gpu.o im2col_layer_gpu.o \
	  	mse_layer_gpu.o dropout_layer_gpu.o softmax_layer_gpu.o shortcut_layer_gpu.o normalization_layer_gpu.o

EXECOBJA=lumos.o

ifeq ($(TEST), 1)
OBJ+=   bias_call.o cpu_call.o gemm_call.o im2col_call.o image_call.o pooling_call.o \
	    avgpool_layer_call.o batchnorm_layer_call.o connect_layer_call.o convolutional_layer_call.o im2col_layer_call.o maxpool_layer_call.o \
	    mse_layer_call.o \
	    utest.o benchmark_json.o call.o compare.o tsession.o

OBJ+= bias_gpu_call.o cpu_gpu_call.o gemm_gpu_call.o im2col_gpu_call.o pooling_gpu_call.o \
	  avgpool_layer_gpu_call.o connect_layer_gpu_call.o convolutional_layer_gpu_call.o im2col_layer_gpu_call.o maxpool_layer_gpu_call.o
endif

ifeq ($(TEST),1)
COMMON+= -DLUMOST
CFLAGS+= -DLUMOST
endif

ifeq ($(LINUX),1)
CFLAGS += -fPIC
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
	$(NVCC) $(ARCH) $(COMMON) --compiler-options "$(CFLAGS)" -c $< -o $@

obj:
	mkdir obj

.PHONY: clean

clean:
	rm -rf obj
	rm -rf $(EXEC)
