LINUX=1
TEST=0

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
		./lumos/lumos/: \
		./

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
		-Ilumos/lumos

EXEC=main.exe
OBJDIR=./obj/

CC=gcc

LDFLAGS= -lm -pthread
COMMON+= -Iscripts
CFLAGS=-fopenmp -Wall -Wno-unused-result -Wno-unknown-pragmtensor -Wfatal-errors

ifeq ($(TEST), 1)
COMMON+= -Ilumos_t \
		 -Ilumos_t/core/ops \
		 -Ilumos_t/core/graph \
		 -Ilumos_t/tool
VPATH+=	./lumos_t \
		./lumos_t/tool \
		./lumos_t/core \
		./lumos_t/core/graph \
		./lumos_t/core/ops
endif

OBJ=	avgpool_layer.o connect_layer.o convolutional_layer.o graph.o im2col_layer.o maxpool_layer.o softmax_layer.o dropout_layer.o \
		mse_layer.o weights_init.o \
		active.o bias.o cpu.o gemm.o im2col.o image.o pooling.o random.o softmax.o \
		session.o manager.o dispatch.o \
		progress_bar.o \
		binary_f.o text_f.o \
		str_ops.o \
		cJSON_Utils.o cJSON.o \
		xor.o

EXECOBJA=main.o

ifeq ($(TEST), 1)
OBJ+= bias_call.o cpu_call.o gemm_call.o im2col_call.o image_call.o pooling_call.o \
	  avgpool_layer_call.o batchnorm_layer_call.o connect_layer_call.o convolutional_layer_call.o im2col_layer_call.o maxpool_layer_call.o \
	  mse_layer_call.o \
	  utest.o benchmark_json.o call.o compare.o tsession.o
endif

ifeq ($(TEST),1)
COMMON+= -DLUMOST
CFLAGS+= -DLUMOST
endif

ifeq ($(LINUX),1)
CFLAGS += -fPIC
endif

EXECOBJ = $(addprefix $(OBJDIR), $(EXECOBJA))
OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = makefile

all: obj $(EXEC)

$(EXEC): $(OBJS) $(EXECOBJ)
	$(CC) $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(OBJDIR)%.o: %.c $(DEPS)
	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@

obj:
	mkdir obj

.PHONY: clean

clean:
	rm -rf obj
	rm -rf lumos.exe
