VPATH=	./: \
		./include \
		./demo

COMMON=	-Iinclude \
		-Idemo

EXEC=main.exe
OBJDIR=./obj/

CC=gcc
NVCC=nvcc

LDFLAGS= -lm -pthread
COMMON+= -Iscripts
CFLAGS=-fopenmp -Wall -Wno-unused-result -Wno-unknown-pragmtensor -Wfatal-errors -m64

COMMON+= -DGPU -I/usr/local/cuda/include/
CFLAGS+= -DGPU -Wno-deprecated-gpu-targets
LDFLAGS+= -L/usr/local/cuda-12.4/targets/x86_64-linux/lib -lcudart -lcublas -lcurand
LDFLAGS+= -L./build/lib -llumos

OBJ=	xor.o lenet5_mnist.o lenet5_cifar10.o

EXECOBJA=main.o

CFLAGS += -fPIC
COMMONGPU = $(COMMON)
COMMONGPU += --maxrregcount=64
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
