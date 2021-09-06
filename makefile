LINUX = 0

VPATH=./src/tensor:./src/operator:./include/:./test
EXEC=main.exe
OBJDIR=./obj/

DEBUG = 0

CC=gcc
CPP=g++

LDFLAGS= -lm
COMMON= -Iinclude/ -Isrc/tensor -Isrc/operator -std=c99 -fopenmp
CFLAGS=-Wall -Wno-unused-result -Wno-unknown-pragmtensor -Wfatal-errors

TENSOR=array.o list.o tensor.o victor.o session.o
OPERATOR=active.o cluster.o loss.o
EXECOBJA=tensor_create.o

EXECOBJ = $(addprefix $(OBJDIR), $(EXECOBJA))
TENSORS = $(addprefix $(OBJDIR), $(TENSOR))
OPERATORS = $(addprefix $(OBJDIR), $(OPERATOR))
DEPS = $(wildcard src/*.h) $(wildcard test/*.h) $(wildcard utils/*.h) makefile

ifeq ($(LINUX),1)
CFLAGS += -fPIC
endif

ifeq ($(DEBUG),1)
COMMON += -DDEBUG
endif

all: obj $(EXEC)
#all: obj $(EXEC)

$(EXEC): $(TENSORS) $(OPERATORS) $(EXECOBJ)
	$(CC) $(COMMON) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(OBJDIR)%.o: %.c $(DEPS)
	$(CC) $(COMMON) $(CFLAGS) -c $< -o $@

obj:
	mkdir obj
	mkdir result
	mkdir backup

.PHONY: clean

clean:
	python clean.py