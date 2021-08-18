LINUX = 0

VPATH=./src/:./include/:./test
EXEC=main.exe
OBJDIR=./obj/

DEBUG = 0

CC=gcc
CPP=g++

LDFLAGS= -lm
COMMON= -Iinclude/ -Isrc -std=c99
CFLAGS=-Wall -Wno-unused-result -Wno-unknown-pragmas -Wfatal-errors

OBJ=list.o algebraic_space.o array.o victor.o
EXECOBJA=array_test.o

EXECOBJ = $(addprefix $(OBJDIR), $(EXECOBJA))
OBJS = $(addprefix $(OBJDIR), $(OBJ))
DEPS = $(wildcard src/*.h) $(wildcard test/*.h) $(wildcard utils/*.h) makefile

ifeq ($(LINUX),1)
CFLAGS += -fPIC
endif

ifeq ($(DEBUG),1)
COMMON += -DDEBUG
endif

all: obj $(EXEC)
#all: obj $(EXEC)

$(EXEC): $(OBJS) $(EXECOBJ)
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