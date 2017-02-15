# file: Makefile
# description: Compiles and links convNet code.

# compiler: nvcc is NVIDIA's CUDA compiler
CC = /usr/local/cuda-7.5/bin/nvcc $(OSXOPTS)

# compiler options
# -O3: optimization flag
#-g nvcc debug  makes execution take 20x longer
#-lineinfo nvcc debug
#-Xcompiler -Q passes something to gcc c++ compiler
CCFLAGS = -arch=compute_20 -O3# -lineinfo

# variables
RM = rm
MAKE = make

# root directory of the project
# return the absolute path of the directory  where is located the Makefile
# variable MAKEFILE_LIST lists all Makefiles in the working directory
# `lastword` picks the last element of the list
PROJ_ROOT = $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

# source code directory
SRC_DIR = $(PROJ_ROOT)

# directory where object files are stored
BUILD_DIR = $(PROJ_ROOT)/build

# directory where binary executables are stored
BIN_DIR = $(PROJ_ROOT)/bin

# extension of source files
SRC_EXT = .cu

# convNet executable
TARGET = $(PROJ_ROOT)/bin/convNet

# list all source files in the source directory
SRCS = $(shell find $(SRC_DIR) -type f -name \*$(SRC_EXT))
# absolute path of all object files to be created
OBJS = $(patsubst $(SRC_DIR)/%, $(BUILD_DIR)/%, $(SRCS:$(SRC_EXT)=.o))

# include header files from convNet
INC = -I $(SRC_DIR)

#path of yaml static libary
EXT_LIBS = $(PROJ_ROOT)/external/lib/libyaml-cpp.a
# include YAML header files
INC += -I $(PROJ_ROOT)/external/yaml-cpp/include


.PHONY: all

all: $(TARGET)

$(TARGET): $(OBJS) $(EXT_LIBS)
	@echo "\nLinking ..."
	@mkdir -p $(BIN_DIR)
	$(CC) $^ -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%$(SRC_EXT)
	@mkdir -p $(@D)
	$(CC) $(CCFLAGS) $(INC) -c $< -o $@

################################################################################

.PHONY: doc

DOC_DIR = $(PROJ_ROOT)/doc
DOXYGEN = doxygen

doc:
	@echo "\nGenerating Doxygen documentation ..."
	cd $(DOC_DIR); $(DOXYGEN) Doxyfile

################################################################################

.PHONY: clean cleanexternal cleandoc cleanall

clean:
	@echo "\nCleaning convNet ..."
	$(RM) -rf $(BUILD_DIR) $(BIN_DIR)

cleandoc:
	@echo "\nCleaning documentation ..."
	find $(DOC_DIR) ! -name 'Doxyfile' -type f -delete
	find $(DOC_DIR)/* ! -name 'Doxyfile' -type d -delete

cleanall: clean cleanexternal cleandoc

################################################################################

# commands to run convNet
run:
	$(PROJ_ROOT)/bin/convNet