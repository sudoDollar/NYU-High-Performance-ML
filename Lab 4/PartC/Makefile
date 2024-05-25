### ECE-GY 9143 - High Performance Computing for Machine Learning
### Instructor: Parijat Dubey
### Makefile for Part C Assignment 5

SDK_INSTALL_PATH :=  /share/apps/cuda/11.3.1
CUDNN_PATH:= /share/apps/cudnn/8.6.0.163-cuda11
NVCC=$(SDK_INSTALL_PATH)/bin/nvcc
LIB       :=  -L$(SDK_INSTALL_PATH)/lib64 -L$(SDK_INSTALL_PATH)/samples/common/lib/linux/x86_64 -L$(CUDNN_PATH)/lib
HEADERS := -I $(CUDNN_PATH)/include
#INCLUDES  :=  -I$(SDK_INSTALL_PATH)/include -I$(SDK_INSTALL_PATH)/samples/common/inc
OPTIONS   :=  -O3 
#--maxrregcount=100 --ptxas-options -v 

TAR_FILE_NAME  := ar7996_PartC_CUDA1.tar
EXECS :=  c1c2 c3
all:$(EXECS)

#######################################################################
clean:
	rm -f $(EXECS)

#######################################################################
tar:
	tar -cvf $(TAR_FILE_NAME) Makefile *.h *.cu *.pdf *.txt
#######################################################################

c1c2 : c1c2.cu
	${NVCC} $< -o $@ $(LIB) $(OPTIONS)

c3 : c3.cu
	${NVCC} $< -lcudnn -o $@ $(LIB) $(HEADERS) $(OPTIONS)

