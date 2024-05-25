### ECE-GY 9143 - High Performance Computing for Machine Learning
### Instructor: Parijat Dubey
### Makefile for Part B Assignment 5

SDK_INSTALL_PATH :=  /share/apps/cuda/11.3.1
NVCC=$(SDK_INSTALL_PATH)/bin/nvcc
LIB       :=  -L$(SDK_INSTALL_PATH)/lib64 -L$(SDK_INSTALL_PATH)/samples/common/lib/linux/x86_64
#INCLUDES  :=  -I$(SDK_INSTALL_PATH)/include -I$(SDK_INSTALL_PATH)/samples/common/inc
OPTIONS   :=  -O3 
#--maxrregcount=100 --ptxas-options -v 

TAR_FILE_NAME  := ar7996_PartB_CUDA1.tar
EXECS :=  q2 q3
all:$(EXECS)

#######################################################################
clean:
	rm -f $(EXECS) *.txt *.png

#######################################################################
tar:
	tar -cvf $(TAR_FILE_NAME) Makefile *.h *.cu *.pdf *.txt
#######################################################################

q2 : q2.cu
	${NVCC} $< -o $@ $(LIB) $(OPTIONS)

q3 : q3.cu
	${NVCC} $< -o $@ $(LIB) $(OPTIONS)




