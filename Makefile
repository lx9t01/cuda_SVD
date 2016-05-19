CC = g++
FLAGS = -Wall -g -std=c++11
EXENAME = cuda_SVD
INCLUDES = -I../lib -I/usr/include -I/usr/X11R6/include -I/usr/local/include

all: $(EXENAME)

$(EXENAME): cuda_SVD.o

cuda_SVD.o: cuda_SVD.cpp cuda_SVD.h
	$(CC) $(FLAGS) $(INCLUDES) -c cuda_SVD.cpp

clean: 
	$(RM) cuda_SVD *.o *~