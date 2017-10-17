CC=g++
FC=gfortran
FLAGS=-O3 -lstdc++ -fopenmp -lcudart -lcusparse -lcublas -lOpenCL -L/usr/local/cuda/lib64
INC=-I../../../build/inc
LIB=../../../build/lib/libparalution.so
OBJ=mmio.o paralution_fortran.o paralution_solver.o

default: paralution_solver

paralution_solver: $(OBJ)
	$(FC) -o paralution_solver $(OBJ) $(LIB) $(FLAGS) -Wall

paralution_solver.o: paralution_solver.f90
	$(FC) $(FLAGS) -c paralution_solver.f90

mmio.o: mmio.f
	$(FC) $(FLAGS) -c mmio.f

paralution_fortran.o: ../../plug-ins/paralution_fortran.cpp
	$(CC) $(FLAGS) $(INC) -c ../../plug-ins/paralution_fortran.cpp

clean:
	rm -rf *.o paralution_solver
