CC=g++
FC=gfortran
FLAGS=-O3 -lstdc++ -fopenmp -lcudart -lcusparse -lcublas -L/usr/local/cuda/lib64
INC=-I../../../build/inc
LIB=../../../build/lib/librocalution.so
OBJ=mmio.o rocalution_fortran.o rocalution_solver.o

default: rocalution_solver

rocalution_solver: $(OBJ)
	$(FC) -o rocalution_solver $(OBJ) $(LIB) $(FLAGS) -Wall

rocalution_solver.o: rocalution_solver.f90
	$(FC) $(FLAGS) -c rocalution_solver.f90

mmio.o: mmio.f
	$(FC) $(FLAGS) -c mmio.f

rocalution_fortran.o: ../../plug-ins/rocalution_fortran.cpp
	$(CC) $(FLAGS) $(INC) -c ../../plug-ins/rocalution_fortran.cpp

clean:
	rm -rf *.o rocalution_solver
