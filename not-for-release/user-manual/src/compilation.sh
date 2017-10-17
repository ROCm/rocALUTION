g++ -O3 -Wall -I/path/to/paralution-x.y.z/build/inc -c main.cpp -o main.o
g++ -o main main.o -L/path/to/paralution-x.y.z/build/lib/ -lparalution

