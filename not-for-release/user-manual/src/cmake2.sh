cd paralution-x.y.z

mkdir build
cd build

cmake -DSUPPORT_CUDA=ON -DSUPPORT_OMP=ON ..
make -j
