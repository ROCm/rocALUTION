cd paralution-x.y.z

mkdir build
cd build

cmake -DSUPPORT_MIC=ON -DSUPPORT_CUDA=OFF -DSUPPORT_OCL=OFF -DSUPPORT_MPI=ON ..
make -j
