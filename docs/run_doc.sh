#!/bin/bash

if [ -d docBin ]; then
    rm -rf docBin
fi

doxygen Doxyfile

cd source
make clean
make html
cd ..
