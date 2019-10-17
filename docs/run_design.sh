#!/bin/bash

if [ -d docBin ]; then
    rm -rf docBin
fi

doxygen Doxyfile

cd design
make clean
make latexpdf
make html
cd ..
