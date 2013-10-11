#!/bin/bash

echo '~~~~~~~~~Compiling the code~~~~~~~~'

g++ `pkg-config --cflags opencv` -g -o svd svd.cpp `pkg-config --libs opencv`

./ransac < file3
