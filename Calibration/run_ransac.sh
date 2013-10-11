#!/bin/bash

echo '~~~~~~~~~Compiling the code~~~~~~~~'

g++ `pkg-config --cflags opencv` -g -o ransac ransac.cpp `pkg-config --libs opencv`

./ransac < file2
