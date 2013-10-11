#!/bin/bash

echo '~~~~~~~~~Compiling the code~~~~~~~~'

g++ `pkg-config --cflags opencv` -g -o dlt dlt.cpp `pkg-config --libs opencv`

./ransac < file4
