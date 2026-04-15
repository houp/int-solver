#!/bin/bash

## THIS WORKS ON macOS only with OpenMP installed via homebrew (brew install libomp) or Linux with g++
## Make sure to have c++ / g++ installed 


OS="$(uname -s)"

echo "Building $1.cpp and saving output to $1"

if [[ "$OS" == "Linux" ]]; then
    g++ $1.cpp -fopenmp -O3 -march=native --std=gnu++23 -Wall -o $1
elif [[ "$OS" == "Darwin" ]]; then
    c++ -Xclang -fopenmp -L/opt/homebrew/opt/libomp/lib -I/opt/homebrew/opt/libomp/include -lomp $1.cpp -Wall -O3 -march=native --std=gnu++23 -o $1
else
    echo "System not supported"
fi

