#!/bin/bash

mkdir -p bin
find . -name '*.cpp' | sed 's,\.cpp,,' | xargs -i g++ {}.cpp -o bin/{} -O3
