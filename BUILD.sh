#!/bin/bash

rm -rf lumos-build
mkdir lumos-build
cp -r ./include ./lumos-build/include
cp -r ./lumos/lib ./lumos-build/lib
cp -r ./lumos ./lumos-build/src
cp -r ./Lumos-BUILD ./lumos-build/makefile

tar zcvf lumos.tar.gz ./lumos-build
cat install.sh lumos.tar.gz > lumos.run
