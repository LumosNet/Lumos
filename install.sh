#!/bin/bash

lines=39
tail +$lines $0 >./lumos.tar.gz

echo "unpack: lumos.tar.gz"
tar -zxvf ./lumos.tar.gz
echo "unpack finished"
wait

mv ./lumos-build/include ./lumos-include
mv ./lumos-build/lib ./lumos-lib
mv ./lumos-build/src ./lumos-src
mv ./lumos-build/makefile ./lumos-makefile

mkdir ./lumos-build/lib
mkdir ./lumos-build/bin
mkdir ./lumos-build/include

make -f lumos-makefile
wait

mv ./lumos-include/lumos.h ./lumos-build/include/lumos.h

rm -rf /usr/local/lumos
mkdir /usr/local/lumos
cp -r ./lumos-build/bin /usr/local/lumos/bin
cp -r ./lumos-build/include /usr/local/lumos/include
cp -r ./lumos-build/lib /usr/local/lumos/lib

rm -rf ./lumos-build
rm -rf ./lumos-include
rm -rf ./lumos-lib
rm -rf ./lumos-src
rm -rf ./lumos-obj
rm -f ./lumos-makefile

exit 0
