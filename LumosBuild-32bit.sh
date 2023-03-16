#!/bin/bash
VERSION=v0.4a
CDIR=`pwd`

BUILDDIR=$CDIR/lumos-$VERSION

if [ -d "$BUILDDIR" ]; then
    rm -rf "$BUILDDIR"
fi

mkdir $BUILDDIR
mkdir $BUILDDIR/bin
mkdir $BUILDDIR/include
mkdir $BUILDDIR/lib
mkdir $BUILDDIR/data
mkdir $BUILDDIR/obj
wait

make -f LumosMake VERSION=$VERSION BYTE32=1 LINUX=1
wait

if [ -d "$BUILDDIR/obj" ]; then
    rm -rf "$BUILDDIR/obj"
fi

cp $CDIR/include/lumos.h $BUILDDIR/include/lumos.h
cp -r ./data/mnist ./lumos-$VERSION/data/mnist
cp -r ./data/xor ./lumos-$VERSION/data/xor
wait

tar zcvf lumos-$VERSION.tar.gz lumos-$VERSION
wait

if [ -d "$BUILDDIR" ]; then
    rm -rf "$BUILDDIR"
fi

cat LumosInstall.sh lumos-$VERSION.tar.gz > lumos-$VERSION.run
rm lumos-$VERSION.tar.gz
