#!/bin/bash

VERSION=v1.0
CDIR=`pwd`

INSTALLDIR=~/lumos
CUDAINCLUDE=NULL
CUDALIB=BULL

if [ -d "$INSTALLDIR" ]; then
    rm -rf "$INSTALLDIR"
fi

mkdir $INSTALLDIR

PATHES=$LD_LIBRARY_PATH
PATHES+=$PATH
PATHARR=(${PATHES//:/ })

for var in ${PATHARR[@]}
do
    if echo "$var" | grep -q 'cuda'; then
        ARR=(${var//// })
        ele=${ARR[$((${#ARR[@]}-1))]}
        if [[ $ele == *lib* ]]; then
            CUDALIB=$var
        fi
        if [[ $ele == *include* ]]; then
            CUDAINCLUDE=$var
        fi
    fi
done

if [[ $CUDAINCLUDE == *NULL* ]]; then
    echo "We can not find cuda include!"
    echo "Please make sure cuda include path is in system variable"
    exit 0
fi

if [[ $CUDALIB == *NULL* ]]; then
    echo "We can not find cuda lib!"
    echo "Please make sure cuda lib path is in system variable"
    exit 0
fi

ARCHIVE=`awk '/^__ARCHIVE_BOUNDARY__/ { print NR + 1; exit 0; }' $0`

tail -n +$ARCHIVE $0 > lumos.tar.gz
tar -zxvf lumos.tar.gz
wait

rm -f lumos.tar.gz

mv ./build/include $INSTALLDIR/include

mkdir ./build/lulib
mkdir ./build/obj
mkdir ./build/bin

make -f ./build/makefile CUDAINCLUDE=$CUDAINCLUDE CUDALIB=$CUDALIB
wait

mv ./build/lulib $INSTALLDIR/lib
mv ./build/bin $INSTALLDIR/bin

rm -rf build
rm -f lumos-$VERSION.run

echo "Installation Successful"

exit 0
__ARCHIVE_BOUNDARY__
