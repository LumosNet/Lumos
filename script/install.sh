#!/bin/bash

VERSION=v0.1
CDIR=`pwd`

INSTALLDIR=/usr/local/lumos

if [ -d "$INSTALLDIR" ]; then
    rm -rf "$INSTALLDIR"
fi

ARCHIVE=`awk '/^__ARCHIVE_BOUNDARY__/ { print NR + 1; exit 0; }' $0`

