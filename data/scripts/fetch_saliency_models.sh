#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/../" && pwd )"
cd $DIR

FILE=senet.caffemodel
URL=https://drive.google.com/uc?id=16tT2FjFQFpAL7hXqbY5M30Jv6sZA3VJX
CHECKSUM=ac334dfa1684d543d33657a307d3056b

if [ -f $FILE ]; then
  echo "File already exists. Checking md5..."
  os=`uname -s`
  if [ "$os" = "Linux" ]; then
    checksum=`md5sum $FILE | awk '{ print $1 }'`
  elif [ "$os" = "Darwin" ]; then
    checksum=`cat $FILE | md5`
  fi
  if [ "$checksum" = "$CHECKSUM" ]; then
    echo "Checksum is correct. No need to download."
    exit 0
  else
    echo "Checksum is incorrect. Need to download again."
  fi
fi

echo "Downloading pretrained saliency models (0.1G)..."

wget $URL -O $FILE

echo "Done. Please run this command again to verify that checksum = $CHECKSUM."
