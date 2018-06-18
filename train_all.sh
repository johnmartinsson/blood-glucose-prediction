#!/bin/bash
FILES=$1/*
for f in $FILES
do
  echo "Processing $f file..."
  python run.py -f $f -m train
done
