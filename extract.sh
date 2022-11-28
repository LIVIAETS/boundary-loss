#!/usr/bin/env bash


echo Extracting $1

name=$(basename $1)
tar_=${name%.*}
stem=${tar_%.*}

mkdir -p results/$stem
tar xf $1 -C results/$stem --strip-components=2


# TRN="`ls -d results/constrained_cnn-200508-93fce3e-spartacus-isles/*/ | tr '\n' ' '`"
