#!/bin/bash

for name in ./imagenet_val/*.JPEG; do
    convert -resize 227x227\! $name $name
done
