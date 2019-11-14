#!/bin/bash
DIR=$1
#   classifications go to stdout
[ -z "$DIR" ] &&  echo "usage: dir_of_jpg_to_process"  &&  exit 1

FILES="$(ls -1 $DIR/*jpg)"
for f in $FILES ; do
  ./infer_image.py --image="$f" --graph=./retrained_graph.pb --labels=out/retrained_labels.txt \
    --input_layer=Placeholder --output_layer=final_result 
done
echo "process:  $FILES"
