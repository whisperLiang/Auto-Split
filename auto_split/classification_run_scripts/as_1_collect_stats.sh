source as_0_set_path.sh
pushd ${CURRENTDIR}/../ > /dev/null

# Iterate the string array using for loop
for val in ${APPS[@]}; do
   echo $val
   export MODELNAME=$val
   CUDA_VISIBLE_DEVICES=1 $PYTHON tools/bit_search/compression/compress_classifier.py --pretrained --arch $MODELNAME  --data $DATASET -j 4 -b 256 --evaluate --quantize-eval --qe-lapq --qe-config-file $GETSTATSYAML --effective-test-size 0.1 --lapq-eval-size 0.1
done

popd > /dev/null




