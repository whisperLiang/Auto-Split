# FasterRCNN - edge cloud collaboration


# Step 1: Generate Layer dependencies for FasterRCNN backbone.

`edge_model = resnet.resnet_edge_v1()` consists of resnet50 + FPN model as edge DNN.
`edge_model = resnet.resnet_edge_v0()` consists of Resnet50 as edge DNN and FPN is part of the cloud DNN. 
Check `get_fpn_layer_rank` flag to select between `resnet_edge_v0()` and `resnet_edge_v1()`
in `edge_cloud_detection.py, __main__()`.
To get layer dependencies for Resnet + FPN i.e., intermediate features, use `resnet_edge_v1()`.
It adds the dependencies similar to the FPN network. 
FPN requires intermediate features from the output of `layer1,layer2,layer3 and layer4`. 

````
$ python  obj_detection/models/edge_cloud_detection.py --evaluate \
--quantize-eval --quant-method bit_search --lapq-search-clipping
````
This generates  `layer_rank.csv`. Currently, it is not accurate so we modified
it to represent FPN network dependencies. It can be found in 
`$ROOTFOLDER/data/resnet_fpn_graph`. Please note, since, the purpose of `layer_rank.csv`
is to only get the transmission dependencies into account for auto split algorithm. 
The end-to-end evaluation in edge_cloud_detection.py does not work 
with `resnet.resnet_edge_v1()`. Therefore, once the `auto_split` algorithm 
suggests a split point and bit-width assignment. Use `edge_model = resnet.resnet_edge_v0()` 
from `Step2: Distributed inference with Objection Detection` onwards, 
since most likely the split point does not exist in FPN network, 
i.e., FPN network is not part of the edge DNN. 
If it does, then it is not supported currently. 

#### Follow the steps from autosplit image classification tutorial.

````
$ export MODELNAME=resnet_fpn_graph
$ python tools/bit_search/add_rank_min_max_stats.py --arch $MODELNAME
$ python tools/bit_search/auto_split.py --arch $MODELNAME
````
Look for the generated folder in `$ROOTFOLDER/generated/bit_search`
````
$ export TIMESTAMPFOLDER=resnet_fpn_graph_20200717-103343
$ python tools/bit_search/model_latency_2_gen.py -n $TIMESTAMPFOLDER -t 16
$ python tools/bit_search/model_latency_3_collect.py -n $TIMESTAMPFOLDER 
````
At the end of the above steps, you should have the bit allocation for suggested split points 
and the estimated end-to-end latencies. 
Output: `generated/latency/$TIMESTAMPFOLDER/(latency_mse_stats.csv & latency_summary.csv`


# Step 2: Distributed inference with Faster RCNN 
This readme explains how to quantize faster-rcnn with resnet-50 backbone
based on the split suggested by the auto-split engine. 

For object detection we use Faster RCNN with resnet 50 backbone 
and use the split point index =12 suggested by the auto split engine
to split the object detection pipeline into edge DNN and cloud DNN.

The edge dnn contains layers 0 -- 12, and is quantized with 
mixed precision with bitwidths per layer suggested by the 
auto split engine. The selected split layer for demonstration 
is `layer2.3`.

The cloud DNN consists of layers 12 -- 50, and 
the rest of the pipeline of faster rcnn which executes in floating point precision.


#### Set Root directory
````
$ export PYTHONPATH=~/edge-cloud-collaboration/auto_split:~/edge-cloud-collaboration/libs/distiller-master
````

#### Test edge DNN quantization set up 
````
python obj_detection\pq_detection\ptq_compress.py --evaluate \
--quantize-eval \
--quant-method  'pq' \ 
--qe-config-file \
./obj_detection/data/edge_dnn/resnet2/pq_yaml/test.yaml
````
To enter debug mode, in `if __name__ == '__main__':` function set `debug=True`. 
To get float model results, i.e., no quantization. Set `is_float_model=True`.


| Quantization method     | Arguments                                                                                                                        |
|-------------------------|----------------------------------------------------------------------------------------------------------------------------------|
| Post Train Quantization | --evaluate --quantize-eval --quant-method 'pq' --qe-config-file  'test.yaml'                                                     |
| LAPQ, Fake Quantization | --evaluate --quantize-eval --quant-method 'fake_pq' --qe-lapq --lapq-maxiter 1 --qe-config-file test.yaml --lapq-search-clipping |
| Evaluate float          | --evaluate --quantize-eval --quant-method 'eval'                                                                                 |
| Bit search - dump stats | --evaluate --quantize-eval --quant-method 'bit_search'                                                                           |



#### Execute edge-cloud split Faster RCNN
````
python obj_detection\models\edge_cloud_detection.py
````

### Auxiliary Scripts 


#### Evaluation of Faster RCNN on COCO 
Execute Faster RCNN evaluation on coco dataset 
````
python obj_detection\pq_detection\compress_detector.py --pretrained --evaluate
````

#### Split DNN Utils
Testing utility functions on resnet50 Image classification model 

````
python auto_split\obj_detection\models\split_dnn_utils.py
````

#### Test edge cloud set up 
````
python obj_detection\models\test_edge_cloud_setup.py
````

#### weight quantization only 

````
$ python obj_detection\models\edge_cloud_detection.py --evaluate \
--quantize-eval \
--quant-method \
fake_pq \
--lapq-search-clipping \
--qe-config-file \
obj_detection/data/edge_dnn/resnet2/lapq_yaml/wgt_quant_only.yaml
````

#### at split FPN1: 8 bit wgt & act solution
````
$ python obj_detection\models\edge_cloud_detection.py \
--evaluate --quantize-eval --quant-method fake_pq \
--lapq-search-clipping \
--qe-config-file obj_detection/data/edge_dnn/resnet2/lapq_yaml/8_12_8_FPN1.yaml
````