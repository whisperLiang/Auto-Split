This readme contains the details of the changes that were applied on top of the
original Yolov3 repository . The original readme can be found here :- 
[Yolov3 original readme](README2.md) 

# Installation instructions
Follow the installation instructions from the main [README](../../../README.md). 
Unlike, [Yolov3 original readme](README2.md) we use `pytorch==1.3.1`.
````
python==3.7.6 
pytorch==1.3.1
tensorflow==1.15
````
### Setting up the COCO directory 
This repository needs `coco2017` directory in two formats. 
This is because yolov3 original repository requires coco2017 in a different format compared to standard COCO2017. 

In [auto_split/detection_models/yolov3_master/quantize_edge_dnn.py](quantize_edge_dnn.py) arguments :-
*  `--cocodir` points to the coco directory format used in rest of the repository i.e, similar to the format used by
 distiller.
* `--data` points to the [auto_split/detection_models/yolov3_master/data/coco2017.data](data/coco2017.data) file. 
This file contains paths to the yolov3 specific coco directory which is in a different format. 
Note, that the coco directory for yolo repository uses a different format. To download the yolov3 specific 
coco dataset follow [Yolov3 original readme](README2.md) .
 


### Run Script 
Bash script to run all steps of YOLOv3 models can be found here - 
`auto_split/yolov3_run_scripts` directory. For example, to run a yolo-tiny model end-to-end. Run the 
scripts in the following order. It follows the same steps as auto-split algorithm (`auto_split/README.md`) :-
* 1_get_stats_yolo-tiny.sh
* 2_auto_split_yolo-tiny.sh
* 3_latency_yolo-tiny.sh
* 4_quantization_yolo-tiny.sh
* 5_collect_stats_yolo-tiny.sh

The script runs yolov3-tiny model in three different image resolutions :- 416, 512 and 608.
***Make sure to set the correct paths at different stages***. 
***Also, make sure to set complete paths and not relative paths in the scripts***.
Since, the python scripts call bash scripts from within as subprocesses which may break. 

For an explanation of step by step break down follow the steps below. Each step is similar to 
`auto_split/README.md`.

# Step 1: Generate statistics
This step generates 
* Quantization statistics such as layer wise - min,max, mean, and std.
* Sample weight and activation statistics, 
* DNN graph information such as layer ranks, predecessors and successors. 
````
export  ROOTFOLDER=~/edge-cloud-collaboration/auto_split
pushd $ROOTFOLDER
export PYTHON=<path to python>/python
export PYTHONPATH=~/edge-cloud-collaboration/auto_split:~/edge-cloud-collaboration/libs/distiller-master
export CUDA_VISIBLE_DEVICES=0

export yolocfg=~/edge-cloud-collaboration/auto_split/detection_models/yolov3_master/cfg/yolov3-tiny.cfg
export wgtsfile=<path to>/yolov3-tiny.pt

#FOR IMAGE-416
export IMGSIZE=416
export MODELNAME=yolov3-tiny-416

$ CUDA_VISIBLE_DEVICES=0 $PYTHON detection_models/yolov3_master/quantize_edge_dnn.py -b 16 -j 16  \ 
--img-size $IMGSIZE --cfg $yolocfg --weights $wgtsfile  \ 
--evaluate --quantize-eval --qe-lapq --quant-method bit_search

````

# Step 2: Apply DNN graph operator fusion. 
Copy stats to the correct folder. 
````
$ cd latest_log_dir
$ cp -r  * $ROOTFOLDER/data/$MODELNAME/.
$ cd $ROOTFOLDER
````

# Step 3:  Post process post_prepare_model_df.csv
Calculate layer rank, predecessor, and successor information. 
Add the above information along with min, max, std etc information.

* Input: `tools/auto_split/data/$MODELNAME/post_prepare_model_df.csv`
* Output: `tools/generated/hw_simulator/post_process2/$MODELNAME.csv`

````
$ python tools/bit_search/add_rank_min_max_stats.py --arch $MODELNAME
````

# Step 4: Run Auto-split algorithm

The auto_split.py searches for optimal split and generates mse stats 
and bit allocations. 

* Auto_split-v1: Generates random activations and weights based on the layer shapes.
* Auto-split-v2: Reads sample activations and weights (.pkl files) generated from the DNN in 
[Step 2](#step-2-apply-dnn-graph-operator-fusion)

````
# For running Auto-split-v1
$ python tools/bit_search/auto_split.py --arch $MODELNAME
# For running Auto-split-v2 
$ python tools/bit_search/auto_split.py --arch $MODELNAME --logdir data/$MODELNAME
$ export TIMESTAMPFOLDER=yolov3-tiny-416_20200910-033828
````

# Step 5: Generate and collect Latency stats 
````
$ python tools/bit_search/model_latency_2_gen.py -n $TIMESTAMPFOLDER -t 16 
$ python tools/bit_search/model_latency_3_collect.py -n $TIMESTAMPFOLDER
````
# Step 6: Quantization 
### Step 1: Generate yaml config from bitsearch
````
 $ python tools/run_quantization/set_yaml_config_lapq.py -n $TIMESTAMPFOLDER --arch $MODELNAME
````
### Step 2: Run distiller quantization 

To run one configuration of bit-widths 
````
$  python detection_models/yolov3_master/quantize_edge_dnn.py --evaluate --cfg $yolocfg\
--img-size $IMGSIZE \
--weights $wgtsfile \
--evaluate \
--quantize-eval \
--qe-lapq \
--qe-config-file \
data/yolov3-512/yolov3_post_train.yaml
````

To run all auto-split generated configurations
````
$ CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES $PYTHON tools/run_quantization/run_distiller_yolo.py \ 
    --pythonpath $PYTHON --img-size $IMGSIZE --deviceid $CUDA_VISIBLE_DEVICES \ 
    --arch $MODELNAME -n $TIMESTAMPFOLDER --cfg $yolocfg --weights $wgtsfile
````

### Step 3: Collect results 
````
$ python tools/run_quantization/get_accuracy_yolo.py -n $TIMESTAMPFOLDER
````
