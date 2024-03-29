# Auto-split Face detection demo
This project takes a Yolo-v3 based face detection model and runs a demo to view 
a) Auto-split solution and b) cloud-only solution side-by-side. 
This is a proof of concept for auto-split.
 
# Run scripts -- Quick without detailed explanation
To run the scripts directly follow this section. 
For a more detailed explanation, follow next section onwards. 
Make sure to comment out ``cv2.imshow('frame', img_to_draw)`` in 
the function `run_many_video_server` from the following files ``cloud_split_inference.py`` 
and ``cloud_only_inference.py``. This will make sure that the videos are not displayed live on the screen. 

**Issues with displaying live:**
* Using MobaXterm has high overhead of displaying two videos live over ssh connection and becomes the bottleneck. 
* Hence, we store the frame result on the local file and post process the files to display them side-by-side for the
 demo.
* If not using ssh connection, can uncomment ``cv2.imshow(`` in the code 
and both videos (split and cloud-only) will be displayed simultaneously side by side.


Next run the following scripts. 

The first script runs the demo on the same machine. 
It uses two different gpus for the cloud machines. Onee for cloud-only solution and one for auto-split solution. 
For the edge device, two differnt cpu threads are called. 

The second script loads the videos with bounding boxes stored on the local files. 
Stitches each frame side-by-side for viewing. 
 
 **1. Download the models** in 
 [Code/models.tar.gz](https://drive.google.com/drive/folders/1DX8tS1KeFA2QfPdlzHvaf1JCgnfDOFCN), untar, 
 and copy them to `edge_dnn/` and `cloud_dnn/` folders. The directory structure should look like below :-
 
 ````
 $ mkdir -p edge_dnn/models
 $ mkdir -p cloud_dnn/models
 $ ls models/darknet53_288x512_person&face/* 

models/darknet53_288x512_person&face/checkpoint
models/darknet53_288x512_person&face/yolo3_darknet53.pb
models/darknet53_288x512_person&face/yolo3-epoch000-val_loss19.37.ckpt-15000.data-00000-of-00001
models/darknet53_288x512_person&face/yolo3-epoch000-val_loss19.37.ckpt-15000.index

models/darknet53_288x512_person&face/frozen_pb:
yolo3_edge_darknet53.pb  yolo3_edge_darknet53.pbtxt
````
**2. Run the demo where edge and cloud both are same device.**
````
$ bash run_demo_same_device.sh
$ bash post_process.sh 
````
**3. The output demo video** is stored in ``/face_detection_demo/cloud_dnn/results/``

------

# Detailed Explanation
# Environment Set up
* To run code on the cloud device, go to `$ cd split/cloud_dnn/`. Run it on the cloud device. 
* To run code on the edge device, go to `$ cd split/edge_dnn`. Run the code on the edge device.

**"IGNORE `yolo3_darknet53/` folder"**

# Image example 
## 1. Split Solution

#### Step 1: cloud side 
The cloud DNN code `split/cloud_dnn` requires the following additional data :-
* `test_images/` containing `.jpg` images [Only used to verify results]
* `models/darknet53_288x512_person&face/` containing the original model checkpoint

The cloud dnn runs an XML RPCServer at port:8000 (``see run_cloud.py``). 
**Start the server** by running the following conmmand :- 

````
$ cd split/cloud_dnn
$ python run_cloud.py --split-inference test_images/
# You should see 
# Listening on port 8000...
````

The cloud DNN takes the following inputs generated by the edge dnn :- 
* quantized activation, 
* scale, 
* zero_point 
* and image_shape 
via RPC calls and generates `all_classes`, `all_scores`, 
and `all_bboxes`. 
The results are used to overlay on top of the original image from the `test_images/`
folder and the results are written to the `results/` folder on the server.


#### Step 2: Edge side 
The edge DNN code `split/edge_dnn` executes on the edge device. 
It requires the following additional data :-
* `test_images/` containing `.jpg` images 
* `models/darknet53_288x512_person&face/` containing the original model checkpoint

It takes `.jpg` images as input and generates the following output 
:- a) quantized activation, b) scale, c) zero_point and d) image_shape in `../output/` folder.

The output is transferred over the RPC client to the cloud DNN server. 
(see ``edge_dnn/split_inference.py: with xmlrpc.client.ServerProxy``)

To run the split-inference for auto-split on the edge device 

````

$ cd edge_dnn
# Make sure the cloud DNN server is running already and listening on port:8000
$ python run_edge.py --split-inference test_images/ -q --num-bits 4 --act-compress
````

* `-q` flags tells whether to quantize the transmitted activations or not.
If `-q` flag is given, then we also need to provide `--num-bits` value 
to tell the quantization bit-width for the transmitted activation. 
* The `act-compress` flag controls whether the activations are compressed before transmission or not. 
When this flag is provided, the activations are compressed on the edge device, and then 
decompressed on the cloud side.  



###### To save Frozen ".pb" files for split model 
This saves the parameters alogn with the DNN graph as frozen .pb files for inference.
The following command saves both Edge and Cloud DNN. 
````
$ python run_edge.py --split-save-frozen-pb
````



## 2. Cloud-only 

Step 1: On the cloud side :- 
````
$ cd split/cloud_dnn
$ python run_cloud.py --cloud-only test_images/

# You should see 
# Listening on port 8000...
````

Step 2: On the edge side :- 
````
$ cd split/edge_dnn
$ python run_edge.py --split-inference test_images/ --cloud-only
````



# Video Example 

## 1. Split Solution
Step 1: On the cloud side .. 

````
$ PYTHON=<path to python>/bin/python
$ video_dir=<path to video dir>
$ video=face1.mp4

$ cd split/cloud_dnn
$ $PYTHON run_cloud.py --split-inference-video $video --gpu 0 --port 8000 &
# cd .. 

# You should see 
# Listening on port 8000...
````

Step 2: On the Edge side 

In video example, the edge DNN runs the video and transmits 
* quantized activation `x-bit`, 
* scale, 
* zero_point 
* and image_shape 

To run the video example :- 
````
$ video=<path to video dir>/face1.mp4
$ cd edge_dnn
# Make sure the cloud DNN server is running already and listening on port:8000
$ $PYTHON run_edge.py --split-inference-video $video -q --num-bits 4 --act-compress --ipaddr localhost --port 8000 
$ cd ..
````
To not use compression before transmission, ignore `--act-compress` flag.

## 2. Cloud-only 

Step 1: On the cloud side :- 
````
$ cd split/cloud_dnn
$ python run_cloud.py --cloud-only-video  test_images/face1.mp4 --gpu 1 --port 6000

# You should see 
# Listening on port 6000...
````


Step 2: On the edge side :- 
````
$ cd split/edge_dnn
$ python run_edge.py --split-inference-video test_images/face1.mp4 --cloud-only --ipaddr http://localhost:6000/
````


## 3. Everything on cloud
Everything from reading image to yolov3 object detection on cloud. 
````
$ cd split/cloud_dnn
$ python run_cloud.py --all-cloud-video test_images/face1.mp4
````

