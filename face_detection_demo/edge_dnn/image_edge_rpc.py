from edge_split_inference import YoloSplitInference
from jpeg_compression import YoloSplitInference as YoloJPEG

def split_save_frozen_pb():
    yolo_infer = YoloSplitInference(args.ipaddr)
    yolo_infer.save_frozen_pb()

def split_inference(args):
    test_dir_name = args.split_inference
    is_quantized = args.is_quantized
    is_cloud_only = args.cloud_only
    act_compress = args.act_compress
    num_bits = args.num_bits
    yolo_infer = YoloSplitInference(args.ipaddr)
    yolo_infer.run_many(test_dir_name, is_quantized, num_bits, is_cloud_only, act_compress)

def test_jpeg_compression(args):
    test_dir_name = args.jpeg
    is_quantized = args.is_quantized
    is_cloud_only = args.cloud_only

    num_bits = args.num_bits
    yolo_infer = YoloJPEG()
    yolo_infer.run_many(test_dir_name, is_quantized, num_bits, is_cloud_only)


