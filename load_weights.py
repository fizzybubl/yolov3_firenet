import numpy as np
from yolov3_tf2.models import YoloV3, YoloV3Tiny
from yolov3_tf2.utils import load_darknet_weights
import argparse

def Add_Arguments(parser):
    parser.add_argument("--weights",default = "weights/yolov3.weights", type = str, help = "path to weights file")
    parser.add_argument("--output", default = "weights/yolov3.tf", type = str, help = "path to output")
    parser.add_argument("--tiny", default = False, type = bool, help = "yolov3 or yolov3-tiny")
    parser.add_argument("--num_classes", default = 80, type = int, help = "num of classes in the model")
    args = parser.parse_args()
    return args

def load(args):
    if args.tiny:
        yolo = YoloV3Tiny(classes=args.num_classes)
    else:
        yolo = YoloV3(classes=args.num_classes)
    yolo.summary()
    print("model created")

    load_darknet_weights(yolo, args.weights, args.tiny)
    print("weights loaded")

    img = np.random.random((1, 320, 320, 3)).astype(np.float32)
    output = yolo(img)
    print("sanity check passed")

    yolo.save_weights(args.output)
    print("weights saved")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = Add_Arguments(parser)
    load(args)
    
