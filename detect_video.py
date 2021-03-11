#from absl import logging, flags, app
#from absl.flags import FLAGS
import time
import cv2
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs
import argparse


def Add_Arguments(parser):
    parser.add_argument("--classes", default = "data/labels/coco.names", type = str, help = "path to classes file")
    parser.add_argument("--weights", default = "weights/yolov3.tf", type = str, help = "path to weights file")
    parser.add_argument("--tiny", default = False, type = bool, help = "tiny or not")
    parser.add_argument("--size", default = 416, type = int, help = "resize images to")
    parser.add_argument("--video", default = "data/video/paris.mp4", type = str, help = "path to input image")
    parser.add_argument("--output", default = None, type = str, help = "path to output folder")
    parser.add_argument("--num_classes", default = 80, type = int, help = "number of classes in the model")
    parser.add_argument("--output_format", default = 'XVID', type = str, help = "codec used in VideoWriter when saving video to file")
    args = parser.parse_args()
    return args

def detector(args):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if args.tiny:
        yolo = YoloV3Tiny(classes=args.num_classes)
    else:
        yolo = YoloV3(classes=args.num_classes)

    yolo.load_weights(args.weights)
    print('weights loaded')

    class_names = [c.strip() for c in open(args.classes).readlines()]
    print('classes loaded')

    times = []
    
    try:
        vid = cv2.VideoCapture(int(args.video))
    except:
        vid = cv2.VideoCapture(args.video)

    out = None
    if args.output:
        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*args.output_format)
        out = cv2.VideoWriter(args.output, codec, fps, (width, height))
    fps = 0.0
    count = 0

    while True:
        _, img = vid.read()

        if img is None:
            print("Empty Frame")
            time.sleep(0.1)
            count+=1
            if count < 3:
                continue
            else: 
                break
        img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        img_in = tf.expand_dims(img_in, 0)
        img_in = transform_images(img_in, args.size)

        t1 = time.time()
        boxes, scores, classes, nums = yolo.predict(img_in)
        fps  = ( fps + (1./(time.time()-t1)) ) / 2

        img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
        img = cv2.putText(img, "FPS: {:.2f}".format(fps), (0, 30),
                          cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 2)
        
        if args.output:
            out.write(img)
        cv2.imshow('output', img)
        if cv2.waitKey(1) == ord('q'):
            break
        
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = Add_Arguments(parser)
    detector(args)

