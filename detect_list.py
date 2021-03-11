import time
import cv2
import numpy as np
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images, load_tfrecord_dataset
from yolov3_tf2.utils import draw_outputs
import os
import argparse

def Add_Arguments(parser):
    parser.add_argument("--classes", default = "data/labels/coco.names", type = str, help = "path to classes file")
    parser.add_argument("--weights", default = "weights/yolov3.tf", type = str, help = "path to weights file")
    parser.add_argument("--tiny", default = False, type = bool, help = "tiny or not")
    parser.add_argument("--size", default = 416, type = int, help = "resize images to")
    parser.add_argument("--images", default = "data/images/dog.jpg", type = str, help = "path to input image")
    parser.add_argument("--img_list", default = "data/images/test_data.txt", type = str, help = "path to list of input images")
    parser.add_argument("--output", default = "detections/", type = str, help = "path to output folder")
    parser.add_argument("--num_classes", default = 80, type = int, help = "number of classes in the model")
    parser.add_argument("--tfrecord", default = None, type = str, help = "tfrecord instead of images")
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
    yolo.load_weights(args.weights).expect_partial()
    print("weights loaded")

    class_names = [c.strip() for c in open(args.classes).readlines()]
    print("classes loaded")

    if args.tfrecord:
        dataset = load_tfrecord_dataset(args.tfrecord, args.classes, args.size)
        dataset = dataset.shuffle(512)
        img_raw, _label = next(iter(dataset.take(1)))
    else:
        raw_images = []
        with open(args.img_list,'r') as file:
            images = [x.rstrip('\n') for x in file]
        for image in images:
            img_raw = tf.image.decode_image(
                open(image, 'rb').read(), channels=3)
            raw_images.append(img_raw)

    num = 0    
    for raw_img in raw_images:
        num+=1
        img = tf.expand_dims(raw_img, 0)
        img = transform_images(img, args.size)

        t1 = time.time()
        boxes, scores, classes, nums = yolo(img)
        t2 = time.time()
        print('time: {}'.format(t2 - t1))

        print('detections:')
        for i in range(nums[0]):
            print('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                            np.array(scores[0][i]),
                                            np.array(boxes[0][i])))

        img = cv2.cvtColor(raw_img.numpy(), cv2.COLOR_RGB2BGR)
        img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
        cv2.imwrite(args.output + 'detection' + str(num) + '.jpg', img)
        print('output saved to: {}'.format(args.output + 'detection' + str(num) + '.jpg'))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = Add_Arguments(parser)
    detector(args)
