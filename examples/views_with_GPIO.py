from django.shortcuts import render
from django.http import JsonResponse, StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

import datetime
import time
# Create your views here.
import django_eel as eel
eel.init('examples/templates/examples')
eel.expose()
# weights = '/media/tayyab/FA12B25A12B21C174/Working Directory/pig_corridor_deliverable/bad_piggies.pt'
weights = 'yolov5s.pt'
# Initialize
set_logging()
device = select_device('cpu')
half = device.type != 'cpu'  # half precision only supported on CUDA

# Load model
model = attempt_load(weights, map_location=device)  # load FP32 model

# Please check now
import RPi.GPIO as GPIO

GPIO.setmode(GPIO.BOARD)
GPIO.setup(44, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(43, GPIO.OUT, initial=GPIO.HIGH)
GPIO.setup(45, GPIO.OUT, initial=GPIO.LOW)

def detect(source):

    GPIO.output(43, GPIO.HIGH)
    time.sleep(0.2)
    currentH = 0
    thr_h = 0
    height_diff = 0
    conf = 0
    loop2 = 0
    k = 0
    u = 0

    save_img = False
    agnostic_nms = False
    classes = None
    conf_thres = 0.3
    img_size = 640
    iou_thres = 0.45
    # output = os.path.abspath(os.getcwd()) + '/inference/output'
    save_txt = False
    view_img = False
    view_img, save_txt, imgsz = view_img, save_txt, img_size
    # view_img, save_txt, imgsz = opt.view_img, opt.save_txt, opt.img_size
    # webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))

    # Directories
    # save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    # (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir


    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    hei = []
    start=time.time()
    print(start)

    for path, img, im0s, vid_cap, tup in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = model(img, augment=False)[0]

        # Apply NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)

        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)


            s += '%gx%g ' % img.shape[2:]  # print string

            if len(det):

                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                for *xyxy, conf, cls in reversed(det):

                    label = f'{names[int(cls)]} {conf:.2f}'
                    lbl = label.split(' ')
                    if lbl[0] == 'person':
                        currentH, thr_h, height_diff, label = plot_one_box(hei, xyxy, im0, label=label,
                                                                           color=colors[int(cls)], line_thickness=3)

            if currentH > thr_h and float(conf) >= 0.5 and height_diff > 10:
                GPIO.output(44, GPIO.HIGH)
                GPIO.output(45, GPIO.HIGH)

                print('**************')
                print('      alarm')
                print('**************')
                im0 = cv2.putText(im0, 'ALARM!!!', (50, 150), cv2.FONT_HERSHEY_SIMPLEX,
                                  1, (255, 255, 255), 2, lineType=cv2.LINE_AA)
                time.sleep(2)

            elif (currentH > 0.3 * thr_h and currentH <= thr_h and float(conf) >= 0.5) or (
                    currentH > 0.3 * thr_h and height_diff <= 10 and float(
                    conf) >= 0.35):  # how can I seethe results of this script?? is there any way??? can you show me alarm pins on these conditions.. i wanna see how it is running now
                GPIO.output(44, GPIO.LOW)
                GPIO.output(45, GPIO.HIGH)
                time.sleep(0.01)
                print('**************')
                print('      Warning!')
                print('**************')
                loop2 = u
                im0 = cv2.putText(im0, 'Warning!!!', (50, 150), cv2.FONT_HERSHEY_SIMPLEX,
                                  1, (255, 255, 255), 2, lineType=cv2.LINE_AA)


            elif u - loop2 < 30 and u > 30:
                GPIO.output(45, GPIO.HIGH)
                time.sleep(0.01)
                print('WARNING!!!')
                im0 = cv2.putText(im0, 'Warning!!!', (50, 150), cv2.FONT_HERSHEY_SIMPLEX,
                                  1, (255, 255, 255), 3, lineType=cv2.LINE_AA)

            else:
                GPIO.output(44, GPIO.LOW)
                GPIO.output(45, GPIO.LOW)

            k += 1
            u += 1

            conf = 0
            # cv2.imshow('asd', im0)
            # cv2.waitKey(1)
            # ori_image = cv2.resize(im0, (400, 600))
            #
            # imgencode = cv2.imencode('.jpg', ori_image)[1]
            #
            # stringData = imgencode.tostring()
            # yield (b'--frame\r\n'b'Content-Type: text/plain\r\n\r\n' + stringData + b'\r\n')

    GPIO.output(45, GPIO.LOW)
    GPIO.output(44, GPIO.LOW)
    GPIO.output(43, GPIO.LOW)
    time.sleep(0.2)

@csrf_exempt
def index(request):
    # return render(request, "index.html")
    # source = '/home/tayyab/Downloads/video165935db-5180-4a20-a22a-8949e14b9c90video.mp4'
    source = '0'
    return StreamingHttpResponse(detect(source),
                                 content_type="multipart/x-mixed-replace;boundary=frame")

@csrf_exempt
def save_data(request):
    if request.method == 'POST':
        number1 = request.POST.get("one")
        number2 = request.POST.get("two")
        number3 = request.POST.get("three")
        number4 = request.POST.get("four")
        if number1 == '1' and number2 == '2' and number3 == '3' and number4 == '4' :
            return JsonResponse('success', safe=False)
        else:
            return JsonResponse('error',safe=False)

def home(request):
    return render(request,'examples/hello.html')


# eel.start('examples/hello', mode='chrome', cmdline_args=['--kiosk'])
eel.start('examples/hello')