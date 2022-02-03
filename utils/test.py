import cv2
import time
import multiprocessing as mp
import ultimateAlprSdk
import sys
import argparse
import json
import platform
import os.path
from PIL import Image, ExifTags
# EXIF orientation TAG
ORIENTATION_TAG = [orient for orient in ExifTags.TAGS.keys() if ExifTags.TAGS[orient] == 'Orientation']

# Defines the default JSON configuration. More information at https://www.doubango.org/SDKs/anpr/docs/Configuration_options.html
JSON_CONFIG = {
    "debug_level": "info",
    "debug_write_input_image_enabled": False,
    "debug_internal_data_path": ".",

    "num_threads": -1,
    "gpgpu_enabled": True,
    "max_latency": -1,

    "klass_vcr_gamma": 1.5,

    "detect_roi": [0, 0, 0, 0],
    "detect_minscore": 0.1,

    "car_noplate_detect_min_score": 0.8,

    "pyramidal_search_enabled": True,
    "pyramidal_search_sensitivity": 0.28,
    "pyramidal_search_minscore": 0.3,
    "pyramidal_search_min_image_size_inpixels": 800,

    "recogn_minscore": 0.3,
    "recogn_score_type": "min"
}

TAG = "[PythonRecognizer] "

parser = argparse.ArgumentParser(description="""
            This is the recognizer sample using python language
            """)

parser.add_argument("--assets", required=False, default="assets", help="Path to the assets folder")
parser.add_argument("--charset", required=False, default="latin",
                         help="Defines the recognition charset (a.k.a alphabet) value (latin, korean, chinese...)")
parser.add_argument("--car_noplate_detect_enabled", required=False, default=False,
                         help="Whether to detect and return cars with no plate")
parser.add_argument("--ienv_enabled", required=False, default=platform.processor() == 'i386',
                         help="Whether to enable Image Enhancement for Night-Vision (IENV). More info about IENV at https://www.doubango.org/SDKs/anpr/docs/Features.html#image-enhancement-for-night-vision-ienv. Default: true for x86-64 and false for ARM.")
parser.add_argument("--openvino_enabled", required=False, default=True,
                         help="Whether to enable OpenVINO. Tensorflow will be used when OpenVINO is disabled")
parser.add_argument("--openvino_device", required=False, default="CPU",
                    help="Defines the OpenVINO device to use (CPU, GPU, FPGA...). More info at https://www.doubango.org/SDKs/anpr/docs/Configuration_options.html#openvino-device")
parser.add_argument("--klass_lpci_enabled", required=False, default=False,
                    help="Whether to enable License Plate Country Identification (LPCI). More info at https://www.doubango.org/SDKs/anpr/docs/Features.html#license-plate-country-identification-lpci")
parser.add_argument("--klass_vcr_enabled", required=False, default=False,
                    help="Whether to enable Vehicle Color Recognition (VCR). More info at https://www.doubango.org/SDKs/anpr/docs/Features.html#vehicle-color-recognition-vcr")
parser.add_argument("--klass_vmmr_enabled", required=False, default=False,
                    help="Whether to enable Vehicle Make Model Recognition (VMMR). More info at https://www.doubango.org/SDKs/anpr/docs/Features.html#vehicle-make-model-recognition-vmmr")
parser.add_argument("--klass_vbsr_enabled", required=False, default=False,
                    help="Whether to enable Vehicle Body Style Recognition (VBSR). More info at https://www.doubango.org/SDKs/anpr/docs/Features.html#vehicle-body-style-recognition-vbsr")
parser.add_argument("--tokenfile", required=False, default="", help="Path to license token file")
parser.add_argument("--tokendata", required=False, default="", help="Base64 license token data")

args = parser.parse_args()

# Update JSON options using values from the command args
JSON_CONFIG["assets_folder"] = args.assets
JSON_CONFIG["charset"] = args.charset
JSON_CONFIG["car_noplate_detect_enabled"] = (args.car_noplate_detect_enabled == "True")
JSON_CONFIG["ienv_enabled"] = (args.ienv_enabled == "True")
JSON_CONFIG["openvino_enabled"] = (args.openvino_enabled == "True")
JSON_CONFIG["openvino_device"] = args.openvino_device
JSON_CONFIG["klass_lpci_enabled"] = (args.klass_lpci_enabled == "True")
JSON_CONFIG["klass_vcr_enabled"] = (args.klass_vcr_enabled == "True")
JSON_CONFIG["klass_vmmr_enabled"] = (args.klass_vmmr_enabled == "True")
JSON_CONFIG["klass_vbsr_enabled"] = (args.klass_vbsr_enabled == "True")
JSON_CONFIG["license_token_file"] = args.tokenfile
JSON_CONFIG["license_token_data"] = args.tokendata


def checkResult_1(operation, result, frame_1, file, flag1, plates_1):

    dict_1 = {}
    if not result.isOK():
        #     # print(TAG + operation + ": failed -> " + result.phrase())
        assert False
    else:
        # print(TAG + operation + ": OK -> " + result.json())

        numbers = str(result.json()).count('text')

        if numbers > 0:
            patches = str(result.json()).split(',')
            # self.file.write(str(result.json()))
            dictionaryA = eval(result.json())
            for key, val in dictionaryA.items():
                if key == 'plates':

                    # self.file.write(str(val[0]))
                    sub_dict = eval(str(val[0]))
                    for k, v in sub_dict.items():
                        # self.file.write(str(k)+'\n')
                        # self.file.write(str(v)+'\n')
                        if k == 'warpedBox':
                            x1 = int(v[0])
                            y1 = int(v[1])
                            x2 = int(v[2])
                            y2 = int(v[-1])

                        elif k == 'text':
                            data_1 = v
                            dict_1[k] = data_1

                    h, w, _ = frame_1.shape
                    cv2.rectangle(frame_1, (x1, y1), (x2, y2), (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.rectangle(frame_1, (x1, y1 - (y2 - y1) // 6), (x2, y1), (255, 255, 255), -1,
                                  cv2.LINE_AA)  # filled
                    cv2.putText(frame_1, data_1, (x1, y1), 0, h / 480, (255, 100, 230), thickness=2,
                                lineType=cv2.LINE_AA)
        if len(dict_1) != 0:
            plates_1 = f''
            for i in range(len(dict_1)):
                plates_1 += f"{data_1}\n"
            # self.flag1 = 5
            if data_1 is not None:
                file.write(plates_1)

            flag1 = 0
        else:
            flag1 += 1

        if flag1 >= 3:
            plates_1 = 'No Plates Detected'
        return frame_1


flag = 0
plates_1 = 'No Plates Detected'
# Initialize the engine
checkResult_1("Init",
                   ultimateAlprSdk.UltAlprSdkEngine_init(json.dumps(JSON_CONFIG)),
                           None,  None, flag,  plates_1)

class camera():

    def __init__(self,rtsp_url):
        #load pipe for data transmittion to the process
        self.parent_conn, child_conn = mp.Pipe()
        #load process
        self.p = mp.Process(target=self.update, args=(child_conn,rtsp_url))
        #start process

        self.p.daemon = True
        # mp.freeze_support()

        self.p.start()

    def end(self):
        #send closure request to process

        self.parent_conn.send(2)

    def update(self,conn,rtsp_url):
        #load cam into seperate process

        print("Cam Loading...")
        cap = cv2.VideoCapture(rtsp_url,cv2.CAP_FFMPEG)
        print("Cam Loaded...")
        run = True

        while run:

            #grab frames from the buffer
            cap.grab()

            #recieve input data
            rec_dat = conn.recv()


            if rec_dat == 1:
                #if frame requested
                ret,frame = cap.read()
                conn.send(frame)

            elif rec_dat ==2:
                #if close requested
                cap.release()
                run = False

        print("Camera Connection Closed")
        conn.close()

    def get_frame(self,resize=None):
        ###used to grab frames from the cam connection process

        ##[resize] param : % of size reduction or increase i.e 0.65 for 35% reduction  or 1.5 for a 50% increase

        #send request
        self.parent_conn.send(1)
        frame = self.parent_conn.recv()

        #reset request
        self.parent_conn.send(0)

        #resize if needed
        if resize == None:
            return frame
        else:
            return self.rescale_frame(frame,resize)

    def rescale_frame(self,frame, percent=65):

        return cv2.resize(frame,None,fx=percent,fy=percent)


if __name__ == "__main__":
    #cam = camera("rtsp://admin:123456@192.168.1.100:554/ch0")
    cam = camera("test2.mp4")

    print(f"Camera is alive?: {cam.p.is_alive()}")
    flag = 0
    plates_1 = 'No Plates Detected'
    frame_count = 0
    with open('data.txt', 'w') as f:
        while(1):
            frame = cam.get_frame(0.65)
            if frame_count % 10 == 0:  # Frame Skipping
                cv2.imwrite('first_url.jpg', frame)
                image = Image.open('first_url.jpg')
                width, height = image.size
                if image.mode == "RGB":
                    format = ultimateAlprSdk.ULTALPR_SDK_IMAGE_TYPE_RGB24
                elif image.mode == "RGBA":
                    format = ultimateAlprSdk.ULTALPR_SDK_IMAGE_TYPE_RGBA32
                elif image.mode == "L":
                    format = ultimateAlprSdk.ULTALPR_SDK_IMAGE_TYPE_Y
                else:
                    print(TAG + "Invalid mode: %s" % image.mode)
                    assert False
                exif = image._getexif()
                # print(exif)
                exifOrientation = exif[ORIENTATION_TAG[0]] if len(ORIENTATION_TAG) == 1 and exif != None else 1

                # Recognize/Process
                # Please note that the first time you call this function all deep learning models will be loaded
                # and initialized which means it will be slow. In your application you've to initialize the engine
                # once and do all the recognitions you need then, deinitialize it.
                frame_1 = checkResult_1("Process",
                                        ultimateAlprSdk.UltAlprSdkEngine_process(
                                            format,
                                            image.tobytes(),  # type(x) == bytes
                                            width,
                                            height,
                                            0,  # stride
                                            exifOrientation),
                                        frame, f, flag, plates_1
                                        )
                frame_1 = cv2.resize(frame_1, (800, 700))
                cv2.imshow('Processed Stream', frame_1)
                mp.freeze_support()
            else:
               # frame_1 = cv2.resize(frame, (800, 700))
               cv2.imshow("Processed Stream",frame_1)
            frame_count += 1
            if frame_count == 100:
                frame_count = 1
            key = cv2.waitKey(1)

            if key == 13: #13 is the Enter Key
                break

    cv2.destroyAllWindows()

    cam.end()