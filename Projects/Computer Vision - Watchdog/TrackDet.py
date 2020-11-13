import argparse
import os
import warnings
import datetime
import imutils
import json
import numpy as np
import time
import cv2

# argparse to start script without an ide
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True, help="C:/Users/ldmag/Desktop/CV Python/Computer-Vision-with-Python/test.json")
args = vars(ap.parse_args())

warnings.filterwarnings('ignore')
conf = json.load(open(args["conf"]))

if not conf["use_ip_cam"]:
    camera = cv2.VideoCapture(0)
    time.sleep(0.2)

else:
    camera = cv2.VideoCapture(conf["ip_cam_addr"])

# define configurable variables outside JSON file
time.sleep(conf["camera_warmup_time"])
avg = None
lastUpload = datetime.datetime.now()
motionCounter = 0
fourcc = 0x00000020
writer = None
h,w = (None,None)
zeros = None
output = None
recording = False

# tracker parameters
feature_params = dict(maxCorners = 100, qualityLevel = 0.1, minDistance = 10, blockSize = 10)
track_params = dict(winSize=(150,150), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
color = np.random.randint(0,255,(100,3))
ret, prev_frame = camera.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(prev_gray, mask = None, **feature_params)
mask = np.zeros_like(prev_frame)

# begin contour detection and tracking
while True:
    ret, frame = camera.read()
    ret, frame1 = camera.read()
    gray_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    timestamp = datetime.datetime.now()
    motion_detected = False

    if not ret:
        break
    
    # calculate optical flow
    p1, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, gray_frame, p0, None, **track_params)
    goodPts_new = p1[status==1]
    goodPts_old = p0[status==1]
    
    for i, (new,old) in enumerate(zip(goodPts_new, goodPts_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (a,b), (c,d), color[i].tolist(),2)
        frame1 = cv2.circle(frame, (a,b),5,color[i].tolist(),-1)
    img = cv2.add(frame1,mask)
    prev_gray = gray_frame.copy()
    p0 = goodPts_new.reshape(-1,1,2)

    frame = imutils.resize(frame, width=conf["resize_Width"])
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (21,21), 0)
    if avg is None:
        avg = blur.copy().astype('float')
        continue
    
    cv2.accumulateWeighted(blur, avg, 0.5)
    delta = cv2.absdiff(blur, cv2.convertScaleAbs(avg))
    _,thresh = cv2.threshold(delta, conf["delta_thresh"], 255, cv2.THRESH_BINARY)
    dil = cv2.dilate(thresh, None, iterations=2)
    _,contours,_ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #motion_detected = True 
    for c in contours:
        if cv2.contourArea(c) < conf["min_area"]:
            continue
        (x,y,w1,h1) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x,y), (x+w1, y+h1), (0,255,0),2)
        motion_detected = True
    
    
# end contour detection, begin smart recording
        
    fps = int(round(camera.get(cv2.CAP_PROP_FPS)))
    record_fps = 30
    ts = timestamp.strftime('%Y-%m-%d_%H_%M_%S')
    time_fps = ts + ' - fps: ' + str(fps)
    cv2.putText(frame, "Motion Detected: {}".format(motion_detected), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, time_fps, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    if writer is None:
       filename = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
       file_path = (conf["userDir"] + "/{filename}.mp4")
       file_path = file_path.format(filename=filename)
       h2,w2 = frame.shape[:2]
       writer = cv2.VideoWriter(file_path, fourcc, record_fps, (w2,h2), True)
       zeros = np.zeros((h2,w2), dtype='uint8')

    def record_video():
        output = np.zeros((h2,w2,3), dtype='uint8')
        output[0:h2, 0:w2] = frame
        writer.write(output)

    if motion_detected:
        motionCounter += 1

        if motionCounter >= conf["min_motion_frames"]:
            if conf["create_image"]:
                image_path = (conf["userDir"] + f"/{filename}.jpg").format(filename=filename)
                cv2.imwrite(image_path, frame)
            record_video()

            recording = True
            non_motion_timer = conf["nonMotionTimer"]

    else:
        if recording is True and non_motion_timer > 0:
            non_motion_timer -= 1
            record_video()

        else:
            motionCounter = 0
            if writer is not None:
                writer.release()
                writer = None
            if recording is False:
                os.remove(file_path)
            recording = False
            non_motion_timer = conf["nonMotionTimer"]

    if conf["show_video"]:
        cv2.imshow("Feed", frame)
        cv2.imshow('Tracker view', img)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
    
    
camera.release()
cv2.destroyAllWindows()
