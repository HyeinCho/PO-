# Gaze Tracking & Emotion Recognition
# Startdate: 2020-10-16
# Modified by POSCO AI·Bigdata Academy A3 Team


#package and library
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
# 현재 시간, 비디오 시간(시계열 그래프)를 위함
# for real time, video time(for time series)
from datetime import time
import datetime
# 엑셀 파일 관련
# for excel file
from openpyxl import Workbook
# 동공 인식
# gaze tracking
from gaze_tracking import GazeTracking
# 데이터 시각화 패키지
# Graphic data package
import matplotlib
import matplotlib.pyplot as plt

# GU사용
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

    
# 얼굴인식 및 이모션 관련 모델
# parameters for loading data and images
detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
emotion_model_path = 'models/_mini_XCEPTION.102-0.66.hdf5'
# hyper-parameters for bounding boxes shape
# loading models
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=True)
EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised","neutral"]

# 비디오 파일 사용을 위한 코드
# starting video streaming
camera = cv2.VideoCapture("video_to_audio/Test_3min.mp4")
model = load_model('models/facenet_keras.h5')
#재생할 파일의 넓이 얻기
width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
#재생할 파일의 높이 얻기
height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
#재생할 파일의 프레임 레이트 얻기
fps = camera.get(cv2.CAP_PROP_FPS)

# 영상을 처리하기 위한 코드
# For creating video
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('video_output.avi',fourcc, fps, (int(width), int(height)))

# 엑셀파일 생성
# Create Excel File
wb1 = Workbook()
wb2 = Workbook()
#emotion
ws1 = wb1.active
ws1.title = "emotion"
ws1.append(["Time", "Angry", "Happy", "Neutral"])
#Gaze
ws2 = wb2.active
ws2.title = "gaze"
ws2.append(["Right", "Left", "Bottom", "Top"])

# 코드 실행시 시간
# Time when runs the code
jigeum = datetime.datetime.now()+datetime.timedelta(seconds=20)

# gaze tracking 모델 부르기 및 변수 생성
# Variable Setting
gaze = GazeTracking()
gaze_right = 0
gaze_left = 0
gaze_top = 0
gaze_bottom = 0

# 카메라 실행
# Run camera
while (camera.isOpened()):
    # 카메라 정보 불러오기
    ret, frame = camera.read()
    
    # 프레임이 끝난다면 종료
    if frame is None:
        break;
    
    # 비디오에서 gaze tracking 실행
    # Mark pupil for frame
    gaze.refresh(frame)
    frame = gaze.annotated_frame()
    
    # 프레임을 읽는다면 ret == True, otherwise False
    # if frame -> ret == True, otherwise False
    if ret:
        # 프레임 읽기
        # reading the frame
        
        # 프레임 Gray로 바꾸기
        # convert color(BGR -> gray)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detection.detectMultiScale(gray, scaleFactor=1.1,minNeighbors=5,minSize=\
                                                (50,50),flags=cv2.CASCADE_SCALE_IMAGE)

        # Emotion 나타낼 공간 생성
        # Create canvas for emtion
        canvas = np.zeros((250, 300, 3), dtype="uint8")

        # 얼굴인식 및 Emotion 처리
        # Face detect & emotion recognition
        if len(faces) > 0:
            faces = sorted(faces, reverse=True,
                            key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
            (fX, fY, fW, fH) = faces
                # Extract the ROI of the face from the grayscale image, 
                # resize it to a fixed 28x28 pixels, and then prepare
                # the ROI for classification via the CNN
            roi = gray[fY:fY + fH, fX:fX + fW]
            roi = cv2.resize(roi, (64, 64))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            preds = emotion_classifier.predict(roi)[0]
            emotion_probability = np.max(preds)
            label = EMOTIONS[preds.argmax()]
        else: 
            continue
            
        # 눈동자 위치에 따른 변수 변화
        # Check gaze and change variable
        #print(gaze.vertical_ratio(), gaze.horizontal_ratio())
        # Get point from pupil
        if gaze.is_right():
            gaze_right += 1
        elif gaze.is_left():
            gaze_left += 1
        elif gaze.is_bottom():
            gaze_bottom += 1
        elif gaze.is_top():
            gaze_top += 1

        # 실시간 눈동자 파악 print 문
        # Get log consistently - gaze
        print("<< *RIGHT:" , gaze_right, "*LEFT:", gaze_left, 
              "*BOTTOM:", gaze_bottom, "*TOP:", gaze_top, ">>")
        
        # emotion 확률 구하기 및 엑셀 저장 데이터
        # emotion probability and excel data
        tes = []
        for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
            # construct the label text
            tex = "{}: {:.2f}".format(emotion, prob)
            text = "{}: {:.2f}%".format(emotion, prob * 100)

            te = tex.split(': ')

            tes.append(te[1])

            w = int(prob * 300)
            cv2.rectangle(canvas, (7, (i * 35) + 5),
            (w, (i * 35) + 35), (0, 0, 255), -1)
            cv2.putText(canvas, text, (10, (i * 35) + 23),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45,
            (255, 255, 255), 2)
            cv2.putText(frame, label, (fX, fY - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            cv2.rectangle(frame, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)

        # Emotion 추가
        # Append Emotion data
        ws1.append([(((datetime.datetime.now()-jigeum)/3)-datetime.timedelta(seconds=2)), 
                    tes[0], tes[3], tes[6]])
        

        out.write(frame)
        cv2.imshow('Presentation', frame)
        cv2.imshow("Probabilities", canvas)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Gaze 정보 추가
# Append gaze data
ws2.append([gaze_right, gaze_left, gaze_bottom, gaze_top])

ws1.delete_rows(2)
# Excel 파일 저장
# Save Excel
wb1.save(filename='output_emotion.xlsx')
wb2.save(filename='output_gaze.xlsx')

# Realease Camera(cam) and output(video file)
camera.release()
out.release()
cv2.destroyAllWindows()

# Emotion 그래프 생성
# Create Emotion Graph
df= pd.read_excel("output_emotion.xlsx")

plot = df.plot(x="Time", y=["Angry","Happy","Neutral"], figsize=(15,6))
fig = plot.get_figure()

#Image 저장
#Save Image
fig.savefig("output.png")