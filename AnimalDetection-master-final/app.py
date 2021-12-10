import os
from flask import Flask, flash, url_for, redirect, render_template, request, jsonify, Response
import requests
from werkzeug.utils import secure_filename
import mobilenet_ssd_python as cv
import threading
import numpy as np
import pandas as pd
import cv2
import pymongo
from datetime import date
import datetime
import time

client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client['alertsystem']
visit = db['alarm']

time = datetime.datetime.now()
current_time = time.strftime("%H:%M:%S")
print(current_time)
today = datetime.date.today()
outputFrame = None
lock = threading.Lock()

app = Flask(__name__)
stop_the_thread = False


UPLOAD_FOLDER = 'upload'
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'AVI', 'mp4', 'mkv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    
    return render_template('home.html')


@app.route('/moni')
def history():
    return render_template("moni.html")
@app.route('/hist',methods = ['GET','POST'])
def show():
    if request.method == 'POST':
        date = request.form['date']
        res = visit.find({"date":str(date)})
        per = []
        for x in res:
            per.append(x)
        return render_template("moni.html",data = per) 

@app.route('/prediction/')
def res():
    minx, maxx = 5, 6
    return render_template('prediction.html', minx=minx, maxx=maxx)


@app.route('/upload', methods=["GET", 'POST'])
def uploadimg():
    if request.method == 'POST':
        
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            frame = cv2.imread(r'upload/' + filename)

            frame, labels = cv.singleDetection(frame)

            # return redirect(url_for('uploaded_file',
                              # filename=filename))

        return jsonify({"result": labels})


@app.route('/uploadvid', methods=["GET", 'POST'])
def uploadvid():
    print("Uploading In progress...")
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            video = r'upload/' + filename
            print('Starting thread 1... ')
            t = threading.Thread(target=detect_object, args=(
		            video,))
            t.daemon = True
            t.start()

            # cap = cv2.VideoCapture(video)

            # frame_width = int(cap.get(3))

            # frame_height = int(cap.get(4))

            # # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
            # # out = cv2.VideoWriter(r'static/out/outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))

            # # Load the tensorflow model
            # net = cv2.dnn.readNetFromTensorflow(
            #     'frozen_inference_graph.pb', 'graph.pbtxt')
            # i = 1
            # while True:
            #     i = i + 1
            #     # Capture frame-by-frame
            #     ret, frame = cap.read()
            #     # frame_resized = cv2.resize(frame,(300,300)) # resize frame for prediction

            #     frame, labels = cv.singleDetection(frame)

            #     # out.write(frame)

            #     print("Frame", i)

            #     if i > 200:
            #         break
            #     # cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
            #     # cv2.imshow("frame", frame)
            #     # if cv2.waitKey(1) >= 0:  # Break with ESC
            #     #     break

    return Response(generate(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

@app.route('/livevid', methods=["GET", 'POST'])
def livevideo():
    if request.method == 'POST':
        print('Starting thread 1... ')
        t2 = threading.Thread(target=detect_object_live)
        t2.daemon = True
        t2.start()

        # cap = cv2.VideoCapture(0)

        # frame_width = int(cap.get(3))

        # frame_height = int(cap.get(4))

        # # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
        # # out = cv2.VideoWriter(r'static/out/outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
        # # Load the tensorflow model
        # net = cv2.dnn.readNetFromTensorflow(
        #     'frozen_inference_graph.pb', 'graph.pbtxt')
        # i = 1
        # while True:
        #     i = i + 1
        #     # Capture frame-by-frame
        #     ret, frame = cap.read()
        #     # frame_resized = cv2.resize(frame,(300,300)) # resize frame for prediction
        #     frame, labels = cv.singleDetection(frame)
        #     # out.write(frame)

        #     print("Frame", i)

        #     if i > 200:
        #         break
        #     # cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
        #     cv2.imshow("frame", frame)
        #     # if cv2.waitKey(1) >= 0:  # Break with ESC
        #     #     break

    return Response(generate(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")
    

def detect_object_live():
    
    
   

    cap = cv2.VideoCapture(0)

    global vs, outputFrame, lock

    # initialize the motion detector and the total number of frames
    # read thus far

    i = 0

    # loop over frames from the video stream
    while True:
        # read the next frame from the video stream, resize it,
        # convert the frame to grayscale, and blur it
        i = i + 1
         # Capture frame-by-frame
        ret, frame = cap.read()

        # frame_resized = cv2.resize(frame,(300,300)) # resize frame for predictio
        frame, labels = cv.singleDetection(frame)

        # out.write(frame)

        
        if i > 200:
            break

        cv.update(frame)
        
        
        # acquire the lock, set the output frame, and release the
        # lock
        with lock:
            outputFrame = frame.copy()
            
            

def detect_object(video):

    cap = cv2.VideoCapture(video)

    global vs, outputFrame, lock

    # initialize the motion detector and the total number of frames
    # read thus far

    i = 0
    count = 0
    # loop over frames from the video stream
    while True:
        # read the next frame from the video stream, resize it,
        # convert the frame to grayscale, and blur it
        i = i + 1
         # Capture frame-by-frame
        ret, frame = cap.read()

        if ret:
            count += 60 # i.e. at 30 fps, this advances one second
            cap.set(1, count)
        else:
            cap.release()
            break
        
        # frame_resized = cv2.resize(frame,(300,300)) # resize frame for predictio
        frame, labels = cv.singleDetection(frame)

        # out.write(frame)

        # print("Still Running", i)

        if i > 200:
            break

        cv.update(frame)

        # acquire the lock, set the output frame, and release the
        # lock
        with lock:
            outputFrame = frame.copy()


def generate():
    # grab global references to the output frame and lock variables
    global outputFrame, lock

    # loop over frames from the output stream
    while True:
          # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if outputFrame is None:
                continue

            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

            # ensure the frame was successfully encoded
            if not flag:
                continue

        # yield the output frame in the byte format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
              bytearray(encodedImage) + b'\r\n')


@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    
    # start the flask app
    app.run()

 # release the video stream pointer
# cap.stop()
