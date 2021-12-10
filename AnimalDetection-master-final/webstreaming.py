# USAGE
# python webstreaming.py --ip 0.0.0.0 --port 8000

# import the necessary packages
# from pyimagesearch.motion_detection import SingleMotionDetector
import mobilenet_ssd_python as cv
from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
import threading
import argparse
import datetime
import imutils
import time
import cv2

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful for multiple browsers/tabs
# are viewing tthe stream)
outputFrame = None
lock = threading.Lock()

# initialize a flask object
app = Flask(__name__)

# initialize the video stream and allow the camera sensor to
# warmup
vid = r'upload/VID-20200308-WA0014.mp4'

vs = cv2.VideoCapture(vid)
# time.sleep(2.0)

@app.route("/")
def index():
	# return the rendered template
	return render_template("index.html")

def detect_motion():
	# grab global references to the video stream, output frame, and
	# lock variables
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
			ret, frame = vs.read()

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
		mimetype = "multipart/x-mixed-replace; boundary=frame")

# check to see if this is the main thread of execution
if __name__ == '__main__':

	t = threading.Thread(target=detect_motion)
	t.daemon = True
	t.start()

	# start the flask app
	app.run(host='0.0.0.0', debug=True,
		threaded=True, use_reloader=False)

# release the video stream pointer
vs.stop()