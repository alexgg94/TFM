from keras.models import model_from_json
import tensorflow as tf
import json
import cv2
import numpy as np
import os
import sys
import uuid
from imutils.object_detection import non_max_suppression
import time
import pytesseract
from google.cloud import vision
from PIL import Image, ImageEnhance, ImageFilter
from enum import Enum

import threading

frame = None

def decode_predictions(scores, geometry):
    # grab the number of rows and columns from the scores volume, then
    # initialize our set of bounding box rectangles and corresponding
    # confidence scores
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the
        # geometrical data used to derive potential bounding box
        # coordinates that surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability,
            # ignore it
            if scoresData[x] < min_confidence:
                continue

            # compute the offset factor as our resulting feature
            # maps will be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            # extract the rotation angle for the prediction and
            # then compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # use the geometry volume to derive the width and height
            # of the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            # compute both the starting and ending (x, y)-coordinates
            # for the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            # add the bounding box coordinates and probability score
            # to our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    # return a tuple of the bounding boxes and associated confidences
    return (rects, confidences)

def extract_text_gcv(image):
    client = vision.ImageAnnotatorClient()

    cv2.imwrite("tmp.jpg", image)

    with open("tmp.jpg", 'rb') as image_file:
        content = image_file.read()

    image = vision.types.Image(content=content)
    response = client.text_detection(image=image)

    os.remove("tmp.jpg")

    if (len(response.text_annotations) == 0):
        return None

    return ('_'.join([d.description for d in response.text_annotations[1:]]))
    #return response.text_annotations[0].description

class Analysis_Thread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        while (not(frame is None)):
            self.Predict_frame(frame)

    def ResizeImage(self, image):
        return np.array(Image.fromarray(image, 'RGB').resize((50, 50)))

    def Get_frame_name(self, label):
        if label == 0:
            return "cow"
        elif label == 1:
            return "not cow"

    def Predict_frame(self, frame):
        with graph.as_default():
            image = frame.copy()
            score = model.predict(np.array([self.ResizeImage(image)/255]))
            label_index = np.argmax(score)
            accuracy = np.max(score)
            if label_index == 0:
                cv2.imwrite("data/video/frames/" + str(count) + ".jpg", image)

if __name__ == "__main__":

    if(len(sys.argv) != 2):
        print("Usage -> python Predicting.py path_to_video")
        sys.exit()

    global model
    global graph
    global net
    global min_confidence
    global default_height
    global default_width
    global default_padding

    with open("model.json", "r") as json_file:
        model = model_from_json(json_file.read())
        model.load_weights("model.h5")
        graph = tf.get_default_graph()
        model.summary()

    print("[INFO] Playing video")
    count = 0

    vidcap = cv2.VideoCapture(sys.argv[1])

    success, frame = vidcap.read()

    keras_thread = Analysis_Thread()
    keras_thread.start()

    while success:
        print("[INFO] Processing frame " + str(count))
        time.sleep(1)
        vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*1000))
        success, frame = vidcap.read()
        count = count + 1

    # load the pre-trained EAST text detector
    print("[INFO] Loading EAST text detector...")
    net = cv2.dnn.readNet("frozen_east_text_detection.pb")
    min_confidence = 0.75
    default_height = 320
    default_width = 320
    default_padding = 0.15

    # define the two output layer names for the EAST detector model that
    # we are interested -- the first is the output probabilities and the
    # second can be used to derive the bounding box coordinates of text
    layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]

    for image_name in os.listdir("data/video/frames"):
        try:
            print("[INFO] Processing " + image_name)
            image = cv2.imread("data/video/frames/" + image_name)
            orig = image.copy()
            (origH, origW) = image.shape[:2]

            # set the new width and height and then determine the ratio in change
            # for both the width and height
            (newW, newH) = (default_width, default_height)
            rW = origW / float(newW)
            rH = origH / float(newH)

            # resize the image and grab the new image dimensions
            image = cv2.resize(image, (newW, newH))
            (H, W) = image.shape[:2]

            # construct a blob from the image and then perform a forward pass of
            # the model to obtain the two output layer sets
            blob = cv2.dnn.blobFromImage(image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)
            net.setInput(blob)
            (scores, geometry) = net.forward(layerNames)

            # decode the predictions, then  apply non-maxima suppression to
            # suppress weak, overlapping bounding boxes
            (rects, confidences) = decode_predictions(scores, geometry)
            boxes = non_max_suppression(np.array(rects), probs=confidences)

            # initialize the list of results
            results = []

            if len(boxes) == 0:
                print("[INFO] No TAG found...")

            else:
                print("[INFO] Processing text boxes...")
                # loop over the bounding boxes
                for (startX, startY, endX, endY) in boxes:
                    # scale the bounding box coordinates based on the respective
                    # ratios
                    startX = int(startX * rW)
                    startY = int(startY * rH)
                    endX = int(endX * rW)
                    endY = int(endY * rH)

                    # in order to obtain a better OCR of the text we can potentially
                    # apply a bit of padding surrounding the bounding box -- here we
                    # are computing the deltas in both the x and y directions
                    dX = int((endX - startX) * default_padding)
                    dY = int((endY - startY) * default_padding)

                    # apply padding to each side of the bounding box, respectively
                    startX = max(0, startX - dX)
                    startY = max(0, startY - dY)
                    endX = min(origW, endX + (dX * 2))
                    endY = min(origH, endY + (dY * 2))

                    # extract the actual padded ROI
                    roi = orig[startY:endY, startX:endX]

                    # in order to apply Tesseract v4 to OCR text we must supply
                    # (1) a language, (2) an OEM flag of 4, indicating that the we
                    # wish to use the LSTM neural net model for OCR, and finally
                    # (3) an OEM value, in this case, 7 which implies that we are
                    # treating the ROI as a single line of text

                    text = extract_text_gcv(roi)

                    # add the bounding box coordinates and OCR'd text to the list
                    # of results
                    if(text != None):
                        results.append(((startX, startY, endX, endY), text))
                    else:
                        print("[INFO] Could not extract text from region")
                # sort the results bounding box coordinates from top to bottom
                results = sorted(results, key=lambda r:r[0][1])

                # loop over the results
                for ((startX, startY, endX, endY), text) in results:
                    # strip out non-ASCII text so we can draw the text on the image
                    # using OpenCV, then draw the text and a bounding box surrounding
                    # the text region of the input image
                    text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
                    print("Cow " + text + " has been drinking!")
            print("[INFO] Done")
        except:
            print("Could not process " + image_name)
