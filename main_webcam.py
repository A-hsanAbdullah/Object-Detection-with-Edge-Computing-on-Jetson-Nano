import cv2
import numpy as np
import os

# Set up the threshold and NMS threshold for object detection
thres = 0.45  # Threshold to detect object
nms_threshold = 0.5  # NMS

# Set up GStreamer pipeline for Jetson
pipeline = "v4l2src device=/dev/video0 ! video/x-raw, width=1280, height=720 ! videoconvert ! appsink"
cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam with GStreamer pipeline.")
    exit()

# Import the class names for detection
classNames = []
classFile = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'config_files', 'coco.names'))

# Read object classes
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

# Import the config and weights files for the object detection model
configPath = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'config_files', 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'))
weightsPath = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', 'config_files', 'frozen_inference_graph.pb'))

# Load the model with the config and weights
net = cv2.dnn_DetectionModel(weightsPath, configPath)

# Set input parameters for the network
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    # Start Webcam
    success, image = cap.read()

    if not success:
        print("Failed to grab frame")
        break

    # Perform object detection
    classIds, confs, bbox = net.detect(image, confThreshold=thres)

    # Convert bbox and confs to list format for NMS
    bbox = list(bbox)  # Convert tuple to list
    confs = list(np.array(confs).reshape(1, -1)[0])  # Reshape and convert to list
    confs = list(map(float, confs))  # Convert confidence values to float

    # Apply Non-Maximum Suppression (NMS)
    indicies = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)

    if len(indicies) > 0:  # Check if there are any valid detections
        indicies = indicies.flatten()  # Flatten the result
        for i in indicies:
            box = bbox[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            cv2.rectangle(image, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)
            cv2.putText(image, classNames[classIds[i] - 1], (x + 10, y + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    # Show the image with detections
    cv2.imshow("Output", image)

    # Exit the loop if 'ESC' key is pressed
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
