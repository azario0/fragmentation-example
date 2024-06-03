import streamlit as st
import cv2
import numpy as np
import mediapipe as mp

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
    return frameOpencvDnn, faceBoxes

# Load pre-trained models
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# Initialize MediaPipe Selfie Segmentation
mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# Streamlit app
st.title('Gender and Age Detection with Segmentation')
st.write('Upload an image and the app will segment the person, detect faces, predict gender and age.')

# File uploader
uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Convert the uploaded file to an OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    # Display the original image
    st.image(image, channels='BGR', caption='Uploaded Image')

    # Convert the image to RGB for MediaPipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform segmentation
    result = selfie_segmentation.process(image_rgb)

    # Extract mask and apply it to the image
    mask = result.segmentation_mask > 0.5
    segmented_image = np.zeros_like(image)
    segmented_image[mask] = image[mask]

    # Display the segmented image
    st.image(segmented_image, channels='BGR', caption='Segmented Image')

    # Perform face detection on the segmented image
    resultImg, faceBoxes = highlightFace(faceNet, segmented_image)
    
    if not faceBoxes:
        st.write("No face detected")
    else:
        for faceBox in faceBoxes:
            face = segmented_image[max(0, faceBox[1]-20):
                                   min(faceBox[3]+20, segmented_image.shape[0]-1),
                                   max(0, faceBox[0]-20)
                                   :min(faceBox[2]+20, segmented_image.shape[1]-1)]

            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            
            # Gender prediction
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]

            # Age prediction
            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]

            # Display results
            label = f'{gender}, {age}'
            cv2.putText(resultImg, label, (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        # Display the result image
        st.image(resultImg, channels='BGR', caption='Result Image')

