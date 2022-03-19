from flask import Flask, render_template, request
from flask_pymongo import PyMongo
import cv2
import numpy as np
import pickle
from PIL import Image
from io import BytesIO
import base64 

#Models
#facedetection
face_detector_model = cv2.dnn.readNetFromCaffe('deploy.prototxt.txt','./res10_300x300_ssd_iter_140000_fp16.caffemodel')
#feature extraction
face_feature_model = cv2.dnn.readNetFromTorch('openface.nn4.small2.v1.t7')
#emotion recognition model
emotion_recognition_model = pickle.load(open('face_emotion_model_mlp.pkl',mode='rb'))


app = Flask(__name__)

app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 1

app.config["MONGO_URI"] = "mongodb+srv://Sharath067:Skumar(66@songs.qcqcg.mongodb.net/songs?retryWrites=true&w=majority"


mongo = PyMongo(app)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/selfie', methods=['GET', 'POST'])
def selfie():
    # img = request.files['file1']
    im = request.form['file1']
    img = Image.open(BytesIO(base64.b64decode(im)))
    
    img.save('static/file.png')

    ####################################
    image = cv2.imread('static/file.png')
    h,w = image.shape[:2]
    img_blob = cv2.dnn.blobFromImage(image,1,(300,300),(104,177,123),swapRB=False,crop=False)
    face_detector_model.setInput(img_blob)
    detections = face_detector_model.forward()
    if len(detections) > 0:
        i = np.argmax(detections[0,0,:,2])
        confidence = detections[0,0,i,2]

        if confidence > 0.5:
            box= detections[0,0,i,3:7] * np.array([w,h,w,h])
            startx,starty,endx,endy = box.astype('int')

             #Drawing Rectangle around the face
            cv2.rectangle(image,(startx,starty),(endx,endy),(0,255,0),2)

            #feature extraction
            face_roi = image[starty:endy,startx:endx]
            face_blob = cv2.dnn.blobFromImage(face_roi,1/255,(96,96),(0,0,0),swapRB=True,crop=True)

            face_feature_model.setInput(face_blob)
            vectors = face_feature_model.forward()

            #Emotion
            emotion = emotion_recognition_model.predict(vectors)[0]
            emotion_confidence  = emotion_recognition_model.predict_proba(vectors).max()*100
            #text_emotion = '{}-{:.2f}%'.format(emotion,emotion_confidence)

            text_emotion = '{}'.format(emotion)
            cv2.putText(image,text_emotion,(startx+20,starty+20),cv2.FONT_HERSHEY_PLAIN,1.5,(0,255,0),2)
            if text_emotion == 'angry':
                songs_collection = mongo.db.angry.find()
            elif text_emotion == 'happy':
                songs_collection = mongo.db.happy.find()
            elif text_emotion == 'sad':
                songs_collection = mongo.db.sad.find()
            else:
                songs_collection = mongo.db.neutral.find()
        else:
            text_emotion = "none"
            songs_collection = "none"
    


    cv2.imwrite('static/selfie.png', image)


    return render_template('selfie.html',data=text_emotion,songs= songs_collection)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    img = request.files['file2']
    
    img.save('static/file2.png')

    ####################################
    image = cv2.imread('static/file2.png')
    h,w = image.shape[:2]
    img_blob = cv2.dnn.blobFromImage(image,1,(300,300),(104,177,123),swapRB=False,crop=False)
    face_detector_model.setInput(img_blob)
    detections = face_detector_model.forward()
    if len(detections) > 0:
        i = np.argmax(detections[0,0,:,2])
        confidence = detections[0,0,i,2]

        if confidence > 0.5:
            box= detections[0,0,i,3:7] * np.array([w,h,w,h])
            startx,starty,endx,endy = box.astype('int')

             #Drawing Rectangle around the face
            cv2.rectangle(image,(startx,starty),(endx,endy),(0,255,0),2)

            #feature extraction
            face_roi = image[starty:endy,startx:endx]
            face_blob = cv2.dnn.blobFromImage(face_roi,1/255,(96,96),(0,0,0),swapRB=True,crop=True)

            face_feature_model.setInput(face_blob)
            vectors = face_feature_model.forward()

            #Emotion
            emotion = emotion_recognition_model.predict(vectors)[0]
            emotion_confidence  = emotion_recognition_model.predict_proba(vectors).max()*100
            #text_emotion = '{}-{:.2f}%'.format(emotion,emotion_confidence)

            text_emotion = '{}'.format(emotion)
            cv2.putText(image,text_emotion,(startx+20,starty+20),cv2.FONT_HERSHEY_PLAIN,1.5,(0,255,0),2)
            if text_emotion == 'angry':
                songs_collection = mongo.db.angry.find()
            elif text_emotion == 'happy':
                songs_collection = mongo.db.happy.find()
            elif text_emotion == 'sad':
                songs_collection = mongo.db.sad.find()
            else:
                songs_collection = mongo.db.neutral.find()
        else:
            text_emotion = "none"
            songs_collection = "none"


    cv2.imwrite('static/upload.png', image)


    return render_template('upload.html',data=text_emotion, songs= songs_collection)


if __name__ == "__main__":
    app.run(debug=True)