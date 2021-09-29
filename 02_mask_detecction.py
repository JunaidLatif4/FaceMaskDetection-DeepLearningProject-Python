# ----import libraries
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img,img_to_array
import numpy as np
from joblib import dump,load


# load model #Accuracy=97.4, validation Accuracy=99.1 #very light
path='model.last.h5'
model = load_model(path)

# ----model accept below hight and width of the image
img_width,img_hight = 200,200


# ---Load the Cascade face Classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# ---start webcam
cap = cv2.VideoCapture(0) #for webcam
# cap= cv2.VideoCapture('Downloads\\Make Face Mask from Socks! NO Sew!.mp4') #for video

img_count_full = 0

# ---parameters for text
# ---font
font = cv2.FONT_HERSHEY_COMPLEX
# ---org
org=(1,1)
# ---class label
class_label=' '
fontScale=1
color=(255,3,4) #(B,G,R)
# ---line thickness of 2 px
thickness = 2 

# ----start reading images and prediction
while True:
# while cap.isOpened():
    img_count_full +=1
    # read image from camera
    responce, color_img = cap.read()
    # color_img = cv2.imread('sandeep.jpg')

    # ---if responce False the break the loop
    if responce == False:
        break
    # ---resize image with 50% ratio
    scale = 50
    width = int(color_img.shape[1]*scale/100)
    height = int(color_img.shape[0]*scale/100)
    dim = (width,height)
    # ----resize the image
    color_img = cv2.resize(color_img,dim,interpolation=cv2.INTER_AREA)

    # ----convert to grayscale
    gray_img= cv2.cvtColor(color_img,cv2.COLOR_BGR2GRAY)

    # ----Detect the faces
    faces = face_cascade.detectMultiScale(gray_img,1.1,6) 

    # ----take face then predict class mask or mot mask then draw recrangle and text then
    img_count=0
    for (x,y,w,h) in faces:
        org = (x-10,y-10)
        img_count +=1
        color_face = color_img[y:y+h,x:x+w] #color face
        cv2.imwrite('faces/input/%d%dface.jpg'%(img_count_full,img_count),color_face)
        img=load_img('faces/input/%d%dface.jpg'%(img_count_full,img_count),target_size=(img_width,img_hight))
        img = img_to_array(img)/255
        img = np.expand_dims(img,axis=0)
        pred_prob = model.predict(img,verbose=0)
        # print(pred_prob[0][0].round(2))
        pred = np.argmax(pred_prob)

        if pred==0:
            print("User with mask -predic =",pred_prob[0][0])
            class_label='Mask'
            color=(255,3,4)
            # cv2.imwrite('faces/with_mask/%d%dface.jpg'%(img_count_full,img_count),color_face)
        else:
            print('user not wearning mask -prob', pred_prob[0][1])
            class_label = "No Mask"
            color=(0,255,0)
            # cv2.imwrite('faces/without_mask/%d%dface.jpg'%(img_count_full,img_count),color_face)

        cv2.rectangle(color_img,(x,y),(x+w,y+h),(0,0,225),3)

        # ----Add the text Using cv2.putText() method
        cv2.putText(color_img,class_label,org,font,fontScale,color,thickness,cv2.LINE_4)

# ----Displaying the image
    cv2.imshow("Live face mask detection",color_img)

    if cv2.waitKey(25) & 0xFF ==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
