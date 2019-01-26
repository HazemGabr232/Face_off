#OpenCV module
import cv2
#os module for reading training data directories and paths
import os
#numpy to convert python lists to numpy arrays as it is needed by OpenCV face recognizers
import numpy as np

from cloudant.client import Cloudant
from cloudant.error import CloudantException
from cloudant.result import Result, ResultByKey



client = Cloudant.iam("52bb88bb-586b-4273-ab0a-b15cbffeff05-bluemix", "6_hCZXRWwRuzrTEfRbc7V4jbNwTFybAErt0YFRYLYkWd")
client.connect()
databaseName = "attendanceone"
myDatabase = client.create_database(databaseName)
if myDatabase.exists():
   print( "'{0}' successfully created.\n".format(databaseName))


flag=[0,0,0]
cloud=['0' , '5643f1c8a5133f5db5c4731773fc91a9' , 'a8cd83d68c84b834f4fbf42e0d20ef02']
subjects = ["", "Adel","Hazem"]
font = cv2.FONT_HERSHEY_SIMPLEX



##############################################################
# function to detect face using OpenCV
def detect_face(img):
    # convert the test image to gray scale as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    # load OpenCV face detector, I am using LBP which is fast
    # there is also a more accurate but slow: Haar classifier

    face_cascade = cv2.CascadeClassifier('E:\PROGS\opencv\opencv\sources\data\lbpcascades/lbpcascade_frontalface.xml')
    #print(face_cascade.load('/home/hazem/opencv-3.2.0/data/lbpcascades/lbpcascade_frontalface.xml')) #to check the directory
    # let's detect multiscale images(some images may be closer to camera than others)
    # result is a list of faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);

    # if no faces are detected then return original img
    if (len(faces) == 0):
        return None, None

    # under the assumption that there will be only one face,
    # extract the face area
    (x, y, w, h) = faces[0]

    # return only the face part of the image
    return gray[y:y + w, x:x + h], faces[0]
##############################################################

# this function will read all persons' training images, detect face from each image
# and will return two lists of exactly same size, one list
# of faces and another list of labels for each face
def prepare_training_data(data_folder_path):


    # ------STEP-1--------
    # get the directories (one directory for each subject) in data folder
    dirs = os.listdir(data_folder_path)

    # list to hold all subject faces
    faces = []
    # list to hold labels for all subjects
    labels = []

    # let's go through each directory and read images within it
    for dir_name in dirs:

        # our subject directories start with letter 's' so
        # ignore any non-relevant directories if any
        if not dir_name.startswith("s"):
            continue;

        # ------STEP-2--------
        # extract label number of subject from dir_name
        # format of dir name = slabel
        # , so removing letter 's' from dir_name will give us label
        label = int(dir_name.replace("s", ""))

        # build path of directory containing images for current subject subject
        # sample subject_dir_path = "training-data/s1"
        subject_dir_path = data_folder_path + "/" + dir_name

        # get the images names that are inside the given subject directory
        subject_images_names = os.listdir(subject_dir_path)

    # ------STEP-3--------
    # go through each image name, read image,
    # detect face and add face to list of faces
        for image_name in subject_images_names:

            # ignore system files like .DS_Store
            if image_name.startswith("."):
                continue;

            # build image path
            # sample image path = training-data/s1/1.pgm
            image_path = subject_dir_path + "/" + image_name

            # read image
            image = cv2.imread(image_path)

            # display an image window to show the image
            cv2.imshow("Training on image...", image)
            cv2.waitKey(100)

            # detect face
            face, rect = detect_face(image)

        # ------STEP-4--------
        # we will ignore faces that are not detected
        if face is not None:
            # add face to list of faces
            faces.append(face)
            # add label for this face
            labels.append(label)

        cv2.destroyAllWindows()
        cv2.waitKey(1)
        cv2.destroyAllWindows()

    return faces, labels

#################################################
# function to draw rectangle on image
# according to given (x, y) coordinates and
# given width and heigh
def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)


# function to draw text on give image starting from
# passed (x, y) coordinates.
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
#############################################################

# this function recognizes the person in image passed
# and draws a rectangle around detected face with name of the
# subject
def predict(test_img):


    # make a copy of the image as we don't want to change original image
    img = test_img.copy()
    # detect face from the image
    face, rect = detect_face(img)

    # predict the image using our face recognizer
    label = face_recognizer.predict(face)
    # get name of respective label returned by face recognizer
    label_text = subjects[label[0]]

    # draw a rectangle around face detected
    draw_rectangle(img, rect)
    # draw name of predicted person
    draw_text(img, label_text, rect[0], rect[1] - 5)

    return img
###############################################################
###############################################################


# let's first prepare our training data
# data will be in two lists of same size
# one list will contain all the faces
# and the other list will contain respective labels for each face
print("Preparing data...")
faces, labels = prepare_training_data("C:/Users/Adel/Desktop/faceoff")
print("Data prepared")

# print total faces and labels
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))


face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(faces, np.array(labels))
names = ['None', 'Marcelo', 'Paula', 'Ilza', 'Z', 'W']
# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # set video widht
cam.set(4, 480)  # set video height
# Define min window size to be recognized as a face
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)
while True:
    ret, img = cam.read()
    img = cv2.flip(img, 1)  # Flip vertically
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('E:/PROGS/opencv/opencv/sources/data/lbpcascades/lbpcascade_frontalface.xml')

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH))
    )
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        id, confidence = face_recognizer.predict(gray[y:y + h, x:x + w])
        id=int(id)
        #print (type(id))
        # Check if confidence is less them 100 ==> "0" is perfect match
        if (confidence < 100):
           # id = subjects[id]                             ##if you want to send id as a number make this line a comment
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = 0
            confidence = "  {0}%".format(round(100 - confidence))

        cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

    cv2.imshow('camera', img)
    if id==0:
       idindex=0
    if id==1:
       idindex=1
    if id==2:
       idindex=2
    if id==3:
       idindex=3
    
    if flag[idindex]==0:
        ##send id to the cloud here
        if idindex!=0: #then send it
        #print (id)
            doc_exists = cloud[idindex] in myDatabase


            if doc_exists:
                    # print('document with _id 5643f1c8a5133f5d65c4731773fc91a9 exists')

                my_document = myDatabase[cloud[idindex]]

                my_document['numberField'] = (my_document['numberField'] + 1)

                # You must save the document in order to update it on the database
            flag[idindex]=1
            my_document.save()

        k = cv2.waitKey(10) & 0xff  # Presss 'ESC' for exiting video
        if k == 27:
            break

flag=(0)  #make all flags equal zero

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()
