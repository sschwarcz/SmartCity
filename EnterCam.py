#Samuel Schwarcz

from builtins import print
from termcolor import colored #not important
import cv2
import matplotlib.pyplot as plt
import signal
from IPython import display
import dlib
from skimage.transform import resize
from threading import Thread
import os
import face_recognition

import sys
from imutils.video import VideoStream
import imutils
from imutils import face_utils
import argparse


predictor_path = r'C:\Users\Dell\Desktop\projetFinal\Models\shape_predictor_68_face_landmarks2.dat'

database_directory='C:\\Users\\Dell\\Desktop\\DirPers\\person'
#directory for the trash
test_directory=r'C:\Users\Dell\Desktop\DirExit'

class Index():#indexes are used to save or move picture 
    indexImg = 0
    indexDir = 1
    indexPhotog = 0
    img_size = 64
    NbFaces = 0
#this inexes are used in case of not readeability of the picture the program will take new datas
    cam=None
    faces=None
    frame = None


#step 3 the camera has detected people and crated a new thread that will crop resize and save every people on the frame
class Camera(Thread):

    def __init__(self,cam,faces,frame):
        Thread.__init__(self)
        self.frame=frame
        self.faces=faces
        self.cam=cam
        self.n_img_per_person=1

    # move the picture to the trash directory
    def moveImage(self,path):
        os.rename('C:\\Users\\Dell\\Desktop\\DirPers\\'+path,test_directory+'\\'+path)
        pass

    # get the new data picture from the camera runing in the back that update these data
    def getNewPicture(self):
        self.frame =Index.frame
        self.cam=Index.cam
        self.faces=Index.faces
        pass


    #check in the directory if there are already such of a person using a model that will encode the picture 
    #to compare it withe the other picture from the directory, if there is a problem 
    #with the encoding it will move the picture to the trash 
    def isDouble(self,picture):
        image_dir_basepath = 'C:\\Users\\Dell\\Desktop\\DirPers\\'
        list = os.listdir(image_dir_basepath)
        check_man_image = picture
        # check_man_encoding1 = face_recognition.face_encodings(check_man_image)
        # print(check_man_encoding1)
        check_man_encodings = face_recognition.face_encodings(check_man_image)
        if len(check_man_encodings) > 0:
            check_man_encoding = check_man_encodings[0]
        else:
            print(colored("problem with the picture encoding", 'green'))
            self.getNewPicture();
            self.run()
            return True
        for namePic in list:
            path = image_dir_basepath + namePic + '\\image0.jpg'
            if not os.path.exists(path):
                self.moveImage(namePic)
                print(colored("problem with the path\npicture moved", 'green'))
            else:
                search_image = face_recognition.load_image_file(path)


                search_encodings = face_recognition.face_encodings(search_image)

                if len(search_encodings) > 0:
                    search_encoding = search_encodings[0]
                else:
                    print(colored("problem with the search encoding\npicture moved", 'green'))
                    self.moveImage(namePic)
                    continue
                results = face_recognition.compare_faces([check_man_encoding], search_encoding,tolerance=0.6)#compare les visages
                print(results)
                if (results[0] == True):
                    return True

        return False
#step 4: croping and resizing every faces
    def PrintPicture(self, photo, cam,x ,y ,w ,h):

        frame = cam.read()

        crop_img = frame[y:y + h, x:x + w] ##crop the picture

        img = crop_img

        try:
            aligned = cv2.resize(img, (320, 430)) ## resize the picture
        except Exception:
            print(colored("problem with the resize",'green'))
            self.getNewPicture();
            self.run()
            return
        already = self.isDouble(aligned)  # check the doubles


        if(already==True):

            print("already pictured")

        else:
            if (photo.created == False):  # IF for creating new file

                isDir = True

                while (isDir == True):

                    namedir = database_directory + str(Index.indexDir)
                    isDir = os.path.isdir(namedir)
                    Index.indexDir += 1

                os.mkdir(namedir)
                photo.pathDir = namedir

                photo.created = True

            nom = photo.pathDir + '\\image' + str(Index.indexImg) + '.jpg'  # the name of the image
            photo.nameImgDIR=nom
            Index.indexImg += 1  # the index of the image
            cv2.imwrite(filename=nom, img=aligned)  ## to save the frame
        pass

    def run(self):########################################################################################################

        print("2-Camera found new face!")
        frame = self.frame
         #every faces will be worked independently
        for face in self.faces:
            photo = PhotoIndexes()
            i = 0

            while not i == self.n_img_per_person:  # in case i want to take more than one picture
                (x, y, w, h) = face_utils.rect_to_bb(face)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                #fonction that will crop and resize
                self.PrintPicture(photo, self.cam, x - 20, y-80, w+40, h+120 )
                i += 1 #number of picture for each man (=1)

            Index.indexImg=0
            display.clear_output(wait=True)


        print("sorti")
##end CAMERA


#class for saving picture data
class PhotoIndexes():
    synchronyzed = 1;

    def __init__(self):
        Thread.__init__(self)

        self.created = False  # boolean for not duplicate mkdir
        self.no=Index.indexPhotog;
        Index.indexPhotog+=1
        self.pathDir = ""
        self.nameImgDIR = ""
## end PhotoIndex


#step 2 the camera works on the back and wait for people 
class FaceDemo(object):###############the camera works 
    def __init__(self, cascade_path):
        self.vc = None
        self.predictor = cascade_path
        self.margin = 10
        self.batch_size = 1
        self.n_img_per_person = 10
        self.is_interrupted = False
        self.data = {}


    def _signal_handler(self, signal, frame):
        self.is_interrupted = True

    def capture_images(self , name='Unknown'):
        cam = VideoStream(0).start()
        self.vc=cam





        fig = plt.figure(0)
        fig.canvas.set_window_title('כניסה')
        Index.NbFaces = 0
        detector = dlib.get_frontal_face_detector()#dlib
        predictor = dlib.shape_predictor(predictor_path)

 
        #the camera continue to work in the back 
        #when it detect faces it creates a new thread that will take the frame it will crop it on the face resize it
        #check if the picture taken is clean to work with and save it into a directory with all the people that entered
        while True:

            frame = cam.read()
            # frame = imutils.resize(frame, width=400)
            #save the camera in an index case of not readeability of the face or the frame
            Index.cam=cam
            gray = frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)##give a normal color to the camera
             # faces = detector(gray, 0)
            faces = detector(gray, 1)
            #save the face from detector in case of not readeability of the face or the frame
            Index.faces = faces

            if len(faces)!=0:
                   #save the frame in case of not readeability of the face or the frame
                Index.frame=frame
                for face in faces:

                     # this part is completly optional , is it for circlying people when the frame appear 
                     #on the screen of the computer but will not influence the all course of the script
                    shape = predictor(gray, face)
                    shape = face_utils.shape_to_np(shape)
                    for (x, y) in shape:##shape.length=68
                        cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

                    # (x, y, w, h) = face_utils.rect_to_bb(face)
                    # cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    
                    
                #check if the number of face have increased (that will mean that there is a other person
                #that appeared on the camera so we have to create a new thread that will save it)
                if len(faces) > Index.NbFaces:

                    print('1-create new thread')

                    thread1 = Camera(cam,faces,frame)
                    thread1.start()

             ##optional for the screen presentation
            cv2.imshow("Frame",frame)
            # plt.title("Found {0} faces!".format(len(faces)))
            # plt.xticks([])
            # plt.yticks([])
            # display.clear_output(wait=True)
            
            Index.NbFaces = len(faces)

            if cv2.waitKey(1) & 0xFF == ord('q'):

                break

        cv2.destroyAllWindows()
        cam.stop()
##end FaceDemo


#step 1 start the script
print(colored("start",'blue'))
f = FaceDemo(predictor_path)
f.capture_images('ENTER')
