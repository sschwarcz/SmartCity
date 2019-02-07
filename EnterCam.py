# active la camera ,
#repere le visage
#prends une photo
#rogne la photo
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
test_directory=r'C:\Users\Dell\Desktop\DirExit'

class Index():
    indexImg = 0
    indexDir = 1
    indexPhotog = 0
    img_size = 64
    NbFaces = 0

    cam=None
    faces=None
    frame = None



class Camera(Thread):####################################CAMERA

    def __init__(self,cam,faces,frame):
        Thread.__init__(self)
        self.frame=frame
        self.faces=faces
        self.cam=cam
        self.n_img_per_person=1


    def moveImage(self,path):
        os.rename('C:\\Users\\Dell\\Desktop\\DirPers\\'+path,test_directory+'\\'+path)
        pass

    def getNewPicture(self):
        self.frame =Index.frame
        self.cam=Index.cam
        self.faces=Index.faces
        pass



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
        already = self.isDouble(aligned)


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

        for face in self.faces:
            photo = PhotoIndexes()
            i = 0

            while not i == self.n_img_per_person:  # 10 pictures have been taken
                (x, y, w, h) = face_utils.rect_to_bb(face)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

                self.PrintPicture(photo, self.cam, x - 20, y-80, w+40, h+120 )
                i += 1 #number of picture for each man (=1)

            Index.indexImg=0
            display.clear_output(wait=True)


        print("sorti")




class PhotoIndexes():
    synchronyzed = 1;

    def __init__(self):
        Thread.__init__(self)

        self.created = False  # boolean for not duplicate mkdir
        self.no=Index.indexPhotog;
        Index.indexPhotog+=1
        self.pathDir = ""
        self.nameImgDIR = ""






class FaceDemo(object):###############la camera dans le vide
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
        cam = VideoStream(0).start()#########################################################################################################
        ###############################################################################################################################
        self.vc=cam





        fig = plt.figure(0)
        fig.canvas.set_window_title('כניסה')
        Index.NbFaces = 0
        detector = dlib.get_frontal_face_detector()#dlib
        predictor = dlib.shape_predictor(predictor_path)



        while True:

            frame = cam.read()
            # frame = imutils.resize(frame, width=400)
            Index.cam=cam
            gray = frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)##give a normal color to the camera
             # faces = detector(gray, 0)
            faces = detector(gray, 1)

            Index.faces = faces

            if len(faces)!=0:

                Index.frame=frame
                for face in faces:

                     # explain face predictor
                    shape = predictor(gray, face)
                    shape = face_utils.shape_to_np(shape)
                    for (x, y) in shape:##shape.length=68
                        cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

                    # (x, y, w, h) = face_utils.rect_to_bb(face)
                    # cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                if len(faces) > Index.NbFaces:


                    print('1-create new thread')

                    thread1 = Camera(cam,faces,frame)
                    thread1.start()


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


print(colored("start",'blue'))
f = FaceDemo(predictor_path)
f.capture_images('ENTER')