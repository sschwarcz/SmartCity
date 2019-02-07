
import os
import cv2
import dlib
import numpy as np
import argparse
import  time
import datetime
from dlib import pyramid_down
from wide_resnet import WideResNet
from keras.utils.data_utils import get_file
from imageio import imread
from threading import Thread,Event
import pyrebase



config = {
  "apiKey": "AIzaSyCJOFIBO5g-ZVBNlebfldboRuEYQC-KSRo",
  "authDomain": "smartcity-187de.firebaseapp.com",
  "databaseURL": "https://smartcity-187de.firebaseio.com",
  "storageBucket": "smartcity-187de.appspot.com",
  "serviceAccount": "C:/Users/Dell/Desktop/smartcity.json"
}
firebase = pyrebase.initialize_app(config)
auth = firebase.auth()
#authenticate a user
user = auth.sign_in_with_email_and_password("eladha190@gmail.com", "12345678")
storage1 = firebase.storage()
db = firebase.database()

# pretrained_model = r"C:\Users\Dell\Desktop\projetFinal\pretrained_models\weights.18-4.06.hdf5"

pretrained_model = r"C:\Users\Dell\Desktop\projetFinal\Models\gender.caffemodel"


modhash = '89f56a39a78454e96379348bddd78c0d'


pathToDirectory = "C:\\Users\\Dell\\Desktop\\DirPers\\"

class Index():
    index=0
    Nmb_of_people=0
    age_tot = 0
    count_of_women = 0
    count_of_men = 0

class AgeDetector(Thread):
    def __init__(self,event):
        Thread.__init__(self)
        self.event=event
        pass

    def run(self):

        args = get_args()
        depth = args.depth
        k = args.width
        weight_file = args.weight_file

        if not weight_file:
            weight_file = get_file("weights.18-4.06.hdf5", pretrained_model, cache_subdir="pretrained_models",
                                   file_hash=modhash, cache_dir=os.path.dirname(os.path.abspath(__file__)))

        # for face detection
        detector = dlib.get_frontal_face_detector()

        # load model and weights
        img_size = 64
        index_image = 0
        model = WideResNet(img_size, depth=depth, k=k)()
        model.load_weights(weight_file)

        # list = os.listdir(pathToDirectory)
        photoN = '\\image' + str(index_image) + '.jpg'

        numOfPerson=0
        while True :
            time.sleep(5)
            list = os.listdir(pathToDirectory)

            print("scan...")
            if len(list) != numOfPerson:
                Index.Nmb_of_people=0
                Index.age_tot = 0
                Index.count_of_men=0
                Index.count_of_women=0
                for person in list:
                    Index.Nmb_of_people+=1
                    name1 = pathToDirectory + person + photoN
                    print(person)
                    if os.path.exists(name1):
                        img = imread(name1)
                       
                        input_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img_h, img_w, _ = np.shape(input_img)

                        # detect faces using dlib detector
                        detected = detector(input_img, 1)
                        faces = np.empty((len(detected), img_size, img_size, 3))

                        if len(detected) > 0:

                            for i, d in enumerate(detected):
                                x1, y1, x2, y2, w, h = d.left(), d.top(), d.right() + 1, d.bottom() + 1, d.width(), d.height()
                                xw1 = max(int(x1 - 0.4 * w), 0)
                                yw1 = max(int(y1 - 0.4 * h), 0)
                                xw2 = min(int(x2 + 0.4 * w), img_w - 1)
                                yw2 = min(int(y2 + 0.4 * h), img_h - 1)
                                # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2) ##change la valeur je ne sais pas pourquoi
                                # cv2.rectangle(img, (xw1, yw1), (xw2, yw2), (255, 0, 0), 2)
                                faces[i, :, :, :] = cv2.resize(img[yw1:yw2 + 1, xw1:xw2 + 1, :], (img_size, img_size))

                            # predict ages and genders of the detected faces####################################################################################################################
                            results = model.predict(faces)  ##contien les deux resultats (sexe et age)

                            predicted_genders = results[0]
                            gender = "F" if predicted_genders[i][0] > 0.5 else "M"
                            print(gender)  # imprime le sexe
                            if gender == "M":
                                Index.count_of_men+=1
                            else :
                                Index.count_of_women+=1

                            ages = np.arange(0, 101).reshape(101, 1)
                            predicted_ages = results[1].dot(ages).flatten()
                            age = round(predicted_ages[0], 0)
                            print(age)  # imprime l age
                            print("images/"+person+".jpg")
                            print(name1)
                            Index.age_tot+=age

                            pyrUp= storage1.child("images/"+person+".jpg").put(name1, user['idToken'])

                            picUrl = storage1.child("images/"+person+".jpg").get_url(pyrUp['downloadTokens'])
                            time1 = datetime.datetime.now().time()


                            data = {"DisplayLabel": "Age", "identifier": "text", "showHideLabel" : "1","showInListing" : "1" ,"value" : str(age) }
                            db.child("person").child(person).child("Age").set(data)
                            data = {"DisplayLabel": "Gender", "identifier": "text", "showHideLabel": "1",
                                    "showInListing": "1", "value": gender}
                            db.child("person").child(person).child("Gender").set(data)
                            data = {"DisplayLabel": "Picture", "identifier": "file", "showHideLabel": "1",
                                    "showInListing": "1", "value": str(picUrl)}
                            db.child("person").child(person).child("Picture").set(data)
                            data = {"DisplayLabel": "Time", "identifier": "text", "showHideLabel": "1",
                                    "showInListing": "1",
                                    "value": str(time1)}
                            db.child("person").child(person).child("Number Of People").set(data)
                            if not os.path.isfile(pathToDirectory + '\\' + person + '\\predicted.txt'):
                                f = open(pathToDirectory + '\\' + person + '\\predicted.txt', 'w')
                                f.write('gender = ' + gender + ' age = ' + str(age))
                                f.close()
                data = {"DisplayLabel": "People", "identifier": "text", "showHideLabel": "1",
                            "showInListing": "1", "value": "Total : "+str(Index.Nmb_of_people)+" => Women: "+str(Index.count_of_women)+" Men: "+str(Index.count_of_men)}
                db.child("person").child("person0").child("Number Of People").set(data)
                AVG_age = Index.age_tot/Index.Nmb_of_people
                data = {"DisplayLabel": ".......Age AVG", "identifier": "text", "showHideLabel": "1",
                        "showInListing": "1", "value": AVG_age}
                db.child("person").child("person0").child("Age").set(data)
                    # cv2.imshow("result", img)  # affiche le resultat a l'ecran (facultatif)

            numOfPerson = len(list)

        pass


def get_args():
    parser = argparse.ArgumentParser(description="This script detects faces from web cam input, "
                    "and estimates age and gender for the detected faces.",formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--weight_file", type=str, default=None,help="path to weight file (e.g. weights.18-4.06.hdf5)")
    parser.add_argument("--depth", type=int, default=16,help="depth of network")
    parser.add_argument("--width", type=int, default=8,help="width of network")
    args = parser.parse_args()
    return args


def main():
    event=Event()
    event.clear()
    thread1= AgeDetector(event)
    thread1.start()





if __name__ == '__main__':
    main()
