# coding: utf-8
import pandas as pd
import numpy as np
import cv2
import os
import imutils
from PIL import Image
import keras
from keras import Sequential, losses, optimizers
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, Activation
from keras.utils import to_categorical, plot_model,vis_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.callbacks import TensorBoard, EarlyStopping
from keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
from skimage import io
import imutils
import os
os.getcwd()



class Create_Model :

    def __init__(self):
        self.items_data_labels = None
        self.images = []
        self.curr_data_set_list = {}
    def add_image(self, image_array):
        self.images.append(image_array)
    def add_data_labels(self, df):
        self.item_data_labels = df
    def add_data_set_list_data_labels(self, image, emotion):
        self.curr_data_set_list['image'] = image
        self.curr_data_set_list['emotion'] = emotion


def test_image_case(image_url):

    img = imutils.url_to_image(image_url)
    img = cv2.resize(img, (80,80))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    plt.imshow(img)
    prediction = np.argmax(model.predict(img.reshape(1,80,80,1)))
    print ( "The image is for classification: ", get_emotion_from_class(prediction))

def insert_data_set_list(noi=10):
    path = "IMFDB_FINAL"
    data_set_list = [] 
    curr_data_set_list = {"images":[], "emotions":[]}
    for actor_name_list in os.listdir(path)[:noi]:
        data_set_list_model = Create_Model ()
        if actor_name_list == ".DS_Store":
            continue
        print ("looping through actor directory:" + actor_name_list)
        for movie_fn in os.listdir(path+"/"+actor_name_list):
            print ("Goining through movie list : ", movie_fn)
            if movie_fn == ".DS_Store":
                continue
            for items in os.listdir(path+"/"+actor_name_list+"/"+movie_fn):
                if items == ".DS_Store":
                    continue
                if items.endswith(".txt"):
                    print (items)
                    try:
                        print ("added")
                        df = pd.read_table(path+"/"+actor_name_list+"/"+movie_fn+"/"+items,header=None, engine="c")
                        data_set_list_model.add_data_labels(df) 
                        for image in os.listdir(path+"/"+actor_name_list+"/"+movie_fn+"/"+"images"):
                            print (image)
                            if image == ".DS_Store":
                                continue
                            print ("going through image: ", image)
                            for d in df.values:
                                if d[2] == image:  
                                    
                                    im = cv2.imread(path+"/"+actor_name_list+"/"+movie_fn+"/"+"images"+"/"+image)
                                    im = cv2.resize(im, (80,80)) # Changing into 80x80X3
                                    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                                    
                                    
                                    data_set_list_model.add_data_set_list_data_labels(im, d[11])
                                    curr_data_set_list['images'].append(im)
                                    curr_data_set_list['emotions'].append(d[11])  # d[11] is where the emotion is there
                    except:
                        df = None                    
            data_set_list.append(data_set_list_model) # Save all the data_set_list

    return curr_data_set_list




                    


# Inserting the Data
data_set_list = insert_data_set_list(noi=5)

# Extracting the features_data + data_labels
encoder = LabelEncoder()
encoder.fit(data_set_list['emotions'])


# In[12]:
data_set_list['emotions'] = encoder.transform(data_set_list['emotions'])


# In[13]:
all_encoded_class = encoder.classes_
map_dictionary = {0: 'ANGER',
 1: 'ANGER NONE',
 2: 'DISGUST',
 3: 'FEAR',
 4: 'HAPPINESS',
 5: 'NEUTRAL',
 6: 'SADNESS',
 7: 'SURPRISE'}


def get_emotion_from_class(class_number):
    if map_dictionary.get(class_number,None):
        return map_dictionary.get(class_number)
    else:
        return -1 


features_data = np.array(data_set_list['images'])
data_labels = data_set_list['emotions']





data_labels = to_categorical(data_labels, num_classes=len(map_dictionary))



model = Sequential()
model.add(Conv2D(16, kernel_size=5, input_shape=(81,81,1), activation="relu"))
model.add(MaxPool2D(2))
model.add(Dropout(0.25))
model.add(Conv2D(16, kernel_size=5, activation="relu"))
model.add(MaxPool2D(2))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(50, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(50, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(map_dictionary), activation="softmax"))
model.summary()
model.compile(optimizer=optimizers.adam(lr=0.001), loss=keras.losses.categorical_crossentropy, metrics=['acc'])
features_data_train, features_data_test, data_labels_train, data_labels_test = train_test_split(features_data, data_labels, shuffle=True, random_state=34)
tensorboard = TensorBoard()
earlystopping = EarlyStopping(patience=3)
features_data_train = features_data_train.astype("float32")
features_data_test = features_data_test.astype("float32")
features_data_train = features_data_train / 1/255
features_data_test = features_data_test / 1/255
features_data_train = features_data_train.reshape(len(features_data_train), 80, 80, 1)
features_data_test = features_data_test.reshape(len(features_data_test), 80, 80, 1)
model.fit(features_data_train, data_labels_train, epochs=100, batch_size=32, callbacks=[tensorboard,earlystopping], validation_data_set_list=(features_data_test, data_labels_test))







test_image_case("https://i.pinimg.com/originals/3f/40/69/3f40691c27a1f94ab2f79497ed3aebb1.jpg")
test_image_case("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRbo7qhfawDt_1tYBO0pOMM7WLHGhyjtA3JpMfhp88GBII27Z9c")
test_image_case("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQrtm12fpK9P28zmPW2eFHvWceg5gv2-X3JebeXsOh9FVdox7g8tQ")