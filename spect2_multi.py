print("[INFO] Importing Libraries")
import matplotlib as plt
import matplotlib.pyplot as plt
plt.style.use('ggplot')
# matplotlib inline
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os
import time   # time1 = time.time(); print('Time taken: {:.1f} seconds'.format(time.time() - time1))
import warnings
import keras
from keras.preprocessing.image import ImageDataGenerator
warnings.filterwarnings("ignore")
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.utils import to_categorical
from keras.preprocessing import image
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from keras import regularizers
from keras import optimizers
from keras.layers import LeakyReLU
from keras.layers import ELU
from keras.models import Model
from keras.layers import Input, Dense
from keras.layers import ZeroPadding2D, GlobalAveragePooling2D, GlobalMaxPooling2D
from PIL import Image 
import numpy
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import roc_auc_score
import time
from sklearn.metrics import classification_report, confusion_matrix
from keras_applications.resnet import ResNet50
from keras_applications.mobilenet import MobileNet
SEED = 50   # set random seed
print("[INFO] Libraries Imported")


#adam = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
#leakyrelu = keras.layers.LeakyReLU(alpha=0.3)
#elu = keras.layers.ELU(alpha=1.0)


from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.utils import plot_model
from keras.applications.inception_resnet_v2 import InceptionResNetV2


#%%

def nn(shape):
    
    
    inputb = Input(shape=shape)
    
    x = Dense(23,activation="relu")(inputb)
    x = BatchNormalization()(x)
    # x = Dense(100,activation="relu")(inputb)
    # x = BatchNormalization()(x)
    # x = Dense(30,activation="relu")(inputb)
    # x = Dense(2,activation="relu")(x)
    
    model = Model(inputs=inputb, outputs=x)
    
    
    return model


def make_inceptionv3 (in_shape, tune, classes):
    
    base_model = keras.applications.InceptionV3(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=in_shape,
    pooling=None,
    classes=classes)
    
    layer_dict = dict([(layer.name, layer) for layer in base_model.layers])
    
    if tune == 1:
        for layer in base_model.layers:
            layer.trainable = True
    
    if tune == 0:
        for layer in base_model.layers:
            layer.trainable = False
        
    if tune is not 0:   
        for layer in base_model.layers:
            layer.trainable = False
        for layer in base_model.layers[tune:]:
            layer.trainable = True
    
    x1 = layer_dict['mixed10'].output 
    x1= GlobalAveragePooling2D()(x1)
    #x1 = Flatten()(x1) 
    
   # x = Flatten()(x1)
    #x = Dense(256, activation='relu')(x)
    x = Dense(2500, activation='relu')(x1)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(classes, activation='softmax')(x)
    model = Model(input=base_model.input, output=x)

    #model.summary()
    model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
    #plot_model(model, to_file='vggmod19.png')
    print("[INFO] Model Compiled!")
    return model   
              
def nn_only(shape):

    inputb = Input(shape=shape)
    
    x = Dense(200,activation="relu")(inputb)
    x = BatchNormalization()(x)
    x = Dense(100,activation="relu")(inputb)
    x = BatchNormalization()(x)
    x = Dense(30,activation="relu")(inputb)
    x = Dense(2,activation="relu")(x)
    
    model = Model(inputs=inputb, outputs=x)
    model.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])
    
    
    return model                  
              
#%%
import csv

csvpath = 'J:\\Python files\\SPECT_P2_600\\lablled\\dset.csv'
imagepath = 'J:\\Python files\\SPECT_P2_600\\lablled\\'

def load_img (csvpath, imagepath):
    print("[INFO] loading images from private data...")
    data = []
    labels = []
    
    label_file = csvpath
    
    with open(label_file, newline='') as csvfile:
        data2 = list(csv.reader(csvfile))
        
    
    
    
    imagePaths = sorted(list(paths.list_images(imagepath)))
    origin = data2
    data_origin = np.array(data2)
    data2 = np.delete(data2,0,0)
    data2 = np.delete(data2,1,1)
    
    datax = np.empty([len(data2),data2.shape[1]])
    k = 0
    for i in range(len(data2)):
        
        found_match = False
        for image in imagePaths:
            path, file = os.path.split(image)
            starts = int(file[:4])
            
            if int(data2[i,0]) == starts:
                  found_match = True
                  
                  for j in range(data2.shape[1]):
                      datax[k,j]=data2[i,j]
                  k=k+1
    print (k)
    
    datax = datax[:k,:]
    
    # grab the image paths and randomly shuffle them
    imagePaths = sorted(list(paths.list_images('J:\\Python files\\SPECT_P2_600\\lablled\\')))   # data folder with 2 categorical folders
    random.seed(SEED)
    random.shuffle(imagePaths)
    found_match = False
    data = []
    labels = []
    data_f = np.empty((0,datax.shape[1]))
        
    for image in imagePaths:
          path, file = os.path.split(image)
          matches = 0
          try:
              starts = int(file[:4])
              print(starts)
          except Exception as err:
              print(err)
              print(image)
              continue
          for i in range(len(datax)):
              
              lista = datax[i,0]
              if int(lista) == starts:
                  found_match = True
                  matches = matches+1
                  
                  where = i
                  point = datax[i,28]
            
          if found_match:
              image = cv2.imread(image)
              image = cv2.resize(image, (200, 200))/255
              
              data_f = np.append(data_f,datax[where,:].reshape(1,-1),axis=0)
    
            
              if int(point) == 1:
                 label = 'z_cad'
                 labels.append(label)
                 data.append(image)
                 
                
              elif int(point) == 0:
                 label = 'healthy'
                 labels.append(label)
                 data.append(image)
              else:
                print(f'Did not find label in {file}')
        
          if not found_match: 
                  print(f'Did not find match in {file}')
          
          found_match = False    
    
    # scale the raw pixel intensities to the range [0, 1]
    data = np.array(data, dtype="float")
    labeltemp=labels
    labels = np.array(labels)
    
    print("[INFO] Private data images loaded!")
    
    print("Reshaping data!")
    
    #data = data.reshape(data.shape[0], 32, 32, 1)
    
    print("Data Reshaped to feed into models channels last")
    
    print("Labels formatting")
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels) 
    labels = keras.utils.to_categorical(labels, num_classes=2, dtype='float32')
    print("Labels ok!")        
    
    return data,labels,data_f,origin,data_origin,data_origin,data2,datax



data,labels,data_f,origin,data_origin,data_origin,data2,datax = load_img(csvpath,imagepath)

data_f_all = data_f

import pandas as pd     

data_f = np.delete(data_f,[0,24,25,26,27,28],1)
            
    

#%%  INCEPTION + RF META

'''INCEPTION + RF META'''


time1 = time.time() #initiate time counter
n_split=10 #10fold cross validation
scores = [] #here every fold accuracy will be kept
predictions_all = np.empty(0) # here, every fold predictions will be kept
predictions_all_num = np.empty([0,2]) # here, every fold predictions will be kept
test_labels = np.empty(0) #here, every fold labels are kept
name2 = 5000 #name initiator for the incorrectly classified insatnces
conf_final = np.array([[0,0],[0,0]]) #initialization of the overall confusion matrix
omega = 1
i = 0
j=4944

for train_index,test_index in KFold(n_split).split(data) :
    trainX,testX=data[train_index],data[test_index]
    trainY,testY=labels[train_index],labels[test_index]
    
    trainX22,testX22=data_f[train_index],data_f[test_index]
    trainY22,testY22=labels[train_index],labels[test_index]
    
    print('[INFO] PREPARING FOLD: '+str(omega))    
    # cnn = make_model()
    cnn = make_inceptionv3((200,200,3), 30, 2)

    
    model3 = cnn
    # history = model3.fit(trainX, trainY,epochs=7, batch_size=64)  

    

    
    aug = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1) #rotation_range=20) #horizontal_flip=True, vertical_flip=True, fill_mode = 'nearest'
    aug.fit(trainX)
    history = model3.fit_generator(aug.flow(trainX, trainY,batch_size=32), epochs=15, steps_per_epoch=len(trainX)//32)
    
    # CNN ONLY
   # history = model3.fit(trainX, trainY,epochs=7, batch_size=64)
    
    

    if omega == 10:
        acc = history.history['accuracy']
        loss = history.history['loss']
        epochs = range(1, len(acc) + 1)
        plt.plot(epochs, acc, 'bo', label='Training accuracy')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.figure()
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.show()
        #model3.summary()

    
    omega = omega + 1



    # COMBINATION
    score = model3.evaluate(testX,testY)
    
    # CNN ONLY
    # score = model3.evaluate(testX,testY)


    
    print(score)
    score = score[1] #keep the accuracy score, not the loss
    scores.append(score) #put the fold score to list
    testY2 = np.argmax(testY, axis=-1) #make the labels 1column array
    print('Model evaluation ',score)
    predict = model3.predict(testX) #for def models functional api
    predict_num = predict
    predict = predict.argmax(axis=-1) #for def models functional api
    conf = confusion_matrix(testY2, predict) #get the fold conf matrix
    conf_final = conf + conf_final #sum it with the previous conf matrix
    name2 = name2 + 1
    predictions_all = np.concatenate([predictions_all, predict]) #merge the two np arrays of predicitons
    predictions_all_num = np.concatenate([predictions_all_num, predict_num])
    test_labels = np.concatenate ([test_labels, testY2]) #merge the two np arrays of labels
    
predictions_confidence = predictions_all_num[:,1]
    
auc_score = roc_auc_score(test_labels,predictions_confidence)
scores = np.asarray(scores)
final_score = np.mean(scores)


print("[INFO] Results Obtained!")
print('Time taken: {:.1f} seconds'.format(time.time() - time1))

#%%
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

final_data = np.concatenate((data_f,predictions_all_num), axis=1)
n_split = 10

rf_accs = []
rf_predictions_all_num = np.empty([0,2])
rf_conf_final = np.array([[0,0],[0,0]])


for train_index,test_index in KFold(n_split).split(final_data):
    trainX,testX=final_data[train_index],final_data[test_index]
    trainY,testY=labels[train_index],labels[test_index]
    
    
    rf = RandomForestClassifier(n_estimators = 300, random_state = 42)
    rf.fit(trainX, trainY)
    rf_pred = rf.predict(testX)
    
    acc = accuracy_score(testY, rf_pred)
    rf_accs.append(acc)

    rf_predictions_all_num = np.concatenate([rf_predictions_all_num,rf_pred])
    
    rf_pred2 = rf_pred.argmax(axis=-1)
    testYa = testY.argmax(axis=-1)
    rf_conf = confusion_matrix(testYa, rf_pred2)
    rf_conf_final = rf_conf_final + rf_conf
    

acc = np.mean(rf_accs)

#%% RF ONLY

'''RF ONLY'''

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

final_data = data_f
n_split = 10

rf_accs = []
rf_predictions_all_num = np.empty([0,2])
rf_conf_final = np.array([[0,0],[0,0]])


for train_index,test_index in KFold(n_split).split(final_data):
    trainX,testX=final_data[train_index],final_data[test_index]
    trainY,testY=labels[train_index],labels[test_index]
    
    
    rf = RandomForestClassifier(n_estimators = 50, random_state = 42)
    rf.fit(trainX, trainY)
    rf_pred = rf.predict(testX)
    
    acc = accuracy_score(testY, rf_pred)
    rf_accs.append(acc)

    rf_predictions_all_num = np.concatenate([rf_predictions_all_num,rf_pred])
    
    rf_pred2 = rf_pred.argmax(axis=-1)
    testYa = testY.argmax(axis=-1)
    rf_conf = confusion_matrix(testYa, rf_pred2)
    rf_conf_final = rf_conf_final + rf_conf
    

acc = np.mean(rf_accs)


#%%
from sklearn.datasets import load_iris
iris = load_iris()

# Model (can also use single decision tree)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=10)

# Train
model.fit(iris.data, iris.target)
# Extract single tree
estimator = model.estimators_[5]

feature_names = []
from sklearn.tree import export_graphviz
# Export as dot file
export_graphviz(estimator, out_file='tree.dot', 
                feature_names = iris.feature_names,
                class_names = iris.target_names,
                rounded = True, proportion = False, 
                precision = 2, filled = True)

# Convert to png using system command (requires Graphviz)
from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

# Display in jupyter notebook
from IPython.display import Image
Image(filename = 'tree.png')


#%% INCEPTION + NN (INTRA) 

'''INCEPTION + NEURAL NETWORK INTRA'''

time1 = time.time() #initiate time counter
n_split=10 #10fold cross validation
scores = [] #here every fold accuracy will be kept
predictions_all = np.empty(0) # here, every fold predictions will be kept
predictions_all_num = np.empty([0,2]) # here, every fold predictions will be kept
test_labels = np.empty(0) #here, every fold labels are kept
name2 = 5000 #name initiator for the incorrectly classified insatnces
conf_final = np.array([[0,0],[0,0]]) #initialization of the overall confusion matrix
omega = 1
i = 0
j=4944

shape = (23,)

for train_index,test_index in KFold(n_split).split(data) :
    trainX,testX=data[train_index],data[test_index]
    trainY,testY=labels[train_index],labels[test_index]
    
    trainX22,testX22=data_f[train_index],data_f[test_index]
    trainY22,testY22=labels[train_index],labels[test_index]
    
    
    cnn = make_inceptionv3((200,200,3), 30, 2)
    
    # COMBINATION
    nn1 = nn(shape)    
    combinedInput = keras.layers.concatenate([cnn.output, nn1.output])    
    x = Dense(500,activation="relu")(combinedInput)
    x = Dropout(0.2)(x)
    x = Dense(250,activation="relu")(x)
    x = Dropout(0.2)(x)    
    x = Dense(100,activation="relu")(x)
    x = Dropout(0.2)(x)    
    x = Dense(25,activation="relu")(x)
    x = Dense(2,activation="softmax")(x)
    model3 = Model(inputs=[cnn.input, nn1.input], outputs=x)
    model3.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])   
        
    print('[INFO] PREPARING FOLD: '+str(omega))
    

    history = model3.fit(x=[trainX,trainX22], y=trainY,epochs=15, batch_size=64)
    

    if omega == 10:
        acc = history.history['accuracy']
        loss = history.history['loss']
        epochs = range(1, len(acc) + 1)
        plt.plot(epochs, acc, 'bo', label='Training accuracy')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.figure()
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.show()

    
    omega = omega + 1



    # COMBINATION
    score = model3.evaluate([testX,testX22],testY)
        
    print(score)
    score = score[1] #keep the accuracy score, not the loss
    scores.append(score) #put the fold score to list
    testY2 = np.argmax(testY, axis=-1) #make the labels 1column array
    print('Model evaluation ',score)
    predict = model3.predict([testX,testX22]) #for def models functional api
    predict_num = predict
    predict = predict.argmax(axis=-1) #for def models functional api
    conf = confusion_matrix(testY2, predict) #get the fold conf matrix
    conf_final = conf + conf_final #sum it with the previous conf matrix
    name2 = name2 + 1
    predictions_all = np.concatenate([predictions_all, predict]) #merge the two np arrays of predicitons
    predictions_all_num = np.concatenate([predictions_all_num, predict_num])
    test_labels = np.concatenate ([test_labels, testY2]) #merge the two np arrays of labels
    
predictions_confidence = predictions_all_num[:,1]
    
auc_score = roc_auc_score(test_labels,predictions_confidence)
scores = np.asarray(scores)
final_score = np.mean(scores)


print("[INFO] Results Obtained!")
print('Time taken: {:.1f} seconds'.format(time.time() - time1))


#%% NN ONLY

shape = (23,)

'''NEURAL NETWORK ONLY'''

time1 = time.time() #initiate time counter
n_split=10 #10fold cross validation
scores = [] #here every fold accuracy will be kept
predictions_all = np.empty(0) # here, every fold predictions will be kept
predictions_all_num = np.empty([0,2]) # here, every fold predictions will be kept
test_labels = np.empty(0) #here, every fold labels are kept
name2 = 5000 #name initiator for the incorrectly classified insatnces
conf_final = np.array([[0,0],[0,0]]) #initialization of the overall confusion matrix
omega = 1
i = 0
j=4944

for train_index,test_index in KFold(n_split).split(data_f) :
    trainX,testX=data_f[train_index],data_f[test_index]
    trainY,testY=labels[train_index],labels[test_index]
    

    
    print('[INFO] PREPARING FOLD: '+str(omega))    
    model3 = nn_only(shape)
    history = model3.fit(trainX, trainY,epochs=20, batch_size=64)  


    if omega == 10:
        acc = history.history['accuracy']
        loss = history.history['loss']
        epochs = range(1, len(acc) + 1)
        plt.plot(epochs, acc, 'bo', label='Training accuracy')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.figure()
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.show()
        #model3.summary()

    
    omega = omega + 1



    # COMBINATION
    score = model3.evaluate(testX,testY)
    


    
    print(score)
    score = score[1] #keep the accuracy score, not the loss
    scores.append(score) #put the fold score to list
    testY2 = np.argmax(testY, axis=-1) #make the labels 1column array
    print('Model evaluation ',score)
    predict = model3.predict(testX) #for def models functional api
    predict_num = predict
    predict = predict.argmax(axis=-1) #for def models functional api
    conf = confusion_matrix(testY2, predict) #get the fold conf matrix
    conf_final = conf + conf_final #sum it with the previous conf matrix
    name2 = name2 + 1
    predictions_all = np.concatenate([predictions_all, predict]) #merge the two np arrays of predicitons
    predictions_all_num = np.concatenate([predictions_all_num, predict_num])
    test_labels = np.concatenate ([test_labels, testY2]) #merge the two np arrays of labels
    
predictions_confidence = predictions_all_num[:,1]
    
auc_score = roc_auc_score(test_labels,predictions_confidence)
scores = np.asarray(scores)
final_score = np.mean(scores)





#%% INCEPTION + NN (META)

'''INCEPTION + NN (META)'''

time1 = time.time() #initiate time counter
n_split=2 #10fold cross validation
scores = [] #here every fold accuracy will be kept
predictions_all = np.empty(0) # here, every fold predictions will be kept
predictions_all_num = np.empty([0,2]) # here, every fold predictions will be kept
test_labels = np.empty(0) #here, every fold labels are kept
name2 = 5000 #name initiator for the incorrectly classified insatnces
conf_final = np.array([[0,0],[0,0]]) #initialization of the overall confusion matrix
omega = 1
i = 0
j=4944

for train_index,test_index in KFold(n_split).split(data) :
    trainX,testX=data[train_index],data[test_index]
    trainY,testY=labels[train_index],labels[test_index]
    
    trainX22,testX22=data_f[train_index],data_f[test_index]
    trainY22,testY22=labels[train_index],labels[test_index]
    
    print('[INFO] PREPARING FOLD: '+str(omega))    
    # cnn = make_model()
    cnn = make_inceptionv3((200,200,3), 0, 2)

    
    model3 = cnn
    history = model3.fit(trainX, trainY,epochs=1, batch_size=64)  

    

    
    # aug = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1) #rotation_range=20) #horizontal_flip=True, vertical_flip=True, fill_mode = 'nearest'
    # aug.fit(trainX)
    # history = model3.fit_generator(aug.flow(trainX, trainY,batch_size=32), epochs=25, steps_per_epoch=len(trainX)//32)
    
    # CNN ONLY
   # history = model3.fit(trainX, trainY,epochs=7, batch_size=64)
    
    

    if omega == 10:
        acc = history.history['accuracy']
        loss = history.history['loss']
        epochs = range(1, len(acc) + 1)
        plt.plot(epochs, acc, 'bo', label='Training accuracy')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.figure()
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.show()
        #model3.summary()

    
    omega = omega + 1



    # COMBINATION
    score = model3.evaluate(testX,testY)
    
    # CNN ONLY
    # score = model3.evaluate(testX,testY)


    
    print(score)
    score = score[1] #keep the accuracy score, not the loss
    scores.append(score) #put the fold score to list
    testY2 = np.argmax(testY, axis=-1) #make the labels 1column array
    print('Model evaluation ',score)
    predict = model3.predict(testX) #for def models functional api
    predict_num = predict
    predict = predict.argmax(axis=-1) #for def models functional api
    conf = confusion_matrix(testY2, predict) #get the fold conf matrix
    conf_final = conf + conf_final #sum it with the previous conf matrix
    name2 = name2 + 1
    predictions_all = np.concatenate([predictions_all, predict]) #merge the two np arrays of predicitons
    predictions_all_num = np.concatenate([predictions_all_num, predict_num])
    test_labels = np.concatenate ([test_labels, testY2]) #merge the two np arrays of labels
    
predictions_confidence = predictions_all_num[:,1]
    
auc_score = roc_auc_score(test_labels,predictions_confidence)
scores = np.asarray(scores)
final_score = np.mean(scores)


final_data = np.concatenate([data_f,predictions_all])

shape = (25,)

'''NEURAL NETWORK ONLY'''

time1 = time.time() #initiate time counter
n_split=2 #10fold cross validation
scores = [] #here every fold accuracy will be kept
predictions_all = np.empty(0) # here, every fold predictions will be kept
predictions_all_num = np.empty([0,2]) # here, every fold predictions will be kept
test_labels = np.empty(0) #here, every fold labels are kept
name2 = 5000 #name initiator for the incorrectly classified insatnces
conf_final = np.array([[0,0],[0,0]]) #initialization of the overall confusion matrix
omega = 1
i = 0
j=4944

for train_index,test_index in KFold(n_split).split(data_f) :
    trainX,testX=final_data[train_index],final_data[test_index]
    trainY,testY=labels[train_index],labels[test_index]
    

    
    print('[INFO] PREPARING FOLD: '+str(omega))    
    model3 = nn_only(shape)
    history = model3.fit(trainX, trainY,epochs=1, batch_size=64)  


    if omega == 10:
        acc = history.history['accuracy']
        loss = history.history['loss']
        epochs = range(1, len(acc) + 1)
        plt.plot(epochs, acc, 'bo', label='Training accuracy')
        plt.title('Training and validation accuracy')
        plt.legend()
        plt.figure()
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.title('Training and validation loss')
        plt.legend()
        plt.show()
        #model3.summary()

    
    omega = omega + 1



    # COMBINATION
    score = model3.evaluate(testX,testY)
    


    
    print(score)
    score = score[1] #keep the accuracy score, not the loss
    scores.append(score) #put the fold score to list
    testY2 = np.argmax(testY, axis=-1) #make the labels 1column array
    print('Model evaluation ',score)
    predict = model3.predict(testX) #for def models functional api
    predict_num = predict
    predict = predict.argmax(axis=-1) #for def models functional api
    conf = confusion_matrix(testY2, predict) #get the fold conf matrix
    conf_final = conf + conf_final #sum it with the previous conf matrix
    name2 = name2 + 1
    predictions_all = np.concatenate([predictions_all, predict]) #merge the two np arrays of predicitons
    predictions_all_num = np.concatenate([predictions_all_num, predict_num])
    test_labels = np.concatenate ([test_labels, testY2]) #merge the two np arrays of labels
    
predictions_confidence = predictions_all_num[:,1]
    
auc_score = roc_auc_score(test_labels,predictions_confidence)
scores = np.asarray(scores)
final_score = np.mean(scores)

#%%

model3.save('J:\\Python files\\SPECT_P2_600\\lvgg.h5')

from matplotlib import pyplot
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.models import  load_model


# load the model
model = load_model('J:\\Python files\\SPECT_P2_600\\lvgg.h5')
# redefine model to output right after the first hidden layer
ixs = [2, 12, 30, 40, 80]
# ixs = [2,4]
outputs = [model.layers[i].output for i in ixs]
model = Model(inputs=model.inputs, outputs=outputs)
# load the image with the required shape
img = load_img('J:\\Python files\\SPECT_P2_600\\SPECT 2 (4POLARS)\\CANG PTS\\0002-VERRAS ALEXIOS\\0002-VERRAS ALEXIOS_ 4IN1 .TIFF', target_size=(200, 200))
# convert the image to an array
img = img_to_array(img)
# expand dimensions so that it represents a single 'sample'
img = expand_dims(img, axis=0)
# prepare the image (e.g. scale pixel values for the vgg)
img = preprocess_input(img)
# get feature map for first hidden layer
feature_maps = model.predict(img)
# plot the output from each block
square = 4
for fmap in feature_maps:
	# plot all 64 maps in an 8x8 squares
	ix = 1
	for _ in range(square):
		for _ in range(square):
			# specify subplot and turn of axis
			ax = pyplot.subplot(square, square, ix)
			ax.set_xticks([])
			ax.set_yticks([])
			# plot filter channel in grayscale
			pyplot.imshow(fmap[0, :, :, ix-1], cmap='magma_r')
			ix += 1
	# show the figure
	pyplot.show()