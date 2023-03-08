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
def make_half_vgg():

    base_model = keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_tensor=None, input_shape=(300, 300, 3), pooling=None, classes=2)
    #base_model = keras.applications.resnet.ResNet50(include_top=False, weights='imagenet', input_shape=(32, 32, 3), pooling='max')
    layer_dict = dict([(layer.name, layer) for layer in base_model.layers])
    #base_model.summary()
    for layer in base_model.layers:
        layer.trainable = False
    for layer in base_model.layers[6:]:
        layer.trainable = True
        
    map1 = layer_dict['block2_conv2']
    
    early2 = layer_dict['block2_conv2'].output 
    #early2 = Conv2D(64, (3, 3), padding='valid', activation='relu', kernel_regularizer=regularizers.l2(0.001))(early2)
    early2 = BatchNormalization()(early2)
    early2 = Dropout(0.5)(early2)
    early2= GlobalAveragePooling2D()(early2)
    
    x = Dense(2500, activation='relu')(early2)
    x = Dropout(0.5)(x)
    
    x = Dense(2, activation='softmax')(x)
    model = Model(input=base_model.input, output=x)
    #for layer in model.layers[:17]:
        #layer.trainable = True
    
     
    # for layer in model.layers[17:]:
    #     layer.trainable = True  
    #model.summary()
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #plot_model(model, to_file='vggmod19.png')
    print("[INFO] Model Compiled!")
    return model
    
    
    

def make_lvgg():
    
#import pydot
    
    base_model = keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_tensor=None, input_shape=(300, 300, 3), pooling=None, classes=2)
    #base_model = keras.applications.resnet.ResNet50(include_top=False, weights='imagenet', input_shape=(32, 32, 3), pooling='max')
    layer_dict = dict([(layer.name, layer) for layer in base_model.layers])
    #base_model.summary()
    for layer in base_model.layers:
        layer.trainable = False
    for layer in base_model.layers[18:]:
        layer.trainable = True
    #base_model.summary()
    
    # early1 = layer_dict['block1_pool'].output
    # #early1 = Conv2D(32, (3, 3), padding='valid', activation='relu', kernel_regularizer=regularizers.l2(0.001))(early1)
    # early1 = BatchNormalization()(early1)
    # early1 = Dropout(0.5)(early1)
    # early1= GlobalAveragePooling2D()(early1)
    # #early1 = Flatten()(early1)
    
    map1 = layer_dict['block2_conv2']
    
    early2 = layer_dict['block2_pool'].output 
    #early2 = Conv2D(64, (3, 3), padding='valid', activation='relu', kernel_regularizer=regularizers.l2(0.001))(early2)
    early2 = BatchNormalization()(early2)
    early2 = Dropout(0.5)(early2)
    early2= GlobalAveragePooling2D()(early2)
        
    early3 = layer_dict['block3_pool'].output   
    #early3 = Conv2D(128, (3, 3), padding='valid', activation='relu', kernel_regularizer=regularizers.l2(0.001))(early3)
    early3 = BatchNormalization()(early3)
    early3 = Dropout(0.5)(early3)
    early3= GlobalAveragePooling2D()(early3)    
        
    early4 = layer_dict['block4_pool'].output   
    #early4 = Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001))(early4)
    early4 = BatchNormalization()(early4)
    early4 = Dropout(0.5)(early4)
    early4= GlobalAveragePooling2D()(early4)     
        
       
        
        
    x1 = layer_dict['block5_conv3'].output 
    x1= GlobalAveragePooling2D()(x1)
    #x1 = Flatten()(x1)
    x = keras.layers.concatenate([x1, early4, early3], axis=-1)  
    
    
    
    #x = Flatten()(x)
    #x = Dense(256, activation='relu')(x)
    
    x = Dense(2500, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = Dense(2, activation='softmax')(x)
    model = Model(input=base_model.input, output=x)
    #for layer in model.layers[:17]:
        #layer.trainable = True
    
     
    # for layer in model.layers[17:]:
    #     layer.trainable = True  
    #model.summary()
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #plot_model(model, to_file='vggmod19.png')
    print("[INFO] Model Compiled!")
    return model

def make_model():
    
#import pydot
    
    base_model = keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_tensor=None, input_shape=(300, 300, 3), pooling=None, classes=2)
    #base_model = keras.applications.resnet.ResNet50(include_top=False, weights='imagenet', input_shape=(32, 32, 3), pooling='max')
    layer_dict = dict([(layer.name, layer) for layer in base_model.layers])
    #base_model.summary()
    for layer in base_model.layers:
        layer.trainable = False
    for layer in base_model.layers[19:]:
        layer.trainable = True
    #base_model.summary()
    
    
        
    x1 = layer_dict['block5_pool'].output 
    x1= GlobalAveragePooling2D()(x1)
    
    #x = Flatten()(x)
    #x = Dense(256, activation='relu')(x)
    
    x = Dense(400, activation='relu')(x1)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = Dense(200, activation='relu')(x1)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = Dense(2, activation='softmax')(x)
    model = Model(input=base_model.input, output=x)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model    



def nn():
    
    
    inputb = Input(shape=(23,))
    
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
    x = Dense(classes, activation='tanh')(x)
    model = Model(input=base_model.input, output=x)

    #model.summary()
    #model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #plot_model(model, to_file='vggmod19.png')
    print("[INFO] Model Compiled!")
    return model

def make_mine(in_shape, tune, classes):
    
    input_img = (200,200,3)
    model = Sequential()
    
    model.add(Conv2D(32, (3,3), activation='relu', input_shape=input_img))
    model.add(MaxPooling2D(pool_size=(4,4)))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Conv2D(128, (3,3), activation='relu'))
    model.add(Conv2D(128, (3,3), activation='relu'))
    model.add(Conv2D(256, (3,3), activation='relu'))
    model.add(Conv2D(256, (3,3), activation='relu'))             
    model.add(Conv2D(512, (3,3), activation='relu'))  
    model.add(Conv2D(512, (3,3), activation='relu'))  
    model.add(GlobalAveragePooling2D())  

    model.add(Dense(512))  
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))   

    model.compile(loss='categorical_crossentropy',optimizer='SGD',metrics=['accuracy'])
    # model.summary()  

    return model    
              
              
              
#%%
import csv

  
print("[INFO] loading images from private data...")
data = []
labels = []

label_file = 'J:\\Python files\\SPECT_P2_600\\lablled\\dset.csv'

with open(label_file, newline='') as csvfile:
    data2 = list(csv.reader(csvfile))
    



imagePaths = sorted(list(paths.list_images('J:\\Python files\\SPECT_P2_600\\lablled\\')))
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
             label = 'A_cad'
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
    
    
#%%
import pandas as pd     

data_f = np.delete(data_f,[0,24,25,26,27,28],1)
# datax = np.delete(datax,0,0)



# def process_data (data):
        
#         columns = ['SEX','SAGE', 'BMI', 'known CAD', 'previous AMI', 'previous PCI', 'previous CABG', 'prev STROKE', 'diabetes', 'smoking', 'hypertension', 'dislipidemia', 'angiopathy', 'xna', 'FHCAD', 'PREV ETT NEG/POS', 'ASYMPTOMATIC', 'ATYPICAL SYMPTOMS', 'ANGINA LIKE','DYSPNOEA ON EXERTION', 'INCIDENT OF PRECORDIAL PAIN','RST ECG','MPI ischem']
#         data = pd.DataFrame(data)
#         data.columns = columns
        
#         categorical = ['SEX', 'known CAD', 'previous AMI', 'previous PCI', 'previous CABG', 'prev STROKE', 'diabetes', 'smoking', 'hypertension', 'dislipidemia', 'angiopathy', 'xna', 'FHCAD', 'PREV ETT NEG/POS', 'ASYMPTOMATIC', 'ATYPICAL SYMPTOMS', 'ANGINA LIKE','DYSPNOEA ON EXERTION', 'INCIDENT OF PRECORDIAL PAIN','RST ECG','MPI ischem']
#         zipBinarizer = LabelBinarizer().fit(data['SEX'])
        
#         for att in categorical:
#             data[att] = zipBinarizer.transform(data[att])
            
    

#%%
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
    
    
    # cnn = make_model()
    cnn = make_inceptionv3((200,200,3), 0, 2)
    # cnn = make_mine((200,200,3), 0, 2)
    
    # COMBINATION
    nn1 = nn()    
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
    
    # UNIVERSAL COMPILER
    
    #model3 = cnn
    
    

    
    
 
    
    print('[INFO] PREPARING FOLD: '+str(omega))
    
    # aug = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1) #rotation_range=20) #horizontal_flip=True, vertical_flip=True, fill_mode = 'nearest'
    # aug.fit(trainX)
    # history = model3.fit_generator(aug.flow(trainX, trainY,batch_size=32), epochs=25, steps_per_epoch=len(trainX)//32)
    
    # CNN ONLY
   # history = model3.fit(trainX, trainY,epochs=7, batch_size=64)
    
    # COMBINATION
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
        #model3.summary()

    
    omega = omega + 1



    # COMBINATION
    score = model3.evaluate([testX,testX22],testY)
    
    # CNN ONLY
    # score = model3.evaluate(testX,testY)


    
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