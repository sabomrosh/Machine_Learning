import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Dropout,GlobalAveragePooling2D, AveragePooling2D
import inception_v3 as inception
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
#from keras.utils import plot_model
#import matplotlib.pyplot as plt
#from keras.utils.dot_utils import Grapher
#from IPython.display import SVG
#from keras.utils.visualize_util import model_to_dot
#from keras.utils.vis_utils import model_to_dot
#grapher=Grapher()
N_CLASSES = 2
IMSIZE = (299, 299)
batchsize=16
nbepoch1=30
nbepoch2=50
#a=41143//batchsize
#b=10286//batchsize
# Training directory
train_dir = '/home/cmpt726/eye/train'
# Testing directory
test_dir = '/home/cmpt726/eye/test'

base_model = inception.InceptionV3(weights='imagenet',include_top=False)
print 'Loaded Inception model'
#for layer in base_model.layers:
    #layer.trainable = False
##x = Dense(256, activation='elu')(base_model.get_layer('avg_pool').output)
#x = Dense(256, activation='elu')(base_model.get_layer('flatten').output)
#x = AveragePooling2D((8, 8), strides=(8, 8), name='avg_pool')(base_model.get_layer('mixed10').output)
#x = Flatten(name='flatten')(x)
x=base_model.output
x=GlobalAveragePooling2D()(x)
x = Dense(1024, activation='elu')(x)
#x = Dense(256, activation='elu')(x)
#x = Dropout(0.5)(x)
predictions = Dense(N_CLASSES, activation='softmax', name='predictions')(x)

model = Model(input=base_model.input, output=predictions)
#print 'Loaded Inception model'
for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='rmsprop', loss='binary_crossentropy',metrics=['accuracy'])

# Data generators for feeding training/testing images to the model.
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,zoom_range=0.2)
train_generator = train_datagen.flow_from_directory(
        train_dir, 
        target_size=IMSIZE,  
        batch_size=batchsize,
        class_mode='categorical')

test_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,zoom_range=0.2)
test_generator = test_datagen.flow_from_directory(
        test_dir,  
        target_size=IMSIZE,  
        batch_size=batchsize,
        class_mode='categorical')

model.fit_generator(
        train_generator,
        samples_per_epoch=16,
        nb_epoch= nbepoch1,
        validation_data=test_generator,
        nb_val_samples=16,
        verbose=2)
model.save_weights('kaggledatabase_pretrain.h5')  

#finetuning

for i, layer in enumerate(model.layers):
   print(i, layer.name)

for layer in model.layers[:172]:
   layer.trainable = False
for layer in model.layers[172:]:
   layer.trainable = True


from keras.optimizers import SGD
sgd=SGD(lr=0.0001, momentum=0.9)
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])


checkpoint = ModelCheckpoint('model.h5', monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

history = model.fit_generator(
        train_generator,
        samples_per_epoch=16,
        nb_epoch= nbepoch2,
        validation_data=test_generator,
        verbose=2,
        nb_val_samples=16,
        callbacks=callbacks_list)
model.save_weights('kaggledatabase_pretrain2.h5')  

# Show some debug output
print (model.summary())

print 'Trainable weights'
print model.trainable_weights


#load weights
model.load_weights("model.h5")
#save.history.history("history.xlsx")
print(history.history)
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#precision(test_generator)
#recall(test_generator)

#img_path = '/home/cmpt726/eye/test/patient/30_left.jpeg'
#img = image.load_img(img_path, target_size=IMSIZE)
#x = image.img_to_array(img)
#x = np.expand_dims(x, axis=0)

#x = inception.preprocess_input(x)

#preds = model.predict(x)
#print('Predicted:', preds)