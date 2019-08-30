import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda-9.0/lib64:/usr/local/cuda/extras/CUPTI/lib64"
os.environ["LD_LIBRARY_PATH"]
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from keras import backend as keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
# from DeepLearningDataset import dataset
# from data import *
import h5py
import numpy as np
from scipy import misc

import model.u_net as unet
import cv2
import matplotlib.pyplot as plt

import keras


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, path, partition, batch_size=32, batch_per_epoch=None, dim=(32,32,32), final_size=(512,512), n_channels=1,
         n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.final_size = final_size
        self.batch_size = batch_size
        self.partition = partition
        self.batch_per_epoch = batch_per_epoch
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.H5 = h5py.File(path,'r')
        self.Xs = self.H5.get("{}/images".format(self.partition))
        self.Ys = self.H5.get("{}/labels".format(self.partition))
        if self.batch_per_epoch > self.Xs.shape[0]:
            self.batch_per_epoch = self.Xs.shape[0] - 1
        self.on_epoch_end()
        self.X = None
        self.Y = None

    def __len__(self):
        'Denotes the number of batches per epoch'
        if not self.batch_per_epoch:
            return int(np.floor(self.Xs.shape[0] / self.batch_size))
        else:
            return self.batch_per_epoch

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        # indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        if type(self.X) != np.ndarray:
            self.__data_generation()

        return self.X[index], self.Y[index]
        # Find list of IDs
        # list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # # Generate data
        # X, y = self.__data_generation()
        #
        # return X, y

    def on_epoch_end(self):
        'Updates random index after each epoch'
        self.index = np.random.choice(self.Xs.shape[0]-(self.batch_per_epoch+1), 1)[0]
        self.X = None
        self.Y = None

    def __data_generation(self):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization

        X = self.Xs[self.index:self.index+self.batch_per_epoch+1]
        Y = self.Ys[self.index:self.index+self.batch_per_epoch+1][:,:,:,1]

        self.X = []
        self.Y = []

        for i in range(X.shape[0]):
            x = misc.imresize(X[i], self.final_size)
            y = misc.imresize(Y[i], self.final_size)
            y = np.expand_dims(y, axis=2)
            self.X.append([x])
            self.Y.append([y])

        self.X = np.array(self.X)
        self.Y = np.array(self.Y)


        # y = cv2.resize(y, self.final_size)

        # return X, y

class MyUnet:
    def __init__(self, img_shape=(512,512,3), nb_class=1, batch_size=1, batch_per_epoch=100, shuffle=True, epochs=100):
        self.img_shape = img_shape
        self.nb_class = nb_class
        self.epochs = epochs
        self.batch_per_epoch = batch_per_epoch
        self.batch_size = batch_size
        self.path= "/home/lblondel/Documents/Harvard/ExtavourLab/projects/Project_Parhyale/Microscopy/JaneliaLightSheet/Bro1/testingSets.h5"
        self.params = {'dim': (img_shape[0], img_shape[1]),
                  'batch_size': batch_size,
        		  'batch_per_epoch':batch_per_epoch,
                  'n_classes': nb_class,
                  'n_channels': img_shape[2],
                  'shuffle': shuffle}
        self.set_generators()

    def get_unet(self, weights=None):
        model = unet.get_unet_512(input_shape=self.img_shape, num_classes=self.nb_class)
        if weights:
            model.load_weights(weights)
        print(model.summary())
        return model

    def set_generators(self):
        self.training_generator = DataGenerator(self.path, "train", **self.params)
        self.test_generator = DataGenerator(self.path, "test", **self.params)
        #self.validation_generator = DataGenerator(self.path, "validate", **self.params)

    def train(self, weights=None):
        if weights:
            model = self.get_unet(weights=weights)
        else:
            model = self.get_unet()

        callbacks = [EarlyStopping(monitor='val_loss',
                           patience=8,
                           verbose=1,
                           min_delta=1e-4),
             ReduceLROnPlateau(monitor='val_loss',
                               factor=0.1,
                               patience=4,
                               verbose=1,
                               epsilon=1e-4),
             ModelCheckpoint(monitor='val_loss',
                             filepath='best_weights.hdf5',
                             save_best_only=True,
                             save_weights_only=True),
             TensorBoard(log_dir='./logs', 
histogram_freq=1, 
batch_size=1, 
write_graph=True, 
write_grads=False, 
write_images=True, 
embeddings_freq=0, 
embeddings_layer_names=None, 
embeddings_metadata=None, 
embeddings_data=None)]

        history = model.fit_generator(generator=self.training_generator,
                    steps_per_epoch=self.batch_per_epoch,
                    epochs=self.epochs,
                    callbacks=callbacks,
                    validation_data=self.test_generator,
                    validation_steps=self.batch_per_epoch)
        model.save("finalModel.hdf5")
        f = open("history.tsv")
        keys = history.history.keys()
        l = "\t".join(keys)
        f.write(l + '\n')
        for epoch in range(len(history.history[keys[0]])):
            l = []
            for k in keys:
                l.append(history.history[k][epoch])
            l = "\t".join(l) + '\n'
            f.write(l)
        f.close()
        self.predict()

    def predict(self, n_image=10):
        if not os.path.isfile('best_weights.hdf5'):
            print("Error no model found ! start by training a model !")
            return None
        model = self.get_unet(weights='best_weights.hdf5')
        # model = load_model()
        for i in range(n_image):
            imgs_test, imgs_labels = self.test_generator[i]
            imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
            np.save('results/predicted_{}_imgs_mask_test.npy', imgs_mask_test)
            for j in range(imgs_mask_test.shape[0]):
                # for c in range(imgs_mask_test.shape[-1]):
                img = imgs_mask_test[j][:,:,0]
                print(img.shape)
                img = misc.imsave("results/{}.jpg".format(i),img)

    def predictAndShowFive(self):
        if not os.path.isfile('best_weights.hdf5'):
            print("Error no model found ! start by training a model !")
            return None
        model = self.get_unet(weights='best_weights.hdf5')
        self.test_generator.on_epoch_end()
        for i in range(5):
            imgs_test, imgs_labels = self.test_generator[i]
            imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
            fig = plt.figure()
            ax1 = fig.add_subplot(131)
            ax1.imshow(imgs_test[0])
            ax1.set_title("Orginal Image")
            ax2 = fig.add_subplot(132, sharex=ax1, sharey=ax1)
            ax2.imshow(imgs_labels[0][:,:,0])
            ax2.set_title("Orginal Image Labels")
            ax3 = fig.add_subplot(133, sharex=ax1, sharey=ax1)
            ax3.imshow(imgs_mask_test[0][:,:,0])
            ax3.set_title("Predicted Labels")
        plt.show()

    def Evaluate(self):
        scores = []
        if not os.path.isfile('best_weights.hdf5'):
            print("Error no model found ! start by training a model !")
            return None
        model = self.get_unet(weights='best_weights.hdf5')
        self.test_generator.on_epoch_end()
        scores = model.evaluate_generator(self.test_generator, steps=len(self.test_generator), verbose=1)
        print(scores)
            
            

if __name__=="__main__":
    Unet = MyUnet(epochs=100, batch_per_epoch=2500)
    #Unet.train()
    #Unet.train('best_weights.hdf5')
    #Unet.predict(n_image=100)
    #Unet.Evaluate()
    Unet.predict(n_image=100)
    #Unet.predictAndShowFive()
