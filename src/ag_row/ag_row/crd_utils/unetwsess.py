# -*- coding:utf-8 -*-

import os 
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "0"	

#print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras import backend as K
import cv2
#from data import *
#import tensorflow.experimental.numpy as np
import numpy as np
from collections import defaultdict
import math
from PIL import Image

class myUnet(object):
    def __init__(self, img_rows=512, img_cols=512):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.eps1 = 4
        self.eps2 = 3
        self.htres = 100

    def load_data(self):
        mydata = dataProcess(self.img_rows, self.img_cols)
        imgs_train, imgs_mask_train = mydata.load_train_data()
        imgs_test = mydata.load_test_data()
        imgs_mask_test = mydata.load_test_labels()
        imgs_val, imgs_mask_val = mydata.load_val_data()
        return imgs_train, imgs_mask_train, imgs_test, imgs_mask_test, imgs_val, imgs_mask_val

    def tensor_to_image(self, tensor):
        tensor = tensor*255
        tensor = np.array(tensor, dtype=np.uint8)
        if np.ndim(tensor)>3:
            assert tensor.shape[0] == 1
            tensor = tensor[0]
        return PIL.Image.fromarray(tensor)

    def iou(self, y_true, y_pred):
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])
        intersection = tf.reduce_sum(tf.cast(y_true, tf.float32) * tf.cast(y_pred, tf.float32))
        score = (intersection + 1.) / (tf.reduce_sum(tf.cast(y_true, tf.float32)) + tf.reduce_sum(tf.cast(y_pred, tf.float32)) - intersection + 1.)
        return score

    def get_unet(self):
        inputs = Input((self.img_rows, self.img_cols, 3))

        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        # print(conv1)
        print ("conv1 shape:", conv1.shape)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
        print ("conv1 shape:", conv1.shape)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        print ("pool1 shape:", pool1.shape)

        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
        print ("conv2 shape:", conv2.shape)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
        print ("conv2 shape:", conv2.shape)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        print ("pool2 shape:", pool2.shape)

        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
        print ("conv3 shape:", conv3.shape)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
        print ("conv3 shape:", conv3.shape)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        print ("pool3 shape:", pool3.shape)

        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(drop5))
        merge6 = concatenate([drop4, up6],axis=3)
        print(up6)
        print(merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
        print(conv6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
        print(conv6)
        up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv6))
        merge7 = concatenate([conv3, up7],axis=3)
        print(up7)
        print(merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
        print(conv7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
        print(conv7)
        up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv7))
        merge8 = concatenate([conv2, up8],axis=3)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

        up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
            UpSampling2D(size=(2, 2))(conv8))
        merge9 = concatenate([conv1, up9], axis=3)
        print(up9)
        print(merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
        print(conv9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        print(conv9)
        conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
        print ("conv9 shape:", conv9.shape)

        conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
        print(conv10)
        model = Model(inputs, conv10)

        weights = [1,2]
        #model.compile(optimizer=Adam(lr=1e-4), loss= self.weightedLoss(tf.keras.losses.BinaryCrossentropy(from_logits=True), weights), metrics=['accuracy', self.iou, self.loss_angle], run_eagerly=True)
        #model.compile(optimizer=Adam(lr=1e-4), loss=BinaryFocalLoss(gamma=2), metrics=['accuracy', self.iou, self.loss_angle], run_eagerly=True)
        #model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy', self.iou], run_eagerly=True)
        #model.compile(optimizer=Adam(lr=1e-4), loss=self.loss_angle, metrics=[tf.keras.metrics.MeanIoU(num_classes=2)])
        model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy', self.iou], run_eagerly=True)


        return model


    def save_img(self):
        print("array to image")
        imgs = np.load('./results/imgs_mask_test.npy')
        piclist = []
        for line in open("./results/pic.txt"):
            line = line.strip()
            picname = line.split('/')[-1]
            piclist.append(picname)
        for i in range(imgs.shape[0]):
            path = "./results/" + piclist[i]
            img = imgs[i]
            img = array_to_img(img)
            img.save(path)
            cv_pic = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            cv_pic = cv2.resize(cv_pic,(512,512),interpolation=cv2.INTER_CUBIC)
            binary, cv_save = cv2.threshold(cv_pic, 127, 255, cv2.THRESH_BINARY)
            cv2.imwrite(path, cv_save)

if __name__ == '__main__':
    myunet = myUnet()
    model = myunet.get_unet()
    model.summary()
    #plot_model(model, to_file='model.png')
    myunet.train()
    myunet.save_img()
