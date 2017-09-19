from __future__ import print_function
import h5py
import sys
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers.merge import Concatenate
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D, MaxPooling3D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.models import model_from_json

from keras import losses
from keras.utils import to_categorical
from keras.models import load_model
import keras.backend as K
import os
import matplotlib.pyplot as plt
import argparse
from sklearn.metrics import f1_score,recall_score,confusion_matrix
import scipy.stats as st

import numpy as np
np.random.seed(1337) 
import tensorflow as tf
tf.set_random_seed(2016)

def log(log, logger = None):
    if logger is not None:
        logger.write(log + "\n")
    
    print(log)

def unweighted_acc(predictions, one_hot_vector_label, original_n_class):
    print("prediction shape:", predictions.shape)
    print("label shape:", one_hot_vector_label.shape)
    
    #for debugging
    #print(predictions[0:10,])
    #print(one_hot_vector_label[0:10,])
    
    #dropping column for validity
    predictions = np.delete(predictions, original_n_class, 1 )
    one_hot_vector_label = np.delete(one_hot_vector_label, original_n_class, 1 )
    
    #for debugging
    #print("after dropping valdidty\nprediction shape:", predictions.shape)
    #print("label shape:", one_hot_vector_label.shape)
    #print(predictions[0:10,])
    #print(one_hot_vector_label[0:10,])
    
    labels = np.argmax(one_hot_vector_label,1)
    pred = np.argmax(predictions,1)
    
    score = recall_score(labels, pred, average='macro')
    cm = confusion_matrix(labels, pred)
    prob_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    #return unweighted recall, confusion matrix, and it's probablistic distribution
    return score, cm, prob_cm

def compose_idx(args_train_idx, args_test_idx, args_valid_idx):
    train_idx = []
    test_idx = []
    valid_idx = []
    kf_idx = []
    if args_train_idx:
        if ',' in args_train_idx:
            train_idx = args_train_idx.split(',')
        elif ':' in args_train_idx:
            indice = args_train_idx.split(':')
            for idx in range(int(indice[0]), int(indice[1]) + 1, +1):
                train_idx.append(idx)
        else:
            train_idx = args_train_idx.split(",")

    if args_test_idx:
        if ',' in args_test_idx:
            test_idx = args_test_idx.split(',')
        elif ':' in args_test_idx:
            indice = args_test_idx.split(':')
            for idx in range(int(indice[0]), int(indice[1]) + 1, +1):
                test_idx.append(idx)
        else:
            test_idx = args_test_idx.split(",")

    if args_valid_idx:
        if ',' in args_valid_idx:
            valid_idx = args_valid_idx.split(',')
        elif ':' in args_valid_idx:
            indice = args_valid_idx.split(':')
            for idx in range(int(indice[0]), int(indice[1]) + 1, +1):
                valid_idx.append(idx)
        else:
            valid_idx = args_valid_idx.split(",")
    
    return train_idx, test_idx, valid_idx

def building_generator(noise_source_len, depth_G, n_kernels_G, kernel_resolution_G, initial_n_row, initial_n_col, prefix_name = "a"):
    noise = Input(shape=(noise_source_len,))
    
    #generator
    #first layer: fully connected layer
    next_G = Dense(n_kernels_G[0] * initial_n_row * initial_n_col, activation="relu", input_dim=noise_source_len)(noise)
    next_G = Reshape((initial_n_row, initial_n_col, n_kernels_G[0]))(next_G)
    next_G = BatchNormalization(momentum=0.8)(next_G)
        
    #middle layers: CNN
    for idx in range(1, depth_G - 1):
        next_G = UpSampling2D()(next_G)
        next_G = Conv2D(n_kernels_G[idx], kernel_size=kernel_resolution_G[idx], padding="same")(next_G)
        next_G = Activation("relu")(next_G)
        next_G = BatchNormalization(momentum=0.8)(next_G)
        
    #last layer: CNN without batch normalisation, last n_kernel is always 1(n_channel), regardless of setting
    next_G = Conv2D(1, kernel_size=kernel_resolution_G[-1], padding="same")(next_G)
    next_G = Activation("tanh")(next_G)   
    
    G = Model(inputs=noise, outputs=next_G)
    
    n_layers = len(G.layers)
    print("total layers: ", n_layers )
    
    for layer in G.layers:  
        layer.name =  prefix_name + "_" + layer.name
        
    return G

def building_discriminator(depth_D, n_kernels_D, kernel_resolution_D, pooling_resolution_D, n_row, n_col, dropout, prefix_name = "audio"):
    
    layer_idx = 0
    
    img_shape = (n_row, n_col, 1)
    img = Input(shape=img_shape, name = prefix_name + "_input_" + str(layer_idx) )
    layer_idx +=1
    
    #first layer
    next_D = Conv2D(n_kernels_D[0], kernel_size=kernel_resolution_D[0], strides=2, input_shape=img_shape, padding="same",name = prefix_name + "_cnn_" + str(layer_idx) )(img)
    layer_idx +=1
    next_D = LeakyReLU(alpha=0.2,name = prefix_name + "_lrelu_" + str(layer_idx))(next_D)
    layer_idx +=1
    if dropout:
        next_D = Dropout(dropout,name = prefix_name + "_dropout_" + str(layer_idx))(next_D)
        layer_idx +=1
        
    next_D = BatchNormalization(momentum=0.8,name = prefix_name + "_batchnorm_" + str(layer_idx))(next_D)
    layer_idx +=1
    
    if len(pooling_resolution_D) > 0:
        next_D = MaxPooling2D(pool_size = pooling_resolution_D[0],name = prefix_name + "_maxpooling_" + str(layer_idx))(next_D)
        layer_idx +=1
    
    #middle layers
    for idx in range(depth_D):
        next_D = Conv2D(n_kernels_D[idx], kernel_size=kernel_resolution_D[idx], padding="same",name = prefix_name + "_cnn_" + str(layer_idx))(next_D)
        layer_idx +=1
        next_D = LeakyReLU(alpha=0.2,name = prefix_name + "_lrelu_" + str(layer_idx))(next_D)
        layer_idx +=1
        
        if dropout:
            next_D = Dropout(dropout,name = prefix_name + "_dropout_" + str(layer_idx))(next_D)
            layer_idx +=1
        
        next_D = BatchNormalization(momentum=0.8,name = prefix_name + "_batchnorm_" + str(layer_idx))(next_D)
        layer_idx +=1
    
        if len(pooling_resolution_D) > 0:
            next_D = MaxPooling2D(pool_size = pooling_resolution_D[idx],name = prefix_name + "_maxpooling_" + str(layer_idx))(next_D)

    return img, next_D

def str_to_int(str_array):
    return [int(i) for i in str_array]

def building_custom_gan(args, loss_weights = [0.5,0.5], d_lr = 0.0002, g_lr = 0.0004, n_class = 4, fake_real_loss = 'mean_squared_error', noise_source_len = 100):
    #model configuration
    
    generators = []
    discriminators = []
    
    #a_n_row = 100, a_n_col = 128,  v_n_row = 48, v_n_col = 48,
    n_row = str_to_int(args.r_nrow.split(";"))
    n_col = str_to_int(args.r_ncol.split(";"))
        
    modalities = []
    for modality in args.modality.split(";"):
        modalities.append(modality)
    
    depth_G = str_to_int(args.depth_G.split(";"))
    depth_D = str_to_int(args.depth_D.split(";"))
    
    #parsing kernels
    n_kernels_G = []
    for mv in args.n_kernels_G.split(";"):
        n_kernels_G.append(str_to_int(mv.split(",")))

    n_kernels_D = []
    for mv in args.n_kernels_D.split(";"):
        n_kernels_D.append(str_to_int(mv.split(",")))

    #parsing resolutions of convolutional layers
    
    crows = []
    ccols = []
    for mv in args.cnn_n_row_D.split(";"):        
        crows.append(str_to_int(mv.split(",")))
    for mv in args.cnn_n_col_D.split(";"):        
        ccols.append(str_to_int(mv.split(",")))
    
    kernel_resolution_D = []
    for i in range(len(modalities)):
        if len(crows[i]) == len(ccols[i]) and len(ccols[i]) == depth_D[i]:
            kernel_r_c = []
            for j in range(len(crows[i])):
                kernel_r_c.append((crows[i][j], ccols[i][j]))
            kernel_resolution_D.append( kernel_r_c )
        else:
            print("check structure of CNN D!")
            raise ValueError("parsing argument error!")
    
    #parsing pooling layers
    pooling_resolution_D = []
    
    if args.pool_n_row_D and args.pool_n_col_D:
        prows = []
        pcols = []
        for mv in args.pool_n_row_D.split(";"):        
            prows.append(str_to_int(mv.split(",")))
        for mv in args.pool_n_col_D.split(";"):        
            pcols.append(str_to_int(mv.split(",")))
        
        for i in range(len(modalities)):
            if len(crows[i]) == len(ccols[i]) and len(ccols[i]) == len(prows[i]) and len(prows[i]) == len(pcols[i]) and len(pcols[i]) == depth_D[i]:
                kernel_r_c = []
                for j in range(len(prows[i])):
                    kernel_r_c.append((prows[i][j], pcols[i][j]))
                pooling_resolution_D.append( kernel_r_c )
            else:
                print("crows ccols ccols prows prows pcols pcols depth of D: ", len(crows[i]), len(ccols[i]),len(ccols[i]), len(prows[i]),len(prows[i]), len(pcols[i]),len(pcols[i]), depth_D[i])
                print("check structure of pooling of CNN D!")
                raise ValueError("parsing argument error!")
    
    crows = []
    ccols = []
    for mv in args.cnn_n_row_G.split(";"):        
        crows.append(str_to_int(mv.split(",")))
    for mv in args.cnn_n_col_G.split(";"):        
        ccols.append(str_to_int(mv.split(",")))
    
    kernel_resolution_G = []
    for i in range(len(modalities)):
        if len(crows[i]) == len(ccols[i]) and len(ccols[i]) == depth_G[i]:
            kernel_r_c = []
            for j in range(len(crows[i])):
                kernel_r_c.append((crows[i][j], ccols[i][j]))
            kernel_resolution_G.append( kernel_r_c )
        else:
            print("check structure of CNN G!")
            raise ValueError("parsing argument error!")
        
  
    
    
    #default configuration, similar to MNIST DCGAN setup
    if args.default_G_D:
        #TODO: this is too big: now, I'm reshaping it by cut-off.
        initial_a_n_row = n_row[0] / 4
        initial_a_n_col = n_col[0] / 4

        initial_v_n_row = n_row[1] / 4
        initial_v_n_col = n_col[1] / 4

        #build generator and discriminator
        noise = Input(shape=(noise_source_len,))
        next_a_G = Dense(128 * initial_a_n_row * initial_a_n_col, activation="relu", input_dim=noise_source_len)(noise)
        next_a_G = Reshape((initial_a_n_row, initial_a_n_col, 128))(next_a_G)
        next_a_G = BatchNormalization(momentum=0.8)(next_a_G)
        next_a_G = UpSampling2D()(next_a_G)
        next_a_G = Conv2D(128, kernel_size=3, padding="same")(next_a_G)
        next_a_G = Activation("relu")(next_a_G)
        next_a_G = BatchNormalization(momentum=0.8)(next_a_G)
        next_a_G = UpSampling2D()(next_a_G)
        next_a_G = Conv2D(64, kernel_size=3, padding="same")(next_a_G)
        next_a_G = Activation("relu")(next_a_G)
        next_a_G = BatchNormalization(momentum=0.8)(next_a_G)
        next_a_G = Conv2D(1, kernel_size=3, padding="same")(next_a_G)
        next_a_G = Activation("tanh")(next_a_G)   
        a_G = Model(inputs=noise, outputs=next_a_G)
        generators.append(a_G)
        
        noise = Input(shape=(noise_source_len,))
        next_v_G = Dense(128 * initial_v_n_row * initial_v_n_col, activation="relu", input_dim=noise_source_len)(noise)
        next_v_G = Reshape((initial_v_n_row, initial_v_n_col, 128))(next_v_G)
        next_v_G = BatchNormalization(momentum=0.8)(next_v_G)
        next_v_G = UpSampling2D()(next_v_G)
        next_v_G = Conv2D(128, kernel_size=3, padding="same")(next_v_G)
        next_v_G = Activation("relu")(next_v_G)
        next_v_G = BatchNormalization(momentum=0.8)(next_v_G)
        next_v_G = UpSampling2D()(next_v_G)
        next_v_G = Conv2D(64, kernel_size=3, padding="same")(next_v_G)
        next_v_G = Activation("relu")(next_v_G)
        next_v_G = BatchNormalization(momentum=0.8)(next_v_G)
        next_v_G = Conv2D(1, kernel_size=3, padding="same")(next_v_G)
        next_v_G = Activation("tanh")(next_v_G)   
        v_G = Model(inputs=noise, outputs=next_v_G)
        generators.append(v_G)
        
        a_img_shape = (a_n_row, a_n_col, 1)
        a_img = Input(shape=a_img_shape)
        next_a_D = Conv2D(32, kernel_size=3, strides=2, input_shape=img_shape, padding="same")(a_img)
        next_a_D = LeakyReLU(alpha=0.2)(next_a_D)
        next_a_D = Dropout(0.25)(next_a_D)
        next_a_D = Conv2D(64, kernel_size=3, strides=2, padding="same")(next_a_D)
        next_a_D = ZeroPadding2D(padding=((0,1),(0,1)))(next_a_D)
        next_a_D = LeakyReLU(alpha=0.2)(next_a_D)
        next_a_D = Dropout(0.25)(next_a_D)
        next_a_D = BatchNormalization(momentum=0.8)(next_a_D)
        next_a_D = Conv2D(128, kernel_size=3, strides=2, padding="same")(next_a_D)
        next_a_D = LeakyReLU(alpha=0.2)(next_a_D)
        next_a_D = Dropout(0.25)(next_a_D)
        next_a_D = BatchNormalization(momentum=0.8)(next_a_D)
        next_a_D = Conv2D(256, kernel_size=3, strides=1, padding="same")(next_a_D)
        next_a_D = LeakyReLU(alpha=0.2)(next_a_D)
        next_a_D = Dropout(0.25)(next_a_D)
        next_a_D = MaxPooling2D(pool_size = (2,2))(next_a_D)

        v_img_shape = (v_n_row, v_n_col, 1)
        v_img = Input(shape=v_img_shape)
        next_v_D = Conv2D(32, kernel_size=3, strides=2, input_shape=img_shape, padding="same")(v_img)
        next_v_D = LeakyReLU(alpha=0.2)(next_v_D)
        next_v_D = Dropout(0.25)(next_v_D)
        next_v_D = Conv2D(64, kernel_size=3, strides=2, padding="same")(next_v_D)
        next_v_D = ZeroPadding2D(padding=((0,1),(0,1)))(next_v_D)
        next_v_D = LeakyReLU(alpha=0.2)(next_v_D)
        next_v_D = Dropout(0.25)(next_v_D)
        next_v_D = BatchNormalization(momentum=0.8)(next_v_D)
        next_v_D = Conv2D(128, kernel_size=3, strides=2, padding="same")(next_v_D)
        next_v_D = LeakyReLU(alpha=0.2)(next_v_D)
        next_v_D = Dropout(0.25)(next_v_D)
        next_v_D = BatchNormalization(momentum=0.8)(next_v_D)
        next_v_D = Conv2D(256, kernel_size=3, strides=1, padding="same")(next_v_D)
        next_v_D = LeakyReLU(alpha=0.2)(next_v_D)
        next_v_D = Dropout(0.25)(next_v_D)
        next_v_D = MaxPooling2D(pool_size = (2,2))(next_v_D)
        
        next_a_D = Flatten()(next_a_D)
        next_v_D = Flatten()(next_v_D)
        next_a_v_D = Concatenate(name="a_v_concate")[next_a_D, next_v_D]
        
        next_a_v_D = Dense(512, activation="relu")(next_a_v_D)
        next_a_v_D = Dense(512, activation="relu")(next_a_v_D)
    else:
        
        inputs_D = []
        for i in range(len(modalities)):
            initial_n_row = n_row[i] / depth_G[i]
            initial_n_col = n_col[i] / depth_G[i]
            
            #build a generator
            G = building_generator(noise_source_len, depth_G[i], n_kernels_G[i], kernel_resolution_G[i], initial_n_row, initial_n_col, modalities[i])
            
            #build a discriminator
            img, D = building_discriminator(depth_D[i], n_kernels_D[i], kernel_resolution_D[i], pooling_resolution_D[i], n_row[i], n_col[i], args.dropout, modalities[i])
            D = Flatten()(D)
            
            inputs_D.append(img)
            generators.append(G)
            discriminators.append(D)
        
        #merging discriminators
        if len(modalities) > 1:
            next_a_v_D = Concatenate(name="a_v_concate")(discriminators)
        else:
            next_a_v_D = discriminators[0]
                
        #fuly connected layers for merged(or single) D
        for i in range(args.depth_a_v_D):
            next_a_v_D = Dense(args.n_node, activation="relu")(next_a_v_D)
  
    #sigmoid for validity output[real/fake]        
    valid = Dense(1, activation="sigmoid")(next_a_v_D)
    
    #unsupervised mode does not need softmax output node for classification
    if args.unsupervised:        
        merged_discriminator = Model(inputs_D, valid)
    else:
        label = Dense(n_class + 1, activation="softmax")(next_a_v_D)
        merged_discriminator = Model(inputs_D, [valid, label])

    return SGAN(args, modalities, generators, merged_discriminator, loss_weights, d_lr, g_lr, n_class, fake_real_loss, noise_source_len, unsupervised = args.unsupervised)

def reshape_feat_lab(X_train,X_test,X_valid, resize = True, img_row = 28, img_col = 28):
    
    original_n_rows = X_train.shape[1] * X_train.shape[3]
    original_n_cols = X_train.shape[4]
    #remove all rows and cols beyond given dimensions for rows and cols
    slices_to_remove_rows = [x for x in range(img_row, original_n_rows,1)]
    slices_to_remove_cols = [x for x in range(img_col, original_n_cols,1)]
    
    X_train = X_train.reshape((X_train.shape[0], original_n_rows, original_n_cols))
    if resize:
        X_train = np.delete(X_train, slices_to_remove_rows, 1 )
        X_train = np.delete(X_train, slices_to_remove_cols, 2 )
    X_train = np.expand_dims(X_train, axis=3)

    X_test = X_test.reshape((X_test.shape[0], original_n_rows, original_n_cols))
    if resize:
        X_test = np.delete(X_test, slices_to_remove_rows, 1 )
        X_test = np.delete(X_test, slices_to_remove_cols, 2 )
    X_test = np.expand_dims(X_test, axis=3)

    X_valid = X_valid.reshape((X_valid.shape[0],original_n_rows, original_n_cols))
    if resize:
        X_valid = np.delete(X_valid, slices_to_remove_rows, 1 )
        X_valid = np.delete(X_valid, slices_to_remove_cols, 2 )
    X_valid = np.expand_dims(X_valid, axis=3)

    print("train feature shape, ", X_train.shape)
    print("test feature shape, ", X_test.shape)
    print("valid feature shape, ", X_valid.shape)

    return X_train,X_test,X_valid


class SGAN():
    def __init__(self, args, modalities, generators, discriminator, loss_weights = [0.5,0.5], d_lr = 0.0002, g_lr = 0.0004, n_class = 10, fake_real_loss = 'mean_squared_error', noise_source_len = 100, unsupervised = False):
        
        self.num_classes = n_class
        self.earlystopping = []
        self.unsupervised = unsupervised
        self.modality = modalities
        
        d_optimizer = Adam(d_lr, 0.5)
        g_optimizer = Adam(g_lr, 0.5)
        c_optimizer = Adam(g_lr, 0.5)
        
        self.discriminator = discriminator
        
        if self.unsupervised:
            d_loss = fake_real_loss
            loss_weights = [1.0]
        else:
            d_loss = [fake_real_loss, 'categorical_crossentropy']
            
        self.discriminator.compile(loss = d_loss, 
            loss_weights=loss_weights,
            optimizer=d_optimizer,
            metrics=['accuracy'])
        
        print("Current configuration for discriminator")
        self.discriminator.summary()

        # Build and compile the generator
        self.generators = generators
        
        for generator in self.generators:
            generator.compile(loss=[fake_real_loss], optimizer=g_optimizer)
            print("Current configuration for a_generator")
            generator.summary()
            
        #load pretrain_model the generator
        if args.load_model_G:
            
            frozen_layer_list_G = []
            unload_layer_list_G = []
            
            if args.frozen_G:
                for frozen_G in args.frozen_G.split(";"):
                    frozen_layer_list_G.append(self.parse_pretrain_args(frozen_G))
            else:
                for mod in self.modality:
                    frozen_layer_list_G.append(self.parse_pretrain_args(None)) 
                    
            if args.unloaded_G:
                for unloaded_G in args.unloaded_G.split(";"):
                    unload_layer_list_G.append(self.parse_pretrain_args(unloaded_G))
            else:
                for mod in self.modality:
                    unload_layer_list_G.append(self.parse_pretrain_args(None)) 
                    
            load_model_G = args.load_model_G.split(";")
            for i in range(len(self.modality)):
                print("Loading pretrained models for G")
                self.generators[i] = self.load_pretrained_model(self.generators[i], load_model_G[i], frozen_layer_list_G[i], unload_layer_list_G[i])
                
                    
        #load pretrain_model the discriminator    
        if args.load_model_D:
            
            frozen_layer_list_D = []
            unload_layer_list_D = []
            load_model_D = args.load_model_D.split(";")
            
            if args.frozen_D:
                for frozen_D in args.frozen_D.split(";"):
                    frozen_layer_list_D.append(self.parse_pretrain_args(frozen_D))
            else:
                for i in range(len(load_model_D)):
                    frozen_layer_list_D.append(self.parse_pretrain_args(None)) 
                    
            if args.unloaded_D:
                for unloaded_D in args.unloaded_D.split(";"):
                    unload_layer_list_D.append(self.parse_pretrain_args(unloaded_D))
            else:
                for i in range(len(load_model_D)):
                    unload_layer_list_D.append(self.parse_pretrain_args(None)) 
                    
            print("Loading pretrained models for D")
            '''
            note that a multimodal D model is always optimised with multi-modal inputs, we neither optimise nor save their weights separately
            But we can still save single modality models: audio and video separately.
            For this case, we load layers' weights weights, hence, we can load different single modality models several times.
            '''
            for i in range(len(load_model_D)):
                self.discriminator = self.load_pretrained_model(self.discriminator, load_model_D[i], frozen_layer_list_D[i], unload_layer_list_D[i])            
            
                         
        print("Aftering loading pretrained models")            
        for i in range(len(self.modality)):
            self.generators[i].summary()
        self.discriminator.summary()
            
        # The generator takes noise as input and generates imgs
        noise = Input(shape=(noise_source_len,))
        
        inputs_D = []
        for generator in self.generators:
            inputs_D.append(generator(noise))
                   
        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        if self.unsupervised:
            valid = self.discriminator(inputs_D)
        else:
            valid, _ = self.discriminator(inputs_D)

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity 
        self.combined = Model(noise , valid)                
        self.combined.compile(loss=[fake_real_loss],
            optimizer=c_optimizer)

        print("total architecture")
        self.combined.summary()
    
    def parse_pretrain_args(self, args_str):
        if args_str is None or args_str == '':
            return []
        #freezing and unloading
        parsed_list = []
        if args_str:
            if ',' in args_str:
                parsed_list = args_str.split(',')
            elif ':' in args_str:
                indice = args_str.split(':')
                for idx in range(int(indice[0]), int(indice[1]) + 1, +1):
                    parsed_list.append(idx)
            else:
                parsed_list = args_str.split(",")
        
        return parsed_list
    
    def load_pretrained_model(self, model, path, frozen_layer_list = [], unload_layer_list = []):
        if model is None:
            return None
        
        print("Pre-trained model:", path)
        
        with open(path + ".json", "r") as f:
            for json_str in f:
                premodel = model_from_json(json_str)
                premodel.summary()
                del premodel
                break
        
        #total number of layers
        n_layers = len(model.layers)
        print("total layers: ", n_layers)

        for idx in frozen_layer_list:
            print("layer: ", model.layers[int(idx)].name, " is frozen")
            model.layers[int(idx)].trainable = False    

        for idx in unload_layer_list:
            if idx >= n_layers:
                continue
            '''
            if idx >= n_layers:
                print("You can't unload output layers;keep the same tasks")
                continue
            '''
            print("unloaded layer: ", model.layers[int(idx)].name)
            model.layers[int(idx)].name =  model.layers[int(idx)].name + "_un"

        print("loading weights........")
        model.load_weights(path + ".hdf5", by_name=True)
        
        for layer in model.layers:
            layer.name = layer.name.replace("_un","")
        
        return model
    
    def batch_evaluation(self, epoch, index, total_batch, d_loss, d_loss_real, d_loss_fake, g_loss, logger):
        
        if self.unsupervised == False:
            log("epoch:\t%d\tbatch progress:\t%.2f%%\ttotal D loss:\t%.2f\tacc of real/fake:\t%.2f%%\tacc of class:\t%.2f%%\tG loss:\t%.2f" % (epoch, 100 * float(index)/total_batch, d_loss[0], 100*d_loss[3], 100*d_loss[4], g_loss), logger)
            log("epoch:\t%d\tbatch progress:\t%.2f%%\tReal D loss:\t%.2f\tloss of real/fake:\t%.2f\tloss of class:\t%.2f\tacc of real/fake:\t%.2f%%\tacc of class:\t%.2f%%" % (epoch, 100 * float(index)/total_batch, d_loss_real[0],d_loss_real[1],d_loss_real[2], 100*d_loss_real[3], 100*d_loss_real[4]), logger)
            log("epoch:\t%d\tbatch progress:\t%.2f%%\tFake D loss:\t%.2f\tloss of real/fake:\t%.2f\tloss of class:\t%.2f\tacc of real/fake:\t%.2f%%\tacc of class:\t%.2f%%" % (epoch, 100 * float(index)/total_batch, d_loss_fake[0],d_loss_fake[1],d_loss_fake[2], 100*d_loss_fake[3], 100*d_loss_fake[4]), logger)
        else:
            log("epoch:\t%d\tbatch progress:\t%.2f%%\ttotal D loss:\t%.2f\tacc of real/fake:\t%.2f%%\tG loss:\t%.2f" % (epoch, 100 * float(index)/total_batch, d_loss[0], 100*d_loss[1], g_loss), logger)
            log("epoch:\t%d\tbatch progress:\t%.2f%%\tReal D loss:\t%.2f\tacc of real/fake:\t%.2f%%" % (epoch, 100 * float(index)/total_batch, d_loss_real[0],100*d_loss_real[1]), logger)
            log("epoch:\t%d\tbatch progress:\t%.2f%%\tFake D loss:\t%.2f\tacc of real/fake:\t%.2f%%" % (epoch, 100 * float(index)/total_batch, d_loss_fake[0],100*d_loss_fake[1]), logger)
       
    def epoch_evaluation(self, epoch, X_train, train_validity, train_labels, X_valid, valid_validity, valid_labels, logger, patience):
        
        log("epoch:\t%d\tis done" % (epoch), logger) 
        log("evaluating the model using validation data", logger)
        
        #unsupervised mode
        if valid_labels is None:
            valid_loss = self.discriminator.evaluate(X_valid, valid_validity)
            log("Valid D loss:\t%.2f\tacc of real/fake:\t%.2f%%" % (valid_loss[0],100*valid_loss[1]), logger)
            #TODO: what is a creteria for stopping training in unsupervised mode? convergence of G or D?
            #Currently, it does not support early stopping in unsupervised mode
            return True
        
        else:#supervised mode
            valid_loss = self.discriminator.evaluate(X_valid, [valid_validity, valid_labels])
            log("Valid D loss:\t%.2f\tloss of real/fake:\t%.2f\tloss of class:\t%.2f\tacc of real/fake:\t%.2f%%\tacc of class:\t%.2f%%" % (valid_loss[0],valid_loss[1],valid_loss[2], 100*valid_loss[3], 100*valid_loss[4]), logger)

            #calculate unweighted acc for validation data
            log("prediction using validation data", logger)
            predictions = self.discriminator.predict(X_valid)
            V_UWA, V_CM, V_CM_P = unweighted_acc(predictions[1], valid_labels, self.num_classes)
            log("UWA:\t%.2f%%"% (100 * V_UWA), logger)
            log("CM_P:\t%s"% str(V_CM_P), logger)

            #calculate unweighted acc for whole train data
            log("prediction using whole training data", logger)
            predictions = self.discriminator.predict(X_train)
            T_UWA, T_CM, T_CM_P = unweighted_acc(predictions[1], train_labels, self.num_classes)
            log("UWA:\t%.2f%%"% (100 * T_UWA), logger)
            log("CM_P:\t%s"% str(T_CM_P), logger)

            #check convergence
            if patience > 0 and len(self.earlystopping) > patience:
                if min(self.earlystopping) > V_UWA:
                    log("the recent performance is worse than the worst one in %d slots" %(patience), logger)
                    return False #stop training
                del self.earlystopping[0]
            self.earlystopping.append(V_UWA)
            
            return True #continue training
    
    def evaluate_test_set(self, epoch, X_test, test_validity, test_labels, logger):
        test_loss = self.discriminator.evaluate(X_test, [test_validity, test_labels])
        log("Epoch %d\tTraining is done\n Test D loss:\t%.2f\tloss of real/fake:\t%.2f\tloss of class:\t%.2f\tacc of real/fake:\t%.2f%%\tacc of class:\t%.2f%%" % (epoch, test_loss[0],test_loss[1],test_loss[2], 100*test_loss[3], 100*test_loss[4]), logger)
        
        predictions = self.discriminator.predict(X_test)
        UWA, CM, CM_P = unweighted_acc(predictions[1], test_labels, self.num_classes)
        log("UWA:\t%.2f%%"% (100 * UWA), logger)
        log("CM_P:\t%s"% str(CM_P), logger)
    
    def prepare_data_for_train(self,  X_train, X_test, X_valid, y_train, y_test, y_valid):
        train_validity = np.ones((X_train[0].shape[0], 1))
        test_validity = np.ones((X_test[0].shape[0], 1))
        valid_validity = np.ones((X_valid[0].shape[0], 1))
                
        if self.unsupervised:
            train_labels = None
            test_labels = None
            valid_labels = None
        else:
            #expand class to class + 1, "class1,class2,class3" --> "class1,class2,class3,fake"
            train_labels = to_categorical(y_train, num_classes=self.num_classes+1)
            test_labels = to_categorical(y_test, num_classes=self.num_classes+1)
            valid_labels = to_categorical(y_valid, num_classes=self.num_classes+1)
        
        return train_labels,test_labels,valid_labels,train_validity,test_validity,valid_validity
    
    def train(self, X_train, X_test, X_valid, y_train, y_test, y_valid, epochs, batch_size=128, save_img_epoch_interval=2, check_batch_interval = 100, save_img = None, logger = None, turn_off_G = False, patience = -1, save_model_path = "./model/gan"):

        #prepare for labels of training/test/validation data: validity[real/fake] and class
        train_labels,test_labels,valid_labels,train_validity,test_validity,valid_validity = self.prepare_data_for_train(X_train, X_test, X_valid, y_train, y_test, y_valid)
        
        half_batch = int(batch_size / 2)
        noise_until = epochs
        
        self.discriminator.trainable = True

        self.earlystopping = []
        #class weights should not be used here, it makes nan cost for classification

        total_batch = int(X_train[0].shape[0]/ half_batch)
        print("Number of total batches", total_batch)
        print("Assuming training data(labels) are already shuffled.")
        for epoch in range(epochs):
        
            for index in range(total_batch):

                # ---------------------
                #  Train Discriminator
                # ---------------------
                
                # Sample noise and generate a half batch of new images
                noise = np.random.normal(0, 1, (half_batch, 100))
                
                real_imgs = []
                gen_imgs = []
                for idx in range(len(X_train)):
                    #sample from training data
                    real_img = X_train[idx][index * half_batch:(index+ 1) * half_batch]                                
                    #generator images
                    gen_img = self.generators[idx].predict(noise)
                    
                    real_imgs.append(real_img)
                    gen_imgs.append(gen_img)
                                    
                real_validity = np.ones((half_batch, 1))
                fake_validity = np.zeros((half_batch, 1))
                
                # ---------------------
                #  Train Discriminator
                # ---------------------
                #supervised mode requires labels
                if self.unsupervised == False:
                    #load labels
                    y = y_train[index * half_batch:(index+ 1) * half_batch]
                    #expand class to class + 1, "class1,class2,class3" --> "class1,class2,class3,fake"
                    real_labels = to_categorical(y, num_classes=self.num_classes+1)
                    #expand class to class + 1, "class1,class2,class3" --> "class1,class2,class3,fake"
                    fake_labels = to_categorical(np.full((half_batch, 1), self.num_classes), num_classes=self.num_classes+1)

                    # Train the discriminator
                    d_loss_real = self.discriminator.train_on_batch(real_imgs, [real_validity, real_labels])

                    # if the generator is off
                    if turn_off_G:
                        d_loss_fake = d_loss_real
                    else:
                        d_loss_fake = self.discriminator.train_on_batch(gen_imgs, [fake_validity, fake_labels])
                else:
                    # Train the discriminator
                    d_loss_real = self.discriminator.train_on_batch(real_imgs, real_validity)

                    # if the generator is off
                    if turn_off_G:
                        d_loss_fake = d_loss_real
                    else:
                        d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake_validity)
                
                #calculate total discriminiator's loss
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                
                # ---------------------
                #  Train Generator
                # ---------------------
                if turn_off_G == False:
                #turn discriminator off, we only update weights of generator
                #train the whole NN with noise but only TRUE labels, to maximise generator to fake discriminator
                    self.discriminator.trainable = False
                    noise = np.random.normal(0, 1, (batch_size, 100))
                    validity = np.ones((batch_size, 1))
                    # Train the generator
                    g_loss = self.combined.train_on_batch(noise, validity)

                    #turn on discriminator for next iteration
                    self.discriminator.trainable = True
                else:
                    g_loss = 0.0
                    
                # Plot the progress
                if index % check_batch_interval == 0:
                    self.batch_evaluation(epoch, index, total_batch, d_loss, d_loss_real, d_loss_fake, g_loss, logger)

            # If at save interval => save generated image samples, and evaluating using validation data
            if epoch % save_img_epoch_interval == 0:
                
                #check convergence
                if self.epoch_evaluation(epoch, X_train, train_validity, train_labels, X_valid, valid_validity, valid_labels, logger, patience) == False:
                    break
                
                #save image
                if save_img:
                    for idx in range(len(X_train)):
                        self.save_imgs(epoch, save_img, logger, 2, 2)
                        
                    log("epoch:\t%d\t image is saved" % (epoch), logger)
                    
        if self.unsupervised == False:
            #evaluation on test data
            self.evaluate_test_set(epoch, X_test, test_validity, test_labels, logger)
            
        #save model
        self.save_model(save_model_path)
    
    def save_imgs(self, epoch, name = "./images", logger = None, r = 2, c = 2):
        
        if self.generators is None:
            return
        
        r, c = 2, 2
        noise = np.random.normal(0, 1, (r * c, 100))
        gen_imgs = [] 
        
        for i in range(len(self.generators)):
            generator = self.generators[i]
            imgs = generator.predict(noise)
            gen_imgs.append(imgs)
            
            # Rescale images 0 - 1
            gen_imgs = 0.5 * gen_imgs + 1

            fig, axs = plt.subplots(r, c)
            cnt = 0
            for i in range(r):
                for j in range(c):
                    axs[i,j].imshow(gen_imgs[cnt, :,:,0])
                    axs[i,j].axis('off')
                    cnt += 1
            fig.savefig("%s/gan_img_%s_%d.png" % (name, self.modality[i], epoch))
            plt.close()
        
        prob = self.discriminator.predict(gen_imgs)
        
        if logger:
            print(prob)
            
            log("epoch:\t%d\tgenerated %s image"%(epoch, postfix), logger)
            #write down probablity of validation(real/fake) and classification in the log file
            for validity_class in prob:
                log("prob:%s"%(str(validity_class)), logger)
        

    def save_model(self, model_name):
        def save(model, model_name):
            if model is None:
                return
            
            #just save as .H5
            model.save(model_name)
            
        def save_weight(model, model_path, weights_path):
            if model is None:
                return
            
            options = {"file_arch": model_path, 
                        "file_weight": weights_path}
            json_string = model.to_json()
            open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])
        
            
        for idx in range(len(X_train)):
            #save(self.generators[idx], model_name + "." + self.modality[idx] + ".generator.h5")
            save_weight(self.generators[idx], model_name + "." + self.modality[idx] + ".generator.json", model_name + "." + self.modality[idx] + ".generator.hdf5")
        
        save_weight(self.discriminator, model_name + ".discriminator.json", model_name + ".discriminator.hdf5")
        save_weight(self.combined, model_name + ".adversarial.json", model_name + ".adversarial.hdf5")
        #save(self.discriminator, model_name + ".discriminator.h5")
        #save(self.combined, model_name + ".adversarial.h5")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--batch", dest= 'batch', type=int, help="batch size", default=128)
    parser.add_argument("-e", "--epoch", dest= 'epoch', type=int, help="maximum number of epoch", default=10)
    parser.add_argument("-p", "--patience", dest= 'patience', type=int, help="patience size for early stopping", default=5)
    
    parser.add_argument("-check_batch_interval", "--check_batch_interval", dest= 'check_batch_interval', type=int, help="batch interval for validation", default=10)
    parser.add_argument("-save_img_interval", "--save_img_interval", dest= 'save_img_interval', type=int, help="epoch interval for saving generated images", default=1)
    
    #gan
    parser.add_argument("-r_nrow", "--r_nrow", dest= 'r_nrow', type=str, help="reshaped # rows, for multimodal features, separated by ;", default = "28;48")
    parser.add_argument("-r_ncol", "--r_ncol", dest= 'r_ncol', type=str, help="reshaped # columns, for multimodal features, separated by ;", default = "28;48")

    parser.add_argument("-task_weights", "--task_weights", dest= 'task_weights', type=str, help="weigths for task in GAN(validity, classification)", default="0.5,0.5")
    parser.add_argument("-d_lr", "--D_learningrate", dest= 'd_lr', type=float, help="learning rate for a discriminator", default=0.0002)
    parser.add_argument("-g_lr", "--G_learningrate", dest= 'g_lr', type=float, help="learning rate for generator(s)", default=0.0004)
    
    #convolution
    parser.add_argument("-depth_G", "--depth_G", dest = 'depth_G', type = str, help = "depth of generator(s)", default = "4;4")
    parser.add_argument("-depth_D", "--depth_D", dest = 'depth_D', type = str, help = "depth of discriminator(s)", default = "4;4")
    
    parser.add_argument("-depth_a_v_D", "--depth_a_v_D", dest = 'depth_a_v_D', type = int, help = "depth of fully connected layer for a combined discriminator", default = 1)

    parser.add_argument("-n_node", "--n_node", dest = 'n_node', type = int, help = "# nodes of fully-connected layers after CNNs", default = 512)
    parser.add_argument("-drop","--dropout", dest='dropout', type=float, help="dropout")

    parser.add_argument("-n_kernels_G","--n_kernels_G", dest='n_kernels_G', type=str, help="# kernels in generator(s)", default = "128,64,32,1;128,64,32,1")
    parser.add_argument("-cnn_n_row_G","--cnn_n_row_G", dest='cnn_n_row_G', type=str, help="# rows in CNN for generator(s)", default = "4,4,4,4;4,4,4,4")
    parser.add_argument("-cnn_n_col_G","--cnn_n_col_G", dest='cnn_n_col_G', type=str, help="# cols in CNN for generator(s)", default = "4,4,4,4;4,4,4,4")
    
    parser.add_argument("-n_kernels_D","--n_kernels_D", dest='n_kernels_D', type=str, help="# kernels in discriminator(s)", default = "16,32,64,128;16,32,64,128")
    parser.add_argument("-cnn_n_row_D","--cnn_n_row_D", dest='cnn_n_row_D', type=str, help="# rows in CNN for discriminator(s)", default = "4,4,4,4;4,4,4,4")
    parser.add_argument("-cnn_n_col_D","--cnn_n_col_D", dest='cnn_n_col_D', type=str, help="# cols in CNN for discriminator(s)", default = "4,4,4,4;4,4,4,4")
    parser.add_argument("-pool_n_row_D","--pool_n_row_D", dest='pool_n_row_D', type=str, help="# rows in pooling layers for discriminator(s)", default = "2,2,2,2;2,2,2,2")
    parser.add_argument("-pool_n_col_D","--pool_n_col_D", dest='pool_n_col_D', type=str, help="# cols in pooling layers for discriminator(s)", default = "2,2,2,2;2,2,2,2")
    
    #load, save model
    parser.add_argument("-mod", "--modality", dest= 'modality', type=str, help="input feature name in hd5 DB", default='feat')
    parser.add_argument("-sm", "--save_model", dest= 'save_model', type=str, help="a path to save model", default='./model/model')
    parser.add_argument("-lm_G", "--load_model_G", dest= 'load_model_G', type=str, help="load pre-trained generator(s)")
    parser.add_argument("-lm_D", "--load_model_D", dest= 'load_model_D', type=str, help="load pre-trained discriminator(s)")
    parser.add_argument("-frozen_G", "--frozen_G", dest= 'frozen_G', type=str, help="indice of layers to be frozen in generator(s), indice are separated by comma, e.g. 0,1,2,3, and the weights will not be updated")
    parser.add_argument("-unloaded_G", "--unloaded_G", dest= 'unloaded_G', type=str, help="indice of layers to be ignored in generator(s), indice are separated by comma, e.g. 0,1,2,3, and the weights will not be loaded")
    parser.add_argument("-frozen_D", "--frozen_D", dest= 'frozen_D', type=str, help="indice of layers to be frozen in discriminator(s), indice are separated by comma, e.g. 0,1,2,3, and the weights will not be updated")
    parser.add_argument("-unloaded_D", "--unloaded_D", dest= 'unloaded_D', type=str, help="indice of layers to be ignored in discriminator(s), indice are separated by comma, e.g. 0,1,2,3, and the weights will not be loaded")

    #cv 
    parser.add_argument("-dt", "--data", dest= 'data', type=str, help="path of DB")
    parser.add_argument("-test_idx", "--test_idx", dest= 'test_idx', type=str, help="indice of test sets, separated by commas")
    parser.add_argument("-train_idx", "--train_idx", dest= 'train_idx', type=str, help="indice of train sets, separated by commas")
    parser.add_argument("-valid_idx", "--valid_idx", dest= 'valid_idx', type=str, help="indice of validation sets, separated by commas")

    parser.add_argument("-mt", "--multitasks", dest= 'multitasks', type=str, help="variables for a task to classify, the input format is 'name:#classes:idx:[cost_function]:[weight]'", default = 'class:4:0::')

    parser.add_argument("-log", "--log_file", dest= 'log_file', type=str, help="log file path", default='./output/log.txt')
    parser.add_argument("-save_img", "--save_img_folder", dest= 'save_img_folder', type=str, help="folder to save images")
    parser.add_argument("--default_G_D", help="use default setup for generator(s) and discriminator(s)",
                        action="store_true")
    parser.add_argument("--turn_off_G", help="train only discriminator(s)",
                        action="store_true")
    parser.add_argument("--unsupervised", help="train but no use labels, DB should have a label column that may have no actual values",
                        action="store_true")
                    
    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
   
    if args.save_img_folder:
        try: 
            os.makedirs(args.save_img_folder)
        except OSError:
            if not os.path.isdir(args.save_img_folder):
                raise
    
    if args.log_file:
        log_writer = open(args.log_file, 'a')
    
    #write options
    log(str(args), log_writer)
    
    #task parsing
    nameAndClasses = args.multitasks.split(',')
    multiTasks = []
    for task in nameAndClasses:
        params = task.split(':')
        name = params[0]
        multiTasks.append((name, int(params[1]), int(params[2])))
        
    #no support of Multi-task learning yet, only use the first task in multiTasks
    lab_idx = multiTasks[0][2]
    n_class = multiTasks[0][1]
    
    
    #parsing weights for validity and classification tasks for GAN
    task_weights = []
    for weight in args.task_weights.split(","):
        task_weights.append(float(weight))
    
    #compose idx for training, testing, validating data sets
    train_idx, test_idx, valid_idx = compose_idx(args.train_idx, args.test_idx, args.valid_idx)

    with h5py.File(args.data,'r') as hf:
        print('List of arrays in this file: \n', hf.keys())
        train_csv = []
        if args.modality:
            for modality in args.modality.split(";"):
                data = hf.get(modality)
                s_train_csv = np.array(data)
                print('Shape of the array %s feat: %s'% (modality, str(s_train_csv.shape)))
                train_csv.append(s_train_csv)

        #labels
        data = hf.get('label')
        train_lab = np.array(data)
                
        print('Shape of the array lab: ', train_lab.shape)
        if len(test_idx) > 0:
            start_indice = np.array(hf.get('start_indice'))
            end_indice = np.array(hf.get('end_indice'))
            print('Shape of the indice for start: ', start_indice.shape)


    #2d feature structure: (n_samples, max_t_steps, 1, context_length, input_dim)
    if len(train_csv[0].shape) != 5:        
        print('wrong format, the format should be (n_samples, max_t_steps, 1, context_len, input_dim)')
        exit(-1)
     
    
    #model configuration, if you want to change the architecture of G and D, modify the code in "building_custom_gan" function.
    sgan = building_custom_gan(args, loss_weights = task_weights, d_lr = args.d_lr, g_lr = args.g_lr, n_class = n_class)
    
    #cross-validation
    if len(test_idx) > 0 :

        test_indice = []
        valid_indice = []
        remove_indice = []

        for cid in test_idx:
            print("cross-validation test: ", cid)
            start_idx = start_indice[int(cid)]
            end_idx = end_indice[int(cid)]
            
            if start_idx == 0 and end_idx == 0:
                continue
                
            for idx in range(int(start_idx), int(end_idx), + 1):
                test_indice.append(idx)
                remove_indice.append(idx)

        for cid in valid_idx:
            print("cross-validation valid: ", cid)
            start_idx = start_indice[int(cid)]
            end_idx = end_indice[int(cid)]
            
            if start_idx == 0 and end_idx == 0:
                continue
                
            for idx in range(int(start_idx), int(end_idx), + 1):
                remove_indice.append(idx)
                valid_indice.append(idx)
                
        if len(train_idx):
            train_indice = []
            for cid in train_idx:
                print("cross-validation train: ", cid)
                start_idx = start_indice[cid]
                end_idx = end_indice[cid]
            
                if start_idx == 0 and end_idx == 0:
                    continue
                
                for idx in range(int(start_idx), int(end_idx), + 1):
                    train_indice.append(idx)
            
            X_train = []
            for s_train_csv in train_csv:
                X_train.append(s_train_csv[train_indice])
                
            Y_train = train_lab[train_indice]
        else:
            X_train = []
            for s_train_csv in train_csv:
                X_train.append(np.delete(s_train_csv, remove_indice, axis=0))
            
            Y_train = np.delete(train_lab, remove_indice, axis=0)

        #test set
        X_test = []
        for s_train_csv in train_csv:
            X_test.append(s_train_csv[test_indice])
            
        Y_test = train_lab[test_indice]
        
        #valid set
        if len(valid_indice) == 0:
            X_valid = X_test
                
            Y_valid = Y_test
        else:
            X_valid = []
            for s_train_csv in train_csv:
                X_valid.append(s_train_csv[valid_indice])
                       
            Y_valid = train_lab[valid_indice]
            r_valid = 0.0

        Y_train = Y_train[:,lab_idx]
        Y_test = Y_test[:,lab_idx]
        Y_valid = Y_valid[:,lab_idx]
        print("train label shape, ", Y_train.shape)
        print("test label shape, ", Y_test.shape)
        print("valid label shape, ", Y_valid.shape)
        
        # Reshape
        #2d feature structure: (n_samples, max_t_steps, 1, context_length, input_dim)
        n_row = str_to_int(args.r_nrow.split(";"))
        n_col = str_to_int(args.r_ncol.split(";"))
        
        for i in range(len(train_csv)):
            X_train[i],X_test[i],X_valid[i] = reshape_feat_lab(X_train[i],X_test[i],X_valid[i], True, n_row[i], n_col[i])
                  
        #train
        sgan.train(X_train, X_test, X_valid, Y_train, Y_test, Y_valid, epochs=args.epoch, batch_size=args.batch, save_img_epoch_interval=args.save_img_interval, check_batch_interval = args.check_batch_interval, save_img = args.save_img_folder, logger = log_writer, turn_off_G = args.turn_off_G, patience = args.patience, save_model_path = args.save_model)
    
    #close logger
    log_writer.close()
        