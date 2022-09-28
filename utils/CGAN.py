#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Mohammad Zarei
# Created Date: 10 Oct 2020
# ---------------------------------------------------------------------------
"""Implementation of a Conditional-GAN"""
# ---------------------------------------------------------------------------
# Imports
import warnings
warnings.filterwarnings('ignore')

from matplotlib import pyplot

import numpy as np
from numpy import zeros, ones
from numpy.random import randn, randint


from keras import Input, Model
from keras.datasets.mnist import load_data
from tensorflow.keras.optimizers import Adam
from keras.models import Sequential, load_model
from keras.initializers import RandomNormal
from keras.layers import Dense, Reshape, Flatten, BatchNormalization, Dropout
from keras.layers import Conv2D, Conv1D, Conv2DTranspose
from keras.layers import LeakyReLU, ELU, ReLU, concatenate, multiply



class CGAN():
    def __init__(self, latent_dim=1, x_size=4, in_shape = 1, seed=42, activation='elu') -> None:
        self.x_size = x_size
        self.in_shape = in_shape
        self.activation = activation
        self.kerner_initializer = keras.initializers.he_normal(seed=seed)
        self.random_uniform = keras.initializers.RandomUniform(seed=seed)
        self.random_normal = keras.initializers.RandomNormal(seed=seed)
        self.latent_dim = latent_dim
        
        
    def define_discriminator(self, dis_lr):
        y = Input(shape=(self.x_size,), dtype='float')
        y_output = Dense(100, activation=self.activation, kernel_initializer=self.kerner_initializer)(y)

        label = Input(shape=(1,))
        label_output = Dense(100, activation=self.activation, kernel_initializer=self.kerner_initializer)(label)

        concat = concatenate([y_output, label_output])
        concat = Dense(50, activation=self.activation, kernel_initializer=self.kerner_initializer)(concat)
        concat = Dense(50, activation=self.activation, kernel_initializer=self.kerner_initializer)(concat)
        validity = Dense(1, activation="sigmoid", kernel_initializer=self.random_uniform)(concat)

        model = Model(inputs=[y, label], outputs=validity)

        # compile model
        opt = Adam(learning_rate=dis_lr, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        return model

    # define the standalone generator model
    def define_generator(self):
        y = Input(shape=self.in_shape, dtype='float')
        y_output = Dense(100, activation=self.activation, kernel_initializer=self.kerner_initializer)(y)

        noise = Input(shape=(self.latent_dim,))
        noise_output = Dense(100, activation=self.activation, kernel_initializer=self.kerner_initializer)(noise)

        concat = concatenate([y_output, noise_output])

        output = Dense(50, activation=self.activation, kernel_initializer=self.kerner_initializer)(concat)
        output = Dense(50, activation=self.activation, kernel_initializer=self.kerner_initializer)(output)
        output = Dense(50, activation=self.activation, kernel_initializer=self.kerner_initializer)(output)
        output = Dense(self.x_size, activation="relu", kernel_initializer=self.random_normal)(output)

        model = Model(inputs=[noise, y], outputs=output)
        return model

    # define the combined generator and discriminator model, for updating the generator
    def define_gan(self, g_model, d_model, gen_lr):
        # make weights in the discriminator not trainable
        d_model.trainable = False
        # get noise and label inputs from generator model
        gen_noise, gen_label = g_model.input
        # get output from the generator model
        gen_output = g_model.output
        # connect output and label input from generator as inputs to discriminator
        gan_output = d_model([gen_output, gen_label])
        # define gan model as taking noise and label and outputting a classification
        model = Model([gen_noise, gen_label], gan_output)
        # compile model
        opt = Adam(learning_rate=gen_lr, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt)
        return model

    # load and prepare training set
    def load_real_samples(self, x_train,y_train):

        trainX = x_train.to_numpy()
        trainY = y_train.to_numpy()

        return [trainX, trainY]

    # select real samples
    def generate_real_samples(self, dataset, n_samples):
        # choose random instances
        ix = randint(0, dataset[0].shape[0], n_samples)
        # retrieve selected images
        X, labels = dataset[0][ix], dataset[1][ix]
        # generate 'real' class labels (1)
        y = ones((n_samples, 1))
        return [X, labels], y


    # generate points in latent space as input for the generator
    def generate_latent_points(self, n_samples, dataset):
        # generate points in the latent space
        x_input = randn(self.latent_dim * n_samples) #randn(latent_dim * n_samples)
        # reshape into a batch of inputs for the network
        z_input = x_input.reshape(n_samples, self.latent_dim)
        # generate labels
        idx = np.random.randint(0, dataset.shape[0], n_samples)
        labels = dataset[1][idx]
        return [z_input, labels]

    # use the generator to generate n fake examples, with class labels
    def generate_fake_samples(self, generator, n_samples, dataset):
        # generate points in latent space
        z_input, labels_input = self.generate_latent_points(n_samples, dataset)
        # predict outputs
        samples = generator.predict([z_input, labels_input])
        # create class labels
        y = zeros((n_samples, 1))
        return [samples, labels_input], y

    # create a line plot of loss for the gan and save to file
    def plot_history(self, d1_hist, d2_hist, g_hist):
        # plot history
        pyplot.plot(d1_hist, label='Dis loss real')
        pyplot.plot(d2_hist, label='Dis loss fake')
        pyplot.plot(g_hist, label='Gen loss')
        pyplot.legend()
        pyplot.savefig('plot_line_plot_loss.png')

    # evaluate the discriminator, plot generated images, save generator model
    def summarize_performance(self, save_path, epoch, g_model, d_model, dataset, n_samples=100):
        # prepare real samples
        X_real, y_real = self.generate_real_samples(dataset, n_samples)
        # evaluate discriminator on real examples
        _, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
        # prepare fake examples
        x_fake, y_fake = self.generate_fake_samples(g_model, n_samples, dataset)
        # evaluate discriminator on fake examples
        _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
        # summarize discriminator performance
        print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
        # save the generator model tile file
        g_model.save(save_path+f'gen_e{epoch}.h5')

    # train the generator and discriminator
    def train(self, g_model, d_model, gan_model, dataset, n_epochs, n_batch, save_path):
        bat_per_epo = int(dataset[0].shape[0] / n_batch)
        half_batch = int(n_batch / 2)
        # lists for keeping track of loss
        d1_hist, d2_hist, g_hist = list(), list(), list()
        # manually enumerate epochs
        for i in range(n_epochs):
            # enumerate batches over the training set
            for j in range(bat_per_epo):
                # get randomly selected 'real' samples
                [X_real, labels_real], y_real = self.generate_real_samples(dataset, half_batch)
                # update discriminator model weights
                d_loss1, _ = d_model.train_on_batch([X_real, labels_real], y_real)
                # generate 'fake' examples
                [X_fake, labels], y_fake = self.generate_fake_samples(g_model, half_batch)
                # update discriminator model weights
                d_loss2, _ = d_model.train_on_batch([X_fake, labels], y_fake)
                # prepare points in latent space as input for the generator
                [z_input, labels_input] = self.generate_latent_points(n_batch)
                # create inverted labels for the fake samples
                y_gan = ones((n_batch, 1))
                # update the generator via the discriminator's error
                g_loss = gan_model.train_on_batch([z_input, labels_input], y_gan)

            d1_hist.append(d_loss1)
            d2_hist.append(d_loss2)
            g_hist.append(g_loss)
            # evaluate the model performance, sometimes
            if i >= 0 and (i+1) % 100 == 0:
                self.summarize_performance(save_path, i, g_model, d_model, dataset)
                print('Epoch: %d, dLoss real = %.3f, dLoss fake = %.3f, gLoss = %.3f' % (i+1, d_loss1, d_loss2, g_loss))
        self.plot_history(d1_hist, d2_hist, g_hist)
        return d1_hist, d2_hist, g_hist


