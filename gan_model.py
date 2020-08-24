import tensorflow as tf

import os
import time

from matplotlib import pyplot as plt
from IPython import display
from tensorflow.keras import layers

import image_functions as I_F
import h5py


from scipy.io import loadmat
import numpy as np
import pandas as pd
import sklearn.preprocessing
from sklearn import preprocessing

BUFFER_SIZE = 500
SHAPE = 128


gan_train = tf.data.Dataset.list_files(
    os.getcwd()+'/GAN_TRAIN/*.*', shuffle=False)

dec_train = tf.data.Dataset.list_files(
    os.getcwd()+'/TRAIN_IMG/*.*', shuffle=False)

gan_train_img = gan_train.map(I_F.load_image_train,num_parallel_calls=tf.data.experimental.AUTOTUNE)


dec_train_img = dec_train.map(I_F.load_image_train,num_parallel_calls=tf.data.experimental.AUTOTUNE)


h5f = h5py.File('gan_feat.h5', 'r')
gan_train_feat = h5f['train'][:]
h5f.close()

h5f = h5py.File('train_feat.h5', 'r')
dec_train_feat = h5f['train'][:]
h5f.close()

gan_train_feat = tf.data.Dataset.from_tensor_slices(gan_train_feat)
gan_train_data = tf.data.Dataset.zip((gan_train_feat, gan_train_img))
gan_train_data = gan_train_data.shuffle(BUFFER_SIZE)
gan_train_data = gan_train_data.batch(64)
print(gan_train_data)

dec_train_feat = tf.data.Dataset.from_tensor_slices(dec_train_feat)
dec_train_data = tf.data.Dataset.zip((dec_train_feat, dec_train_img))
dec_train_data = dec_train_data.shuffle(BUFFER_SIZE)
dec_train_data = dec_train_data.batch(16)
print(dec_train_data)


def Generator():
    model=tf.keras.Sequential()

    model.add(tf.keras.Input(shape=(8192)))
    model.add(layers.Dense(8192))

    model.add(layers.Reshape((4, 4, 512)))
    assert model.output_shape == (None, 4, 4, 512) # Note: None is the batch size

    model.add(layers.Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 8, 8, 512)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(512, (3, 3), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 16, 16, 512)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 32, 32, 256)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 64, 64, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 128, 128, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, (3, 3), strides=(1, 1), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 128, 128, 3)

    return model

generator = Generator()
# tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64)
# generator.summary()

def Discriminator():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(16, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[128, 128, 3])) 
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    # model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')) 
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same')) 
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same')) 
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

discriminator = Discriminator()
# discriminator =tf.keras.models.load_model('DISCRIMINATOR')
# tf.keras.utils.plot_model(discriminator, show_shapes=True, dpi=64)

LAMBDA = 1000

def generator_loss(disc_generated_output, gen_output, target):
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

  # mean absolute error
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

  total_gen_loss = gan_loss + (LAMBDA * l1_loss)

  return total_gen_loss, gan_loss, l1_loss

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(disc_real_output, disc_generated_output):
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)


@tf.function
def train_step_1(input_image,target, epoch):
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape :

    gen_output = generator(input_image, training=True)

    disc_real_output = discriminator(target, training=True)
    disc_generated_output = discriminator(gen_output, training=True)

    gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)


  generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
  discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

  generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))

def fit(train_data,test_data, epochs):
  for epoch in range(epochs):
    start = time.time()

    display.clear_output(wait=True)


    # Train
    print(epoch)

    for n, (input_image,target) in train_data.enumerate():
      print('.', end='')
      if (n+1) % 100 == 0:
        print()
      train_step_1(input_image,target, epoch)
    print()


    for (example_input,example_target) in test_data.take(1):
        I_F.generate_images(generator, example_input, example_target)
    print("TEST Epoch: ", epoch)

    # saving (checkpoint) the model every 10 epochs
    if (epoch + 1) % 10 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                        time.time()-start))
  # checkpoint.save(file_prefix = checkpoint_prefix)



# checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

EPOCHS = 50

fit(gan_train_data,dec_train_data, EPOCHS)

generator.save('GENERATOR')
# discriminator.save('DISCRIMINATOR')
