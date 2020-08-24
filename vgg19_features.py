import tensorflow as tf
import os
import image_functions as I_F
import h5py

vgg19 = tf.keras.applications.VGG19(include_top=False, weights='imagenet', input_shape=(128, 128, 3), input_tensor=None, pooling=None, classes=1000)
vgg19.trainable = False


gan_train = tf.data.Dataset.list_files(os.getcwd()+'/GAN_TRAIN/*.*', shuffle=False)

dec_train = tf.data.Dataset.list_files(os.getcwd()+'/TRAIN_IMG/*.*', shuffle=False)

dec_test = tf.data.Dataset.list_files(os.getcwd()+'/TEST_IMG/*.*', shuffle=False)


gan_train = gan_train.map(I_F.load_image_train,num_parallel_calls=tf.data.experimental.AUTOTUNE)
gan_train = gan_train.batch(128)

dec_train = dec_train.map(I_F.load_image_train,num_parallel_calls=tf.data.experimental.AUTOTUNE)
dec_train = dec_train.batch(1)


dec_test = dec_test.map(I_F.load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
dec_test = dec_test.batch(1)


GAN_FEAT = vgg19.predict(gan_train)

h5f = h5py.File('/data/gan_feat.h5', 'w')
h5f.create_dataset('train', data=GAN_FEAT)
h5f.close()

TRAIN_FEAT = vgg19.predict(dec_train)

h5f = h5py.File('/data/train_feat.h5', 'w')
h5f.create_dataset('train', data=TRAIN_FEAT)
h5f.close()

TEST_FEAT = vgg19.predict(dec_test)

h5f = h5py.File('/data/test_feat.h5', 'w')
h5f.create_dataset('test', data=TEST_FEAT)
h5f.close()


