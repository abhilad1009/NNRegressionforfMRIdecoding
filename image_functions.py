import tensorflow as tf

from matplotlib import pyplot as plt



IMG_SIZE = 128          # Dimension of resized image


def load(image_file):
  image = tf.io.read_file(image_file)
  image = tf.image.decode_jpeg(image)

  input_image = tf.cast(image, tf.float32)

  return input_image


def resize(input_image, height, width):
  input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  return input_image


def normalize(input_image):
  input_image = (input_image / 127.5) - 1

  return input_image


def load_image_train(image_file):
  input_image = load(image_file)
  input_image = resize(input_image, IMG_SIZE, IMG_SIZE)
  input_image = normalize(input_image)

  return input_image


def generate_images(model, test_input, target):
  prediction = model(test_input, training=False)
  plt.figure(figsize=(5, 5))

  display_list = [target[0], prediction[0]]
  title = ['Ground Truth', 'Predicted Image']

  for i in range(2):
    plt.subplot(1, 2, i+1)
    plt.title(title[i])
    # getting the pixel values between [0, 1] to plot it.
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
  plt.show()
