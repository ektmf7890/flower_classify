import tensorflow as tf
import numpy as np
from PIL import Image

# takes numpy image -> resize, normalize -> return as numpy image
def process_image(img_path):
  img = np.asarray(Image.open(img_path))
  img = tf.convert_to_tensor(img)
  img = tf.image.resize(img, (224, 224))
  img /= 255
  return img.numpy()