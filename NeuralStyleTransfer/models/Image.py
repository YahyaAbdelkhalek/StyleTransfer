import IPython.display as display
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import tensorflow as tf
import time
class Image:
    # Loads an image from path, returns a tensor with values in [0, 1]
#     @staticmethod
    def load_img(img_path, max_dim=512):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)

        shape = tf.cast(tf.shape(img)[:-1], tf.float32)
        long_dim = max(shape)
        scale = max_dim / long_dim

        new_shape = tf.cast(scale * shape, tf.int32)

        img = tf.image.resize(img, new_shape)
        img = img[tf.newaxis, :]
        return img
    
#     @staticmethod
    def tensor_to_image(tensor):
        # Convert value range from [0, 1] to [0, 255]
        tensor = tensor * 255
        tensor = np.array(tensor, dtype=np.uint8)

        # Reduce tensor dimension if appropriate
        if np.ndim(tensor) > 3:
            assert tensor.shape[0] == 1
            tensor = tensor[0]

        return PIL.Image.fromarray(tensor)
    
#     @staticmethod
    def clip_0_1(img):
        return tf.clip_by_value(img, clip_value_min=0.0, clip_value_max=1.0)
    
#     @staticmethod
    def display_image(img, title=None):
        if len(img.shape) > 3:
            img = tf.squeeze(img, axis=0)

        plt.imshow(img)
        if title:
            plt.title(title)