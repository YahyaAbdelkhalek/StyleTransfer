import IPython.display as display
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import tensorflow as tf
import time
from models.Image import Image
vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
vgg.summary()
content_layers = ['block5_conv2']

style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']
class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.model = StyleContentModel.vgg_layers(style_layers + content_layers)
        self.model.trainable = False
        
        self.style_layers = style_layers
        self.content_layers = content_layers
        
        self.num_content_layers = len(content_layers)
        self.num_style_layers = len(style_layers)

    def call(self, inputs):
        # Expects tensor with values ranged [0,1]
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
        outputs = self.model(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])

        style_outputs = [StyleContentModel.gram_matrix(style_output) for style_output in style_outputs]

        content_dict = {content_name: value for content_name, value in zip(self.content_layers, content_outputs)}
        style_dict = {style_name: value for style_name, value in zip(self.style_layers, style_outputs)}

        return {'content': content_dict, 'style': style_dict}
    
    # Compute Gram Matrix, specified from the paper
#     @staticmethod
    def gram_matrix(input_tensor):
        result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
        input_shape = tf.shape(input_tensor)
        num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
        return result / num_locations
    
    # Extract selected VGG19 layers and return a new model
#     @staticmethod
    def vgg_layers(layer_names):
        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False

        outputs = [vgg.get_layer(name).output for name in layer_names]

        model = tf.keras.Model([vgg.input], outputs)
        return model 
class NeuralStyleTransfer:
    def __init__(self, model, content_img_path, style_img_path, max_dim=512, 
                 content_weight=1e4, style_weight=1e-2, total_variation_weight=30,
                 optimizer=tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)):
        self.max_dim = max_dim
        self.model = model
        
        self.content_img = Image.load_img(content_img_path)
        self.style_img = Image.load_img(style_img_path)
        self.result_img = tf.Variable(self.content_img)
        
        self.content_targets = self.model(self.content_img)['content']
        self.style_targets = self.model(self.style_img)['style']
        
        self.content_weight = content_weight
        self.style_weight = style_weight
        
        self.optimizer = optimizer
        self.total_variation_weight = total_variation_weight
    
    # Compute loss
    def style_content_loss(self, outputs):
        style_outputs = outputs['style']
        content_outputs = outputs['content']

        content_loss = tf.add_n([tf.reduce_mean((content_outputs[name] - self.content_targets[name]) ** 2) for name in content_outputs.keys()])
        content_loss *= self.content_weight / self.model.num_content_layers
        
        style_loss = tf.add_n([tf.reduce_mean((style_outputs[name] - self.style_targets[name]) ** 2) for name in style_outputs.keys()])
        style_loss *= self.style_weight / self.model.num_style_layers

        loss = style_loss + content_loss
        return loss
    
    @tf.function
    def train_step(self):
        with tf.GradientTape() as tape:
            outputs = self.model(self.result_img)
            loss = self.style_content_loss(outputs)

            # Regularization term on the high frequency components of the image
            loss += self.total_variation_weight * tf.image.total_variation(self.result_img)

        grad = tape.gradient(loss, self.result_img)
        self.optimizer.apply_gradients([(grad, self.result_img)])
        self.result_img.assign(Image.clip_0_1(self.result_img))
        
    # Transfer style to the resulting image
    def train(self, epochs, steps_per_epoch=50):
        start = time.time()
        step = 0
        for n in range(epochs):
            for m in range(steps_per_epoch):
                step += 1
                self.train_step()
                print(".", end='', flush=True)
            display.clear_output(wait=True)
            display.display(Image.tensor_to_image(self.result_img))
            print("Train step: {}".format(step))

        end = time.time()
        print("Total time: {:.1f}".format(end - start))
    
    # Save results in the working directory
    def export_results(self, file_name):
        img = Image.tensor_to_image(self.result_img)
        img.save(file_name)
        
model = StyleContentModel(style_layers, content_layers)
