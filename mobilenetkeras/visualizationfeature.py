import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from keras.preprocessing import image
import keras
from keras import models
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50
from keras.applications.mobilenet import preprocess_input
from keras import Model, layers
from keras.models import load_model, model_from_json
import tensorflow as tf

model = load_model('mobilenetmodelinputnooutput4kategori.h5')
img_path = "C:/Users/wing/Pictures/Camera Roll/kaca1.jpg"
img = image.load_img(img_path, target_size=(224, 224))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.
print(img_tensor.shape)


x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
img = preprocess_input(x)

feature_maps = model.predict(img)
images = np.vstack([x])

# plot all 64 maps in an 8x8 squares
square = 8
ix = 1
for _ in range(square):
    for _ in range(square):
        # specify subplot and turn of axis
        ax = plt.subplot(square, square, ix)
        ax.set_xticks([])
        ax.set_yticks([])
        # plot filter channel in grayscale
        plt.imshow(feature_maps[0, :, :, ix - 1], cmap='gray')
        ix += 1
# show the figure
plt.show()
