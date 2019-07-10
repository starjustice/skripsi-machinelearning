import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50
from keras.applications.mobilenet import preprocess_input
from keras import Model, layers
from keras.models import load_model, model_from_json
import tensorflow as tf

model = load_model('mobilenetmodelinputnooutput4kategori.h5')

# architecture from JSON, weights from HDF5
with open('architecturemobilenetmodelinputnooutput4kategori.json') as f:
    model = model_from_json(f.read())
model.load_weights('weightsmobilenetmodelinputnooutput4kategori.h5')

validation_img_paths = [
                        "dataset-resizedv1/val/plastic/plastic1.jpg",
                        "dataset-resizedv1/val/glass/glass486.jpg",
                        "dataset-resizedv1/val/glass/glass488.jpg",
                        "dataset-resizedv1/val/plastic/plastic11.jpg",
    "../../../kardus.jpg",
    "../../../test1.jpg",
    "dataset-resized/test/paper1.jpg",
    "dataset-resized/test/paper3.jpg",
    "dataset-resized/test/cardboard85.jpg",
    "dataset-resized/test/cardboard147.jpg",
    "dataset-resized/test/kaleng (3).jpg",
    "dataset-resized/val/metal/metal387.jpg",
"C:/Users/wing/Pictures/Camera Roll/kaleng1.jpg",
                        "C:/Users/wing/Pictures/Camera Roll/kaleng2.jpg",
                        "C:/Users/wing/Pictures/Camera Roll/kaleng3.jpg",
                        "C:/Users/wing/Pictures/Camera Roll/kaleng4.jpg",
                        "C:/Users/wing/Pictures/Camera Roll/plastic1.jpg",
                        "C:/Users/wing/Pictures/Camera Roll/kaca1.jpg",
                        "C:/Users/wing/Pictures/Camera Roll/kaca2.jpg"
]
img_list = [Image.open(img_path) for img_path in validation_img_paths]


validation_batch = np.stack([preprocess_input(np.array(img.resize((224,224))))
                             for img in img_list])

pred_probs = model.predict(validation_batch)

for i, img in enumerate(img_list):
    print("{:.0f}% glass, {:.0f}% metal, {:.0f}% organic, {:.0f}% plastic".format(100*pred_probs[i,0],100*pred_probs[i,1],100*pred_probs[i,2],100*pred_probs[i,3]))