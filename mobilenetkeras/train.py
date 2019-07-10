from __future__ import absolute_import, division, print_function

import os
import keras
from keras.callbacks import TensorBoard
import csv
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import TensorBoard
from keras.applications.mobilenet import preprocess_input
import numpy as np
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras import Model, layers



# traindata = 'C:/Users/wing/Desktop/datarealtrain/data-train-type1/'
# path = 'model5perbedaanmobilenet/model1/callbackacuracy/'

def main(path, traindata):
    def plot_training(history):
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(len(acc))

        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(path + 'accuracymobilenetmodel.png')

        plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(path + 'lossmobilenetmodel.png')

    def setup_to_transfer_learn(model, base_model):
        """Freeze all layers and compile the model"""
        for layer in base_model.layers:
            layer.trainable = False
        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

    img_width, img_height = 224, 224

    train_data_dir = traindata + 'train'
    validation_data_dir = traindata + 'val'
    csv_file = path + "historymodelmobilenet.csv"

    epochs = 100
    batch_size = 15
    finetunepath = path + 'cp.fine_tuned.best.h5'

    tensorboard = TensorBoard(log_dir=path + './logs', histogram_freq=0,
                              write_graph=True, write_images=False)
    earlystop = keras.callbacks.EarlyStopping(monitor='acc',patience=5,mode='max')
    checkpoint = keras.callbacks.ModelCheckpoint('best_model.h5', monitor='acc', verbose=1, save_best_only=True, mode='max')

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    # Train Data Generator
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

    # train_features = model.predict_generator(
    #     train_generator, train_generator.n // batch_size, verbose=1)
    # np.save('train_features.npy', train_features)

    # Testing Data Generator
    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical')

    base_model = keras.applications.mobilenet.MobileNet(input_shape=(224,224,3),
                                                   include_top=False,
                                                   weights='imagenet')


    x = base_model.output
    x = layers.GlobalAvgPool2D()(x)
    # x = layers.Dense(128, activation='relu')(x)
    x=layers.Dropout(0.5)(x)
    # x = layers.Dense(32, activation='relu')(x)
    # x=layers.Dropout(0.5)(x)
    predictions = layers.Dense(4, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)


    # setup_to_transfer_learn(model, base_model)
    # for layer in base_model.layers[:4]:
    #     layer.trainable = False
    # model = keras.Sequential([
    #   base_model,
    #   keras.layers.GlobalAveragePooling2D(),
    #   keras.layers.Dense(4, activation='softmax')
    # ])

    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # adam = Adam(lr=0.0001)
    # model.compile(optimizer=adam,loss='categorical_crossentropy', metrics=['accuracy'])

    step_size_train=train_generator.n//batch_size
    valid_step = validation_generator.n//batch_size

    # history=model.fit_generator(generator=train_generator,steps_per_epoch=step_size_train,epochs=epochs,callbacks=[tensorboard,earlystop,checkpoint],validation_data=validation_generator,validation_steps=valid_step)

    history = model.fit_generator(generator=train_generator, steps_per_epoch=step_size_train, epochs=epochs,
                                  callbacks=[tensorboard], validation_data=validation_generator,
                                  validation_steps=valid_step)

    # mc_fit = keras.callbacks.ModelCheckpoint(finetunepath, monitor='val_loss', verbose=0, save_best_only=True,
    #                          save_weights_only=False, mode='auto', period=1)
    # earlystop2 = keras.callbacks.EarlyStopping(monitor='acc', patience=5, mode='max')
    # for layer in model.layers[:81]:
    #     layer.trainable = False
    #
    # for layer in model.layers[81:]:
    #     layer.trainable = True
    #
    # # model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer=Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    # history = []
    # history = model.fit_generator(generator=train_generator, steps_per_epoch=step_size_train, epochs=epochs,
    #                               callbacks=[tensorboard, mc_fit, earlystop2], validation_data=validation_generator,
    #                               validation_steps=valid_step)

    plot_training(history)

    model.save(path + 'mobilenetmodel.h5')

    # architecture to JSON, weights to HDF5
    model.save_weights(path + 'weightsmobilenetmodel.h5')
    with open(path + 'architecturemobilenetmodelepoch.json', 'w') as f:
            f.write(model.to_json())

    a = [list(a) for a in zip(history.history['val_loss'],history.history['val_acc'],history.history['loss'],history.history['acc'])]
    print(a)


    np.savetxt(csv_file, a,delimiter=',',header= "val_loss, val_acc, loss, acc")


# path = ['model5perbedaanmobilenet/model1/finetuneVBener/','model5perbedaanmobilenet/model2/finetuneVBener/','model5perbedaanmobilenet/model3/finetuneVBener/','model5perbedaanmobilenet/model4/finetuneVBener/','model5perbedaanmobilenet/model5/finetuneVBener/']
# train = ['C:/Users/wing/Desktop/datarealtrain/dat/a-train-type1/','C:/Users/wing/Desktop/datarealtrain/data-train-type2/','C:/Users/wing/Desktop/datarealtrain/data-train-type3/','C:/Users/wing/Desktop/datarealtrain/data-train-type4/','C:/Users/wing/Desktop/datarealtrain/data-train-type5/']
# path = ['model5perbedaanmobilenet/model4/finetuneVBener/','model5perbedaanmobilenet/model5/finetuneVBener/']
# train = ['C:/Users/wing/Desktop/datarealtrain/data-train-type4/','C:/Users/wing/Desktop/datarealtrain/data-train-type5/']
train = 'C:/Users/wing/Desktop/datarealtrain/data-train-type1/'
path = 'model5perbedaanmobilenet/model1/cara1100epoch/'
# for p,t in zip(path,train):
#     main(p,t)

main(path,train)
