from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D

from aitoolbox.kerastrain.train_loop import TrainLoop, TrainLoopModelCheckpointEndSave
from aitoolbox.experiment.result_package.basic_packages import ClassificationResultPackage


batch_size = 32
num_classes = 10
epochs = 100
data_augmentation = True
num_predictions = 20

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train[:200]
y_train = y_train[:200]
x_test = x_test[:100]
y_test = y_test[:100]

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

optimizer = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)


# TrainLoop(model, optimizer, 'categorical_crossentropy', ['accuracy'])(num_epochs=10, batch_size=batch_size,
#                                                                       x=x_train,
#                                                                       y=y_train,
#                                                                       validation_data=(x_test, y_test),
#                                                                       shuffle=True)

TrainLoopModelCheckpointEndSave(model,
                                train_loader=[x_train, y_train], validation_loader=[x_test, y_test], test_loader=[x_test, y_test],
                                optimizer=optimizer, criterion='categorical_crossentropy', metrics=['accuracy'],
                                project_name='kerasloop_final', experiment_name='checkpoint_endsave',
                                local_model_result_folder_path='<PATH>',
                                args={},
                                val_result_package=ClassificationResultPackage(),
                                test_result_package=ClassificationResultPackage())\
    (num_epochs=1, batch_size=batch_size,
     shuffle=True)
