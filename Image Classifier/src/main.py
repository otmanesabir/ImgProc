#!/usr/bin/env python
import os

import tensorflow as tf
from IPython import get_ipython
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Dropout

from tqdm import tqdm

import numpy as np
import io
from packaging import version
from six.moves import range

import matplotlib.pyplot as plt

print("TensorFlow version: ", tf.__version__)
assert version.parse(tf.__version__).release[0] >= 2, "This notebook requires TensorFlow 2.0 or above."

"""this is important because i realized the model might not reset"""
tf.keras.backend.clear_session()

plt.style.use('seaborn')
plt.rcParams['figure.figsize'] = (14.0, 5.0)

np.random.seed(7)

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = tqdm(mnist.load_data(), desc="Loading Data")
x_train, x_test = x_train / 255.0, x_test / 255.0

x2_train = x_train[..., tf.newaxis]
x2_test = x_test[..., tf.newaxis]

train_ds = tf.data.Dataset.from_tensor_slices((x2_train, y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x2_test, y_test)).batch(32)

class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

img = np.reshape(x_train[0], (-1, 28, 28, 1))


os.system("rm -rf logs")

logdir = "logs/sample_data/"
file_writer = tf.summary.create_file_writer(logdir)

with file_writer.as_default():
    images = np.reshape(x_train[0:25], (-1, 28, 28, 1))
    tf.summary.image("25 training data examples", images, max_outputs=25, step=0)

def plot_to_image(figure):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def image_grid():
    figure = plt.figure(figsize=(10, 10))
    for i in range(25):
        # Start next subplot.
        plt.subplot(5, 5, i + 1, title=class_names[y_train[i]])
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_train[i], cmap=plt.cm.binary)
    return figure


figure = image_grid()
with file_writer.as_default():
    tf.summary.image("Training data", plot_to_image(figure), step=0)


class ModelBaseline(Model):
    def __init__(self):
        super(ModelBaseline, self).__init__()
        self.flat = Flatten()
        self.d1 = Dense(45, activation='relu')
        self.drop = Dropout(0.3)
        self.d2 = Dense(35, activation='relu')

    def call(self, x):
        x = self.flat(x)
        x = self.d1(x)
        x = self.drop(x)
        return self.d2(x)


model = ModelBaseline()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_acc(labels, predictions)


@tf.function
def test_step(images, labels):
    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_acc(labels, predictions)


os.system("rm -rf logs/gradient_tape")

train_log_dir = 'logs/gradient_tape/basline_model/train'
test_log_dir = 'logs/gradient_tape/baseline_model/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

EPOCHS = 10

train_loss_results_baseline = []
train_accuracy_results_baseline = []
test_loss_results_baseline = []
test_accuracy_results_baseline = []

for epoch in range(EPOCHS):
    train_loss.reset_states()
    train_acc.reset_states()
    test_loss.reset_states()
    train_acc.reset_states()

    for images, labels in train_ds:
        train_step(images, labels)
    with train_summary_writer.as_default():
        tf.summary.scalar('loss', train_loss.result(), step=epoch)
        tf.summary.scalar('accuracy', train_acc.result(), step=epoch)

    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)
    with test_summary_writer.as_default():
        tf.summary.scalar('loss', test_loss.result(), step=epoch)
        tf.summary.scalar('accuracy', test_acc.result(), step=epoch)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch + 1,
                          train_loss.result(),
                          train_acc.result() * 100,
                          test_loss.result(),
                          test_acc.result() * 100))
    train_loss_results_baseline.append(train_loss.result())
    train_accuracy_results_baseline.append(train_acc.result() * 100)
    test_loss_results_baseline.append(test_loss.result())
    test_accuracy_results_baseline.append(test_acc.result() * 100)

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot([x for x in range(1, EPOCHS + 1)], train_loss_results_baseline, label='Training loss', marker='^')
ax1.plot([x for x in range(1, EPOCHS + 1)], test_loss_results_baseline, label='Testing loss', marker='^')
ax1.set_title("Loss Plot")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax2.plot([x for x in range(1, EPOCHS + 1)], train_accuracy_results_baseline, label='Training accuracy', marker='o')
ax2.plot([x for x in range(1, EPOCHS + 1)], test_accuracy_results_baseline, label='Testing accuracy', marker='o')
ax2.set_title("Accuracy Plot")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("%")
ax1.legend()
ax2.legend()
fig.tight_layout()

model.summary()


class ModelDenser(Model):
    def __init__(self):
        super(ModelDenser, self).__init__()
        self.flat = Flatten()
        self.d1 = Dense(45, activation='relu')
        self.drop = Dropout(0.3)
        self.d2 = Dense(35, activation='relu')
        self.d3 = Dense(23, activation='relu')
        self.d4 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.flat(x)
        x = self.d1(x)
        x = self.drop(x)
        x = self.d2(x)
        x = self.d3(x)
        return self.d4(x)


model = ModelDenser()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_acc(labels, predictions)


@tf.function
def test_step(images, labels):
    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_acc(labels, predictions)


train_log_dir = 'logs/gradient_tape/basline_dense/train'
test_log_dir = 'logs/gradient_tape/baseline_dense/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

EPOCHS = 10

train_loss_results_denser = []
train_accuracy_results_denser = []
test_loss_results_denser = []
test_accuracy_results_denser = []

for epoch in range(EPOCHS):
    train_loss.reset_states()
    train_acc.reset_states()
    test_loss.reset_states()
    train_acc.reset_states()

    for images, labels in train_ds:
        train_step(images, labels)
    with train_summary_writer.as_default():
        tf.summary.scalar('loss', train_loss.result(), step=epoch)
        tf.summary.scalar('accuracy', train_acc.result(), step=epoch)

    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)
    with test_summary_writer.as_default():
        tf.summary.scalar('loss', test_loss.result(), step=epoch)
        tf.summary.scalar('accuracy', test_acc.result(), step=epoch)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch + 1,
                          train_loss.result(),
                          train_acc.result() * 100,
                          test_loss.result(),
                          test_acc.result() * 100))
    train_loss_results_denser.append(train_loss.result())
    train_accuracy_results_denser.append(train_acc.result() * 100)
    test_loss_results_denser.append(test_loss.result())
    test_accuracy_results_denser.append(test_acc.result() * 100)

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot([x for x in range(1, EPOCHS + 1)], train_loss_results_denser, label='Training loss', marker='^')
ax1.plot([x for x in range(1, EPOCHS + 1)], test_loss_results_denser, label='Testing loss', marker='^')
ax1.set_title("Loss Plot")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax2.plot([x for x in range(1, EPOCHS + 1)], train_accuracy_results_denser, label='Training accuracy', marker='o')
ax2.plot([x for x in range(1, EPOCHS + 1)], test_accuracy_results_denser, label='Testing accuracy', marker='o')
ax2.set_title("Accuracy Plot")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("%")
ax1.legend()
ax2.legend()
fig.tight_layout()

model.summary()


class ModelConv(Model):
    def __init__(self):
        super(ModelConv, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10)

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)


model = ModelConv()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_acc(labels, predictions)


# In[32]:


@tf.function
def test_step(images, labels):
    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_acc(labels, predictions)


train_log_dir = 'logs/gradient_tape/convolutional/train'
test_log_dir = 'logs/gradient_tape/convolutional/test'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
test_summary_writer = tf.summary.create_file_writer(test_log_dir)

EPOCHS = 10

train_loss_results_conv = []
train_accuracy_results_conv = []
test_loss_results_conv = []
test_accuracy_results_conv = []

for epoch in range(EPOCHS):
    train_loss.reset_states()
    train_acc.reset_states()
    test_loss.reset_states()
    train_acc.reset_states()

    for images, labels in train_ds:
        train_step(images, labels)
    with train_summary_writer.as_default():
        tf.summary.scalar('loss', train_loss.result(), step=epoch)
        tf.summary.scalar('accuracy', train_acc.result(), step=epoch)

    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)
    with test_summary_writer.as_default():
        tf.summary.scalar('loss', test_loss.result(), step=epoch)
        tf.summary.scalar('accuracy', test_acc.result(), step=epoch)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch + 1,
                          train_loss.result(),
                          train_acc.result() * 100,
                          test_loss.result(),
                          test_acc.result() * 100))
    train_loss_results_conv.append(train_loss.result())
    train_accuracy_results_conv.append(train_acc.result() * 100)
    test_loss_results_conv.append(test_loss.result())
    test_accuracy_results_conv.append(test_acc.result() * 100)

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot([x for x in range(1, EPOCHS + 1)], train_loss_results_conv, label='Training loss', marker='^')
ax1.plot([x for x in range(1, EPOCHS + 1)], test_loss_results_conv, label='Testing loss', marker='^')
ax1.set_title("Loss Plot")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax2.plot([x for x in range(1, EPOCHS + 1)], train_accuracy_results_conv, label='Training accuracy', marker='o')
ax2.plot([x for x in range(1, EPOCHS + 1)], test_accuracy_results_conv, label='Testing accuracy', marker='o')
ax2.set_title("Accuracy Plot")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("%")
ax1.legend()
ax2.legend()
fig.tight_layout()

model.summary()

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot([x for x in range(1, EPOCHS + 1)], test_loss_results_baseline, label='Loss Model-1', marker='^')
ax1.plot([x for x in range(1, EPOCHS + 1)], test_loss_results_denser, label='Loss Model-2', marker='^')
ax1.plot([x for x in range(1, EPOCHS + 1)], test_loss_results_conv, label='Loss Model-3', marker='^')
ax1.set_title("Testing Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax2.plot([x for x in range(1, EPOCHS + 1)], test_accuracy_results_baseline, label='Accuracy Model-1', marker='o')
ax2.plot([x for x in range(1, EPOCHS + 1)], test_accuracy_results_denser, label='Accuracy Model-2', marker='o')
ax2.plot([x for x in range(1, EPOCHS + 1)], test_accuracy_results_conv, label='Accuracy Model-3', marker='o')
ax2.set_title("Testing Accuracy")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("%")
ax1.legend()
ax2.legend()
fig.tight_layout()


os.system("tensorboard --logdir logs/")
