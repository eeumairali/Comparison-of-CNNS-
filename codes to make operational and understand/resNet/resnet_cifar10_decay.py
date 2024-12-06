# Set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# Import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from pyimagesearch.nn.conv import ResNet
from pyimagesearch.callbacks import TrainingMonitor
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import plot_model
import numpy as np
import sys
import os

# Set a high recursion limit so Theano doesn’t complain
sys.setrecursionlimit(5000)

# Define the total number of epochs to train for along with the initial learning rate
NUM_EPOCHS = 100
INIT_LR = 1e-1

def poly_decay(epoch):
    # Initialize the maximum number of epochs, base learning rate, and power of the polynomial
    maxEpochs = NUM_EPOCHS
    baseLR = INIT_LR
    power = 1.0

    # Compute the new learning rate based on polynomial decay
    alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power

    # Return the new learning rate
    return alpha

# Configuration dictionary to replace argparse
config = {
    "model": "output/resnet_cifar10.h5",  # Path to save the trained model
    "output": "output/logs"  # Directory for logs and plots
}

# Ensure the output directory exists
os.makedirs(config["output"], exist_ok=True)

# Load the training and testing data, converting the images from integers to floats
print("[INFO] loading CIFAR-10 data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float")
testX = testX.astype("float")

# Apply mean subtraction to the data
mean = np.mean(trainX, axis=0)
trainX -= mean
testX -= mean

# Convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# Construct the image generator for data augmentation
aug = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest"
)

# Construct the set of callbacks
figPath = os.path.sep.join([config["output"], "training_plot.png"])
jsonPath = os.path.sep.join([config["output"], "training_history.json"])
callbacks = [
    TrainingMonitor(figPath, jsonPath=jsonPath),
    LearningRateScheduler(poly_decay)
]

# Initialize the optimizer and model (ResNet-56)
print("[INFO] compiling model...")
opt = SGD(learning_rate=INIT_LR, momentum=0.9)
model = ResNet.build(32, 32, 3, 10, (9, 9, 9),
                     (64, 64, 128, 256), reg=0.0005)

# Optionally, visualize and save the model architecture
plot_model(model, to_file="resnet.png", show_shapes=True)

# Compile the model
model.compile(
    loss="categorical_crossentropy",
    optimizer=opt,
    metrics=["accuracy"]
)

# Train the network
print("[INFO] training network...")
model.fit(
    aug.flow(trainX, trainY, batch_size=128),
    validation_data=(testX, testY),
    steps_per_epoch=len(trainX) // 128,
    epochs=10,
    callbacks=callbacks,
    verbose=1
)

# Save the network to disk
print("[INFO] serializing network...")
model.save(config["model"])
