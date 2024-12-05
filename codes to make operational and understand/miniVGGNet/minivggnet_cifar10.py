#importing libraries
import matplotlib
matplotlib.use("Agg")
#We need to set the matplotlib backend to Agg to indicate to create a non-interactive that will simply be saved to disk

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from dhruv.nn.conv import minivggnet
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
import argparse
from tensorflow.keras.layers import BatchNormalization

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="path to the output loss/accuracy plot")
args = vars(ap.parse_args())

print("[Info] loading CIFAR-10 data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

labelNames = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

print("[Info] compiling network...")
opt = SGD(learning_rate=0.01, decay=0.01 / 40, momentum=0.9, nesterov=True)
model = minivggnet.MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy",optimizer=opt, metrics=["accuracy"])

print("[Info] training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=64, epochs=40, verbose=1)

print("[Info] evaluating network...")
predictions = model.predict(testX, testY, batch_size=64)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=labelNames))
