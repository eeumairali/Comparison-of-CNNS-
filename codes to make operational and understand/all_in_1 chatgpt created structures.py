import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize images to [0, 1]
y_train, y_test = to_categorical(y_train), to_categorical(y_test)  # One-hot encode labels

def train_and_evaluate(model, name, epochs=5, batch_size=64):
    """
    Train the given model and evaluate its accuracy on CIFAR-10 test data.

    Args:
        model: The Keras model to train.
        name: Name of the model (used for display).
        epochs: Number of epochs to train.
        batch_size: Batch size for training.

    Returns:
        history: Training history of the model.
        test_acc: Test accuracy of the model.
    """
    print(f"Training {name}...")
    history = model.fit(x_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_data=(x_test, y_test),
                        verbose=1)
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"{name} Test Accuracy: {test_acc:.4f}")
    return history, test_acc

def plot_results(histories, names):
    """
    Plot validation accuracy for multiple models.

    Args:
        histories: List of training histories for each model.
        names: List of model names.
    """
    plt.figure(figsize=(12, 6))
    for i, history in enumerate(histories):
        plt.plot(history.history['val_accuracy'], label=f"{names[i]} Val Accuracy")
    plt.title("Validation Accuracy Comparison")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

### MODEL DEFINITIONS ###

def create_lenet():
    """
    LeNet-5:
    - Early CNN architecture (1998) designed for digit recognition (e.g., MNIST).
    - Uses 2 convolutional layers followed by average pooling.
    - Fully connected layers for classification.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(6, (5, 5), activation='tanh', input_shape=(32, 32, 3)),
        tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(16, (5, 5), activation='tanh'),
        tf.keras.layers.AveragePooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120, activation='tanh'),
        tf.keras.layers.Dense(84, activation='tanh'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_alexnet():
    """
    AlexNet:
    - Proposed in 2012, won ImageNet challenge.
    - Deeper and wider network with ReLU activation.
    - Uses max pooling and dropout to reduce overfitting.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(96, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(384, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(384, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_vgg():
    """
    VGG:
    - Introduced in 2014 with a focus on simplicity and depth.
    - Uses small 3x3 filters and deeper layers compared to AlexNet.
    - Maintains uniform structure across blocks.
    """
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_resnet():
    """
    ResNet:
    - Introduced in 2015, introduces residual connections to solve vanishing gradient problems.
    - Adds skip connections to allow gradients to flow directly to earlier layers.
    """
    input_layer = tf.keras.layers.Input(shape=(32, 32, 3))
    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(input_layer)

    # Residual block
    shortcut = x
    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(x)
    x = tf.keras.layers.Add()([x, shortcut])  # Skip connection
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Flatten()(x)
    output = tf.keras.layers.Dense(10, activation='softmax')(x)
    model = tf.keras.Model(inputs=input_layer, outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

### TRAINING AND COMPARISON ###
models = [
    ("LeNet", create_lenet()),
    ("AlexNet", create_alexnet()),
    ("VGG", create_vgg()),
    ("ResNet", create_resnet())
]

histories = []
accuracies = []
names = []

for name, model in models:
    history, acc = train_and_evaluate(model, name)
    histories.append(history)
    accuracies.append(acc)
    names.append(name)

# Plot and Compare Results
plot_results(histories, names)
