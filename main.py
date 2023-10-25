import pandas as pd
import keras
from keras import layers


# Load training and test data from CSV files
train_data = pd.read_csv('mnist_train.csv')
test_data = pd.read_csv('mnist_test.csv')

# Extract features (pixels) and labels from the data
x_train = train_data.drop(columns=['0']).values
y_train = train_data['0'].values
x_test = test_data.drop(columns=['0']).values
y_test = test_data['0'].values


# Normalize the pixel values to the range [0, 1]
x_train = x_train / 255.0
x_test = x_test / 255.0

# This defines our model
inputs = keras.Input(shape=(784,), name="digits")
x = layers.Dense(64, activation="relu", name="dense_1")(inputs)
x = layers.Dense(64, activation="relu", name="dense_2")(x)
outputs = layers.Dense(10, activation="softmax", name="predictions")(x)

model = keras.Model(inputs=inputs, outputs=outputs)

# Compiles our model
model.compile(
    optimizer=keras.optimizers.RMSprop(),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()],
)

# Splits data for validation and training
x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]

# Trains our model
print("Fit model on training data")
history = model.fit(
    x_train,
    y_train,
    batch_size=64,
    epochs=2,
    validation_data=(x_val, y_val),
)

# Evaluates the model on the test data
print("Evaluate on test data")
results = model.evaluate(x_test, y_test, batch_size=128)
print("Test loss, test accuracy:", results)


