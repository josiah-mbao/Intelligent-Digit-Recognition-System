import pandas as pd
import keras
from keras import layers
import numpy as np
import tkinter as tk
from tkinter import Canvas, Button
from PIL import Image, ImageDraw

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
    epochs=5,
    validation_data=(x_val, y_val),
)
print("Training Metrics:", history.history)

# Create Tkinter GUI for drawing
class DigitRecognizerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Digit Recognizer")

        self.canvas = Canvas(root, width=280, height=280, bg="white")
        self.canvas.pack()

        self.predict_button = Button(root, text="Predict", command=self.predict_digit)
        self.predict_button.pack()

        self.prediction_label = tk.Label(root, text="Prediction: ")
        self.prediction_label.pack()

        # Initialize drawing variables
        self.image = Image.new("L", (280, 280), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.prev_x = None
        self.prev_y = None
        self.canvas.bind("<B1-Motion>", self.paint)

    def paint(self, event):
        x, y = event.x, event.y
        if self.prev_x and self.prev_y:
            self.draw.line([self.prev_x, self.prev_y, x, y], fill="black", width=8)
            self.canvas.create_line(self.prev_x, self.prev_y, x, y, width=8, fill="black", capstyle=tk.ROUND, smooth=tk.TRUE)
        self.prev_x = x
        self.prev_y = y

    def predict_digit(self):
        # Get the drawn image from the canvas
        img = self.get_image()

        # Convert to PIL Image
        pil_image = Image.fromarray(img.astype('uint8'))

        # Resize the image to match the model's input shape
        resized_image = pil_image.resize((28, 28), resample=Image.LANCZOS)

        # Convert back to NumPy array
        img = np.asarray(resized_image)

        # Flattens the image
        img = img.flatten() / 255.0  # Normalize pixel values

        # Reshapes the image to match the model's input shape
        img = img.reshape((1, 784))

        # Makes a prediction
        prediction = model.predict(img)
        predicted_digit = np.argmax(prediction)
        print(predicted_digit)

        # This updates the prediction label
        self.prediction_label.config(text=f"Prediction: {predicted_digit}")

        # Clears the canvas after prediction
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.prev_x = None
        self.prev_y = None

    def get_image(self):
        # Converts the canvas to a NumPy array
        img = np.array(self.image)

        # Crops the image to remove borders
        img = img[72:352, 72:352]

        return img


if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerGUI(root)
    root.mainloop()
