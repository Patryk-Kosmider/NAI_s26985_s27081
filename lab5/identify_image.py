import argparse
import numpy as np
import tensorflow as tf
from PIL import Image

CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def preprocess_image(image_name):
    img = Image.open(image_name).convert("L")
    img = img.resize((28, 28))
    img = np.array(img).astype("float32")
    img = 255 - img
    img = img / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)

    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str)
    parser.add_argument("--model", type=str, default="conv_model.keras")
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model)
    processed_image = preprocess_image(args.image)

    prediction = model.predict(processed_image, verbose=0)

    print(f"Prediction: {CLASS_NAMES[np.argmax(prediction)]}")


if __name__ == "__main__":
    main()