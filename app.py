import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from sklearn.preprocessing import LabelEncoder
from astroquery.simbad import Simbad
from astroquery.skyview import SkyView  # Added this line to fix the import issue
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

# Load the CSV file from your local path
data = pd.read_csv('C:/Users/Jyoti/OneDrive/Desktop/Coding/SciRe 2024-25 STAHZAI/star_dataset.csv')

# Convert labels to numerical values using LabelEncoder
label_encoder = LabelEncoder()
data['Encoded Labels'] = label_encoder.fit_transform(data['Type (OBAFGKM Scale)'])

# Define data augmentation and preprocessing
def load_dataset(dataframe):
    image_paths = dataframe['Image'].apply(lambda x: os.path.join('C:/Users/Jyoti/OneDrive/Desktop/Coding/SciRe 2024-25 STAHZAI/images', x)).values
    labels = dataframe['Encoded Labels'].values
    return image_paths, labels

image_paths, labels = load_dataset(data)

# Create a TensorFlow dataset with augmentation
def preprocess_image(image_path, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])  # Adjust as needed
    return image, label

def prepare_for_training(image_paths, labels):
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(lambda x, y: preprocess_image(x, y), num_parallel_calls=AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=1000).batch(32).prefetch(buffer_size=AUTOTUNE)

    # Apply data augmentation
    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
    ])

    dataset = dataset.map(lambda x, y: (augment(x), y), num_parallel_calls=AUTOTUNE)
    return dataset

# Prepare training dataset
train_data = prepare_for_training(image_paths, labels)

# Define a simpler CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(224, 224, 3)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_data, epochs=10)

# Function to fetch and preprocess the Sirius image
def fetch_sirius_image():
    try:
        # Query SkyView for an image of Sirius
        images = SkyView.get_images(position='Sirius', survey='DSS', pixels='300,300')
        if images:
            fits_data = images[0][0]
            img_name = "sirius.jpg"
            
            # Convert FITS to JPG
            data = fits_data.data
            plt.figure()
            plt.imshow(data, cmap='gray', origin='lower')
            plt.axis('off')
            plt.savefig(img_name, bbox_inches='tight', pad_inches=0, dpi=150)
            plt.close()
            
            print(f"Image of Sirius saved as {img_name}")
            return img_name
        else:
            print("No image found for Sirius.")
            return None
    except Exception as e:
        print(f"Failed to fetch image for Sirius: {e}")
        return None

def predict_sirius_type():
    sirius_img_path = fetch_sirius_image()
    if sirius_img_path:
        # Load and preprocess the image
        img = image.load_img(sirius_img_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = tf.expand_dims(img_array, 0)  # Create batch axis

        # Predict using the model
        predictions = model.predict(img_array)
        predicted_type = tf.argmax(predictions[0]).numpy()

        # Decode the predicted label
        decoded_label = label_encoder.inverse_transform([predicted_type])
        print(f"Predicted Star Type for Sirius: {decoded_label[0]}")
    else:
        print("Could not fetch or predict the type for Sirius.")

# Run prediction for Sirius
predict_sirius_type()
