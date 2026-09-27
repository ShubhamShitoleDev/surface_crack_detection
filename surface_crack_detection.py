#################################################################################
#
#  Project Name  : Industrial Surface Crack Detection (CNN)
#  Description   : Loads a raw Positive/Negative crack image dataset, splits
#                   it into train/validation/test folders, trains a 4-block
#                   CNN with augmentation and callbacks for binary crack
#                   classification, evaluates with a confusion matrix and
#                   classification report, and supports single-image
#                   prediction.
#  Date          : 27-Sep-2026
#  Author        : Shubham Shitole
#
#################################################################################

import os
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.metrics import confusion_matrix, classification_report

Border = "-" * 50

ORIGINAL_DATASET = "CrackDataset"
POSITIVE_FOLDER = os.path.join(ORIGINAL_DATASET, "Positive")
NEGATIVE_FOLDER = os.path.join(ORIGINAL_DATASET, "Negative")
PROCESSED_DATASET = "Processed_CrackDataset"

IMAGE_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 15
RANDOM_SEED = 42


#################################################################################
#
# Function Name : set_seeds
# Description :   Fixes random seeds across Python, NumPy, and TensorFlow so
#                 the train/validation/test split and model initialization
#                 are reproducible across runs
# Date :          27-Sep-2026
# Author :        Shubham Shitole
#
#################################################################################

def set_seeds():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)


#################################################################################
#
# Function Name : check_folder
# Input :         folder_path
# Description :   Verifies a required dataset folder exists before
#                 proceeding, exiting with an error message if missing
# Date :          27-Sep-2026
# Author :        Shubham Shitole
#
#################################################################################

def check_folder(folder_path):
    if not os.path.exists(folder_path):
        print("ERROR: Folder not found:", folder_path)
        exit()


#################################################################################
#
# Function Name : get_image_files
# Input :         folder
# Description :   Lists all valid image files (jpg/jpeg/png/bmp/webp) inside
#                 a given folder
# Return Value :  List of image filenames
# Date :          27-Sep-2026
# Author :        Shubham Shitole
#
#################################################################################

def get_image_files(folder):
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    return [
        file for file in os.listdir(folder)
        if file.lower().endswith(valid_extensions)
    ]


#################################################################################
#
# Function Name : prepare_processed_folders
# Description :   Removes any previously generated train/validation/test
#                 split and creates a fresh directory structure for both
#                 classes (Crack, NoCrack)
# Date :          27-Sep-2026
# Author :        Shubham Shitole
#
#################################################################################

def prepare_processed_folders():
    folders = [
        "train/Crack",
        "train/NoCrack",
        "validation/Crack",
        "validation/NoCrack",
        "test/Crack",
        "test/NoCrack"
    ]

    if os.path.exists(PROCESSED_DATASET):
        shutil.rmtree(PROCESSED_DATASET)

    for folder in folders:
        os.makedirs(os.path.join(PROCESSED_DATASET, folder), exist_ok=True)


#################################################################################
#
# Function Name : split_and_copy_images
# Input :         source_folder, image_files, class_name
# Description :   Shuffles a class's image files and splits them 70/15/15
#                 into train/validation/test, copying each into the
#                 corresponding processed dataset subfolder
# Date :          27-Sep-2026
# Author :        Shubham Shitole
#
#################################################################################

def split_and_copy_images(source_folder, image_files, class_name):
    random.shuffle(image_files)

    total_images = len(image_files)
    train_count = int(total_images * 0.70)
    validation_count = int(total_images * 0.15)

    train_files = image_files[:train_count]
    validation_files = image_files[train_count:train_count + validation_count]
    test_files = image_files[train_count + validation_count:]

    split_data = {
        "train": train_files,
        "validation": validation_files,
        "test": test_files
    }

    for split_name, files in split_data.items():
        destination_folder = os.path.join(PROCESSED_DATASET, split_name, class_name)

        for file in files:
            source_path = os.path.join(source_folder, file)
            destination_path = os.path.join(destination_folder, file)

            if not os.path.exists(destination_path):
                shutil.copy(source_path, destination_path)

    print(f"{class_name} Images Split:")
    print("Training Images   :", len(train_files))
    print("Validation Images :", len(validation_files))
    print("Testing Images    :", len(test_files))
    print(Border)


#################################################################################
#
# Function Name : build_data_generators
# Description :   Builds ImageDataGenerators for train (with augmentation),
#                 validation, and test (normalization only) sets, using
#                 binary class mode since this is a 2-class problem
# Return Value :  train_data, validation_data, test_data
# Date :          27-Sep-2026
# Author :        Shubham Shitole
#
#################################################################################

def build_data_generators():
    train_dir = os.path.join(PROCESSED_DATASET, "train")
    validation_dir = os.path.join(PROCESSED_DATASET, "validation")
    test_dir = os.path.join(PROCESSED_DATASET, "test")

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=15,
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )

    validation_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_data = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="binary"
    )

    validation_data = validation_datagen.flow_from_directory(
        validation_dir,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="binary"
    )

    test_data = test_datagen.flow_from_directory(
        test_dir,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False
    )

    print("Class Indices:", train_data.class_indices)

    return train_data, validation_data, test_data


#################################################################################
#
# Function Name : show_sample_images
# Input :         train_data
# Description :   Displays a 2x3 grid of sample training images with their
#                 Crack/No Crack labels, as a visual sanity check on the
#                 data pipeline
# Date :          27-Sep-2026
# Author :        Shubham Shitole
#
#################################################################################

def show_sample_images(train_data):
    sample_images, sample_labels = next(train_data)

    plt.figure(figsize=(10, 6))

    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.imshow(sample_images[i])

        if sample_labels[i] == train_data.class_indices["Crack"]:
            plt.title("Crack")
        else:
            plt.title("No Crack")

        plt.axis("off")

    plt.suptitle("CNN Sample Training Images")
    plt.show()


#################################################################################
#
# Function Name : build_cnn_model
# Input :         input_shape
# Description :   Builds a 4-block CNN (32->64->128->256 filters) with
#                 BatchNormalization after each convolution, followed by two
#                 Dense layers with Dropout, ending in a single sigmoid
#                 output neuron for binary classification
# Return Value :  Compiled-ready Keras Sequential model
# Date :          27-Sep-2026
# Author :        Shubham Shitole
#
#################################################################################

def build_cnn_model(input_shape=(128, 128, 3)):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation="relu", input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))

    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.3))

    model.add(Dense(1, activation="sigmoid"))

    return model


#################################################################################
#
# Function Name : train_model
# Input :         model, train_data, validation_data
# Description :   Compiles the model with binary crossentropy loss and Adam
#                 optimizer, then trains it using three callbacks:
#                 EarlyStopping (restores best weights), ModelCheckpoint
#                 (saves the best validation-accuracy model separately), and
#                 ReduceLROnPlateau (shrinks the learning rate when
#                 validation loss plateaus)
# Return Value :  history object from model.fit
# Date :          27-Sep-2026
# Author :        Shubham Shitole
#
#################################################################################

def train_model(model, train_data, validation_data):
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True
    )

    checkpoint = ModelCheckpoint(
        "Best_Crack_Detection_Model.keras",
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        min_lr=0.00001,
        verbose=1
    )

    history = model.fit(
        train_data,
        epochs=EPOCHS,
        validation_data=validation_data,
        callbacks=[early_stop, checkpoint, reduce_lr]
    )

    return history


#################################################################################
#
# Function Name : plot_training_curves
# Input :         history
# Description :   Plots training vs validation accuracy and loss curves
#                 across epochs, to visually confirm the model is
#                 generalizing rather than overfitting
# Date :          27-Sep-2026
# Author :        Shubham Shitole
#
#################################################################################

def plot_training_curves(history):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("CNN Training vs Validation Accuracy")
    plt.legend()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("CNN Training vs Validation Loss")
    plt.legend()
    plt.show()


#################################################################################
#
# Function Name : evaluate_model
# Input :         model, test_data
# Description :   Evaluates the trained model on the held-out test set,
#                 then generates a confusion matrix and per-class
#                 classification report (precision/recall/F1) comparing
#                 predicted vs actual classes
# Date :          27-Sep-2026
# Author :        Shubham Shitole
#
#################################################################################

def evaluate_model(model, test_data):
    print(Border)
    print("Testing Model on Unseen Test Data")
    print(Border)

    test_loss, test_accuracy = model.evaluate(test_data)

    print("Test Loss     :", test_loss)
    print("Test Accuracy :", test_accuracy * 100)

    predictions = model.predict(test_data)
    predicted_classes = (predictions > 0.5).astype(int).reshape(-1)
    actual_classes = test_data.classes

    print("Confusion Matrix:")
    print(confusion_matrix(actual_classes, predicted_classes))

    print("Classification Report:")
    print(classification_report(
        actual_classes,
        predicted_classes,
        target_names=list(test_data.class_indices.keys())
    ))


#################################################################################
#
# Function Name : predict_single_image
# Input :         model, image_path, class_indices
# Description :   Loads a single image from disk, applies the same
#                 preprocessing used during training (resize, normalize,
#                 add batch dimension), predicts Crack/No Crack, and
#                 displays the image with the predicted label
# Date :          27-Sep-2026
# Author :        Shubham Shitole
#
#################################################################################

def predict_single_image(model, image_path, class_indices):
    if not os.path.exists(image_path):
        print("ERROR: Image not found:", image_path)
        return

    img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    prediction_value = prediction[0][0]

    crack_index = class_indices["Crack"]

    if crack_index == 1:
        final_result = "Crack Detected" if prediction_value > 0.5 else "No Crack"
    else:
        final_result = "No Crack" if prediction_value > 0.5 else "Crack Detected"

    print(Border)
    print("Single Image Prediction")
    print(Border)
    print("Image Path       :", image_path)
    print("Prediction Value :", prediction_value)
    print("Final Result     :", final_result)

    plt.imshow(load_img(image_path))
    plt.title(final_result)
    plt.axis("off")
    plt.show()


#################################################################################
#
# Function Name : main
# Description :   Runs the full pipeline: validate dataset, split into
#                 train/validation/test, build data generators, show sample
#                 images, build and train the CNN, plot training curves,
#                 evaluate on the test set, save the final model, and run a
#                 single-image prediction as a demo
# Date :          27-Sep-2026
# Author :        Shubham Shitole
#
#################################################################################

def main():
    print(Border)
    print("Industrial Surface Crack Detection using CNN")
    print(Border)

    set_seeds()

    print(Border)
    print("Step 1 : Validate Original Dataset")
    print(Border)

    check_folder(POSITIVE_FOLDER)
    check_folder(NEGATIVE_FOLDER)

    positive_images = get_image_files(POSITIVE_FOLDER)
    negative_images = get_image_files(NEGATIVE_FOLDER)

    print("Original Positive Images:", len(positive_images))
    print("Original Negative Images:", len(negative_images))

    if len(positive_images) == 0 or len(negative_images) == 0:
        print("ERROR: Positive or Negative folder contains no images.")
        exit()

    print(Border)
    print("Step 2 : Split Dataset into Train/Validation/Test")
    print(Border)

    prepare_processed_folders()
    split_and_copy_images(POSITIVE_FOLDER, positive_images, "Crack")
    split_and_copy_images(NEGATIVE_FOLDER, negative_images, "NoCrack")

    print(Border)
    print("Step 3 : Build Data Generators")
    print(Border)

    train_data, validation_data, test_data = build_data_generators()
    show_sample_images(train_data)

    print(Border)
    print("Step 4 : Build and Train CNN Model")
    print(Border)

    model = build_cnn_model(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    history = train_model(model, train_data, validation_data)

    print(Border)
    print("Step 5 : Plot Training Curves")
    print(Border)

    plot_training_curves(history)

    print(Border)
    print("Step 6 : Evaluate on Test Set")
    print(Border)

    evaluate_model(model, test_data)

    model.save("Final_Crack_Detection_Model.keras")
    print("Final model saved successfully.")

    print(Border)
    print("Step 7 : Single Image Prediction Demo")
    print(Border)

    sample_test_folder = os.path.join(PROCESSED_DATASET, "test", "Crack")
    test_images = get_image_files(sample_test_folder)

    if len(test_images) > 0:
        predict_single_image(
            model,
            os.path.join(sample_test_folder, test_images[0]),
            train_data.class_indices
        )
    else:
        print("No test image found for single image prediction.")


if __name__ == "__main__":
    main()
