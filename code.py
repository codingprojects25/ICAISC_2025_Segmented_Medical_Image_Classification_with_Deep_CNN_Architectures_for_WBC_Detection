import os
import random
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB3
import matplotlib.pyplot as plt
import seaborn as sns
import time

# ------------------------- Setup -------------------------
SEED = 42  # For reproducibility
SAVE_DIR = "models"  # Directory to save results
os.makedirs(SAVE_DIR, exist_ok=True)

DATA_DIR = {
    "train": "CNNs_Comparison\\ground_truth_data_aug\\train",
    "val": "CNNs_Comparison\\ground_truth_data_aug\\val",
    "test": "CNNs_Comparison\\ground_truth_data_aug\\test"
}

EPOCHS = 40
BATCH_SIZE = 32

# --------------------- Reproducibility --------------------
def set_random_seeds(seed=SEED):
    """Set random seeds"""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# -------------------- Data Preparation --------------------
def create_data_generators(img_size, batch_size, data_dir, color_mode):
    """
    Creates separate data generators for training, validation, and test datasets.
    """
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest"
    )

    val_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_gen = train_datagen.flow_from_directory(
        data_dir["train"],
        target_size=img_size,
        batch_size=batch_size,
        color_mode=color_mode,
        seed=SEED,
        class_mode="categorical"
    )
    val_gen = val_datagen.flow_from_directory(
        data_dir["val"],
        target_size=img_size,
        batch_size=batch_size,
        color_mode=color_mode,
        class_mode="categorical"
    )
    test_gen = test_datagen.flow_from_directory(
        data_dir["test"],
        target_size=img_size,
        batch_size=batch_size,
        color_mode=color_mode,
        class_mode="categorical",
        shuffle=False
    )

    return train_gen, val_gen, test_gen

# ------------------ Model Architectures ------------------
def alexnet(input_shape, num_classes):
    from tensorflow.keras import models, layers
    model = models.Sequential([
        layers.Conv2D(96, (11, 11), strides=4, activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((3, 3), strides=2),
        layers.Conv2D(256, (5, 5), padding='same', activation='relu'),
        layers.MaxPooling2D((3, 3), strides=2),
        layers.Conv2D(384, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(384, (3, 3), padding='same', activation='relu'),
        layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
        layers.MaxPooling2D((3, 3), strides=2),
        layers.Flatten(),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def vgg13(input_shape, num_classes):
    model = models.Sequential([

        layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def vgg16(input_shape, num_classes):
    model = models.Sequential([

        layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def vgg19(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(4096, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# SqueezeNet
def fire_module(x, squeeze_filters, expand_filters):
    x = layers.Conv2D(squeeze_filters, (1, 1), activation='relu')(x)
    left = layers.Conv2D(expand_filters, (1, 1), activation='relu')(x)
    right = layers.Conv2D(expand_filters, (3, 3), padding='same', activation='relu')(x)
    x = layers.concatenate([left, right])
    return x

def squeezenet(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(96, (7, 7), strides=(2, 2), padding="valid", activation="relu")(inputs)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    
    x = fire_module(x, squeeze_filters=16, expand_filters=64)
    x = fire_module(x, squeeze_filters=16, expand_filters=64)
    x = fire_module(x, squeeze_filters=32, expand_filters=128)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    
    x = fire_module(x, squeeze_filters=32, expand_filters=128)
    x = fire_module(x, squeeze_filters=48, expand_filters=192)
    x = fire_module(x, squeeze_filters=48, expand_filters=192)
    x = fire_module(x, squeeze_filters=64, expand_filters=256)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    
    x = fire_module(x, squeeze_filters=64, expand_filters=256)
    x = layers.Dropout(0.5)(x)
    x = layers.Conv2D(num_classes, (1, 1), activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Activation("softmax")(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

def efficientnet_b0(input_shape, num_classes):
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,  
        input_shape=input_shape,
        weights=None        
    )
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),  
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def efficientnet_b1(input_shape, num_classes):
    base_model = EfficientNetB1(include_top=False, input_shape=input_shape, weights=None)
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(1024, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def efficientnet_b3(input_shape, num_classes):
    base_model = EfficientNetB3(include_top=False, input_shape=input_shape, weights=None) 
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(1024, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def resnet_block(input_tensor, filters, strides=1):
    x = layers.Conv2D(filters, (3, 3), strides=strides, padding='same', use_bias=False)(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, (3, 3), strides=1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    if strides != 1 or input_tensor.shape[-1] != filters:
        input_tensor = layers.Conv2D(filters, (1, 1), strides=strides, padding='same', use_bias=False)(input_tensor)
        input_tensor = layers.BatchNormalization()(input_tensor)

    x = layers.add([x, input_tensor])
    x = layers.ReLU()(x)
    return x

def build_resnet18(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, (7, 7), strides=2, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)

    x = resnet_block(x, 64, strides=1)
    x = resnet_block(x, 64, strides=1)

    x = resnet_block(x, 128, strides=2)
    x = resnet_block(x, 128, strides=1)

    x = resnet_block(x, 256, strides=2)
    x = resnet_block(x, 256, strides=1)

    x = resnet_block(x, 512, strides=2)
    x = resnet_block(x, 512, strides=1)

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model

def resnet_block(input_tensor, filters, strides=1):
    x = layers.Conv2D(filters, (3, 3), strides=strides, padding='same', use_bias=False)(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, (3, 3), strides=1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    if strides != 1 or input_tensor.shape[-1] != filters:
        input_tensor = layers.Conv2D(filters, (1, 1), strides=strides, padding='same', use_bias=False)(input_tensor)
        input_tensor = layers.BatchNormalization()(input_tensor)

    x = layers.add([x, input_tensor])
    x = layers.ReLU()(x)
    return x

def build_resnet34(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, (7, 7), strides=2, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)

    for _ in range(3):
        x = resnet_block(x, 64, strides=1)

    x = resnet_block(x, 128, strides=2)
    for _ in range(3):
        x = resnet_block(x, 128, strides=1)

    x = resnet_block(x, 256, strides=2)
    for _ in range(5):
        x = resnet_block(x, 256, strides=1)

    x = resnet_block(x, 512, strides=2)
    for _ in range(2):
        x = resnet_block(x, 512, strides=1)

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model

def resnet_bottleneck_block(input_tensor, filters, strides=1):
    shortcut = input_tensor
    x = layers.Conv2D(filters, (1, 1), strides=strides, use_bias=False)(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, (3, 3), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters * 4, (1, 1), use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    if strides != 1 or input_tensor.shape[-1] != filters * 4:
        shortcut = layers.Conv2D(filters * 4, (1, 1), strides=strides, use_bias=False)(input_tensor)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.add([x, shortcut])
    x = layers.ReLU()(x)
    return x

def build_resnet50(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, (7, 7), strides=2, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((3, 3), strides=2, padding='same')(x)

    for _ in range(3):
        x = resnet_bottleneck_block(x, 64, strides=1)

    x = resnet_bottleneck_block(x, 128, strides=2)
    for _ in range(3):
        x = resnet_bottleneck_block(x, 128, strides=1)

    x = resnet_bottleneck_block(x, 256, strides=2)
    for _ in range(5):
        x = resnet_bottleneck_block(x, 256, strides=1)

    x = resnet_bottleneck_block(x, 512, strides=2)
    for _ in range(2):
        x = resnet_bottleneck_block(x, 512, strides=1)

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model

def build_densenet121(input_shape, num_classes):
    base_model = tf.keras.applications.DenseNet121(
        include_top=False,
        weights=None,
        input_shape=input_shape,
        pooling="avg"
    )
    x = base_model.output
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs=base_model.input, outputs=outputs)
    return model

def build_xception(input_shape, num_classes):
    base_model = tf.keras.applications.Xception(
        include_top=False,
        weights=None,
        input_shape=input_shape,
        pooling="avg"
    )
    x = base_model.output
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs=base_model.input, outputs=outputs)
    return model

def build_inceptionv3(input_shape, num_classes):
    base_model = tf.keras.applications.InceptionV3(
        include_top=False,
        weights=None,
        input_shape=input_shape,
        pooling="avg"
    )
    x = base_model.output
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs=base_model.input, outputs=outputs)
    return model


# -------------------- Utility Functions --------------------
def save_training_history(history, model_name, save_dir):
    """Save training history directly into a universal file."""
    combined_path = os.path.join(save_dir, "all_training_histories.txt")
    with open(combined_path, "a") as f:
        f.write(f"\n--- {model_name} Training History ---\n")
        f.write("Epoch\tTrain Loss\tTrain Accuracy\tVal Loss\tVal Accuracy\n")
        for epoch, (loss, acc, val_loss, val_acc) in enumerate(zip(
            history.history['loss'], history.history['accuracy'],
            history.history['val_loss'], history.history['val_accuracy']
        )):
            f.write(f"{epoch + 1}\t{loss:.4f}\t{acc:.4f}\t{val_loss:.4f}\t{val_acc:.4f}\n")
    print(f"Training history for {model_name} appended to {combined_path}")

def save_classification_report(test_gen, predictions, model_name, save_dir):
    """Save classification report directly into a universal file."""
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_gen.classes
    class_labels = list(test_gen.class_indices.keys())

    readable_report = classification_report(true_classes, predicted_classes, target_names=class_labels)
    combined_path = os.path.join(save_dir, "all_classification_reports.txt")
    with open(combined_path, "a") as f:
        f.write(f"\n--- {model_name} Classification Report ---\n")
        f.write(readable_report + "\n")
    print(f"Classification report for {model_name} appended to {combined_path}")

def save_model_results(results, save_dir):
    """Save all final model results to a single universal file."""
    results_path = os.path.join(save_dir, "all_model_results.txt")
    with open(results_path, "w") as f:
        f.write("Model\tTest Loss\tTest Accuracy\tTraining Time (s)\n")
        for model_name, metrics in results.items():
            f.write(f"{model_name}\t{metrics['Test Loss']:.4f}\t{metrics['Test Accuracy']:.4f}\t{metrics['Training Time']:.2f}\n")
    print(f"All results saved to {results_path}")

# -------------------- Training and Evaluation --------------------
def compile_and_train(model, train_gen, val_gen, model_name, epochs, save_dir):
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
                  loss="categorical_crossentropy", metrics=["accuracy"])

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ModelCheckpoint(filepath=os.path.join(save_dir, f"{model_name}_best.h5"), save_best_only=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5)
    ]

    # timer:
    start_time = time.time()
    history = model.fit(train_gen, validation_data=val_gen, epochs=epochs, callbacks=callbacks)
    end_time = time.time()

    training_time = end_time - start_time
    print(f"{model_name} Training Time: {training_time:.2f} seconds")
    return history, training_time

# -------------------- Evaluation and Visualization --------------------
def evaluate_and_visualize(model, history, test_gen, model_name, save_dir):
    """Evaluate the model and generate visualizations."""
    test_gen.reset()
    test_loss, test_accuracy = model.evaluate(test_gen, verbose=0)
    print(f"{model_name} Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # Accuracy and Loss Plots
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label=f'{model_name} Train Accuracy')
    plt.plot(history.history['val_accuracy'], label=f'{model_name} Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label=f'{model_name} Train Loss')
    plt.plot(history.history['val_loss'], label=f'{model_name} Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Confusion Matrix
    predictions = model.predict(test_gen)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_gen.classes
    class_labels = list(test_gen.class_indices.keys())

    conf_matrix = confusion_matrix(true_classes, predicted_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    save_classification_report(test_gen, predictions, model_name, save_dir)
    save_training_history(history, model_name, save_dir)

    return {"Test Loss": test_loss, "Test Accuracy": test_accuracy}

# -------------------- Aggregated Results --------------------
def aggregate_results(results):
    """Generate aggregated results for all models."""
    print("\nAggregated Results:")
    for model_name, metrics in results.items():
        print(f"{model_name}: Test Loss = {metrics['Test Loss']:.4f}, Test Accuracy = {metrics['Test Accuracy']:.4f}")

# -------------------- Main Workflow --------------------
if __name__ == "__main__":
    models_to_evaluate = {
        "AlexNet": {"model_func": alexnet, "input_size": (227, 227), "color_mode": "grayscale"},
        "VGG13": {"model_func": vgg13, "input_size": (224, 224), "color_mode": "grayscale"},
        "VGG16": {"model_func": vgg16, "input_size": (224, 224), "color_mode": "grayscale"},
        "VGG19": {"model_func": vgg19, "input_size": (224, 224), "color_mode": "grayscale"},
        "SqueezeNet": {"model_func": squeezenet, "input_size": (227, 227), "color_mode": "grayscale"},
        "EfficientNetB0": {"model_func": efficientnet_b0, "input_size": (224, 224), "color_mode": "grayscale"},
        "EfficientNetB1": {"model_func": efficientnet_b1, "input_size": (240, 240), "color_mode": "grayscale"},
        "EfficientNetB3": {"model_func": efficientnet_b3, "input_size": (300, 300), "color_mode": "grayscale"},
        "ResNet-18": {"model_func": build_resnet18, "input_size": (224, 224), "color_mode": "grayscale"},
        "ResNet-34": {"model_func": build_resnet34, "input_size": (224, 224), "color_mode": "grayscale"},
        "ResNet-50": {"model_func": build_resnet50, "input_size": (224, 224), "color_mode": "grayscale"},
        "DenseNet-121": {"model_func": build_densenet121, "input_size": (224, 224), "color_mode": "grayscale"},
        "Xception": {"model_func": build_xception, "input_size": (299, 299), "color_mode": "grayscale"},
        "InceptionV3": {"model_func": build_inceptionv3, "input_size": (299, 299), "color_mode": "grayscale"},
    }

    results = {}

    for model_name, config in models_to_evaluate.items():
        set_random_seeds(SEED) 
        train_gen, val_gen, test_gen = create_data_generators(
            config["input_size"], BATCH_SIZE, DATA_DIR, config["color_mode"]
        )

        model = config["model_func"](input_shape=(*config["input_size"], 1), num_classes=len(train_gen.class_indices))
        model.summary()

        history, training_time = compile_and_train(model, train_gen, val_gen, model_name, EPOCHS, SAVE_DIR)

        evaluation_results = evaluate_and_visualize(model, history, test_gen, model_name, SAVE_DIR)
        evaluation_results["Training Time"] = training_time
        results[model_name] = evaluation_results

    save_model_results(results, SAVE_DIR)
    aggregate_results(results)


# Aggregated visualization of the results:

# File path to the data
file_path = "CNNs_Comparison\\models\\all_model_results.txt"

# Number of epochs for each model (ensure this matches the models in the file)
epochs = [24, 24, 22, 25, 40, 18, 28, 29, 33, 25, 31, 16, 20, 25]

# Load the data from the file
df = pd.read_csv(file_path, sep="\t")

# Add the number of epochs to the dataframe
df["Epochs"] = epochs

# Calculate training time per epoch
df["Time Per Epoch (s)"] = df["Training Time (s)"] / df["Epochs"]

# Normalize data for colormapping
accuracy_normalized = (df["Test Accuracy"] - df["Test Accuracy"].min()) / (df["Test Accuracy"].max() - df["Test Accuracy"].min())
time_normalized = (df["Time Per Epoch (s)"] - df["Time Per Epoch (s)"].min()) / (df["Time Per Epoch (s)"].max() - df["Time Per Epoch (s)"].min())

# Plot 1: Accuracy plot
plt.figure(figsize=(12, 6))
sorted_acc = df.sort_values(by="Test Accuracy", ascending=False)
colors = cm.Blues(accuracy_normalized[sorted_acc.index])

plt.bar(sorted_acc["Model"], sorted_acc["Test Accuracy"] * 100, color=colors, edgecolor='black')
plt.xticks(rotation=45, ha='right', fontsize=11)
plt.ylabel("Accuracy (%)", fontsize=12)
plt.xlabel("Models", fontsize=12)
plt.title("Accuracy of Different Networks", fontsize=13)
plt.ylim(80, 100)

# Add annotations
for i, acc in enumerate(sorted_acc["Test Accuracy"] * 100):
    plt.text(i, acc + 0.8, f"{acc:.2f}%", ha='center', fontsize=10.7)

plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Plot 2: Training time per epoch
plt.figure(figsize=(12, 6))
sorted_time = df.sort_values(by="Time Per Epoch (s)")
colors = cm.Greens(1 - time_normalized[sorted_time.index])  # Reverse for lighter = slower

plt.bar(sorted_time["Model"], sorted_time["Time Per Epoch (s)"], color=colors, edgecolor='black')
plt.xticks(rotation=45, ha='right', fontsize=11)
plt.ylabel("Time Per Epoch (s)", fontsize=12)
plt.xlabel("Models", fontsize=12)
plt.title("Training Time Per Epoch", fontsize=13)
plt.ylim(0, 200)

# Add annotations
for i, time in enumerate(sorted_time["Time Per Epoch (s)"]):
    plt.text(i, time + 5, f"{time:.2f}s", ha='center', fontsize=10.7)

plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Plot 3: Accuracy vs. Training Speed
fig, ax = plt.subplots(figsize=(14, 7))
scatter_colors = cm.cool(accuracy_normalized)

# Scatter plot
ax.scatter(df["Time Per Epoch (s)"], df["Test Accuracy"] * 100, color=scatter_colors, s=100, edgecolor='black')
ax.set_xlabel("Time Per Epoch (s)", fontsize=12)
ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_title("Accuracy vs. Training Speed (Time Per Epoch)", fontsize=14)
ax.grid(axis='both', linestyle='--', alpha=0.7)

# Add model labels with lines
for i, row in df.iterrows():
    ax.annotate(row["Model"], (row["Time Per Epoch (s)"], row["Test Accuracy"] * 100),
                textcoords="offset points", xytext=(10, 10), ha='center', fontsize=10,
                arrowprops=dict(arrowstyle="->", color='black', lw=0.5))

plt.tight_layout()
plt.show()
