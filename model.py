import tensorflow as tf
from keras import layers, models, Input
from keras.applications.resnet50 import ResNet50
from keras.optimizers import Adam
import main
import pandas as pd
from  keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import os
import keras as keras



def get_images_and_labels(image_dict, label_dict):
    image_paths = []
    labels = []
    for study_id, paths in image_dict.items():
        if study_id in label_dict:
            label = label_dict[study_id]
            for path in paths:
                image_paths.append(path)
                # One-hot encode the labels
                labels.append(label)
    # Convert labels list to an array and apply one-hot encoding
    labels = to_categorical(labels, num_classes=2)  # Adjust num_classes as necessary
    return image_paths, labels




# Define a function for loading and preprocessing images
def load_and_preprocess_image(path):
    print("Type of path:", type(path)) 
    print(path)
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [128, 128])  # Resize to 128x128
    img /= 255.0  # Normalize pixel values
    return img

def prepare_dataset(image_paths, labels, batch_size=32):
    # Ensure image_paths are treated as strings
    image_paths = tf.constant(image_paths, dtype=tf.string)

    labels = tf.constant(labels)
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(lambda x, y: (load_and_preprocess_image(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def create_finetuned_model(input_shape, num_classes):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = True
    
    inputs = Input(shape=input_shape)
    x = base_model(inputs)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    model.summary()  # Print the model summary to check the architecture
    return model



if __name__ == '__main__':
    data = pd.read_csv('training_congruent.csv')
    label_dict = {'Pneumonia': 0, 'Pleural Effusion': 1, 'Pneumothorax': 2, "Atelectasis": 3}
    # Filtering for positive cases
    positive_cases_pn = set(data[data['Pneumonia'] == 1]['study_id'].astype(int))
    positive_cases_pe = set(data[(data['Pleural Effusion'] == 1)]['study_id'].astype(int))

    base_path = 'train'
    all_cases = [positive_cases_pn, positive_cases_pe]

    # Assuming base_path and positive_cases_pn are defined
    model_path = '≈model.keras'
    if os.path.exists(model_path):
        print("Loading the existing model.")
        model = keras.models.load_model(model_path)
    else:
        print("Creating a new model.")
        model = create_finetuned_model((128, 128, 3), 2)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    # Binary classification
    # Correct way to instantiate and use the Adam optimizer
    checkpoint_callback = ModelCheckpoint(model_path, save_best_only=True, monitor='val_loss', verbose=1)


    for i in range(len(all_cases)):
        print(i)
        image_dict, study_id_to_label = main.find_images_for_study_ids(base_path, all_cases[i], i)# Example empty dict, replace with actual dictionary from the function
        image_paths, labels = get_images_and_labels(image_dict, study_id_to_label)
        dataset = prepare_dataset(image_paths, labels) 
     
        # # Train the model
        steps_per_epoch = len(dataset) // 32
        model.fit(dataset, steps_per_epoch=steps_per_epoch, epochs=10, callbacks=[checkpoint_callback])

    model.save(model_path)  # Explicitly save the model after all training is complete
    print("Model saved at:", model_path)


  



