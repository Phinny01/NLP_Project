import tensorflow as tf
from keras import layers, models, Input
from keras.applications.resnet50 import ResNet50
from keras.optimizers import Adam
import main
import pandas as pd
from  keras.utils import to_categorical


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
    labels = to_categorical(labels, num_classes=5)  # Adjust num_classes as necessary
    return image_paths, labels
def preprocess_data(input_data):
    # Example of ensuring the input is correctly typed
    if isinstance(input_data, tf.Tensor):
        print("Input is a tensor.")
        print("Tensor dtype:", input_data.dtype)
    else:
        print("Error: Input is not a tensor. Actual type:", type(input_data))

# Replace 'input_data' with the actual data you're processing



# Define a function for loading and preprocessing images
def load_and_preprocess_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [128, 128])  # Resize to 128x128
    img /= 255.0  # Normalize pixel values
    preprocess_data(img)
    return img

def prepare_dataset(image_paths, labels, batch_size=32):
    print("Checking image paths type...")
    preprocess_data(tf.constant(image_paths))
    print("Checking labels type...")
    preprocess_data(tf.constant(labels))
    image_paths = tf.constant(image_paths)
    labels = tf.constant(labels)
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(lambda x, y: (load_and_preprocess_image(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def create_finetuned_model(input_shape, num_classes):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False
    
    inputs = Input(shape=input_shape)
    x = base_model(inputs)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='sigmoid')(x)
    
    model = models.Model(inputs, outputs)
    model.summary()  # Print the model summary to check the architecture
    return model

data = pd.read_csv('training_congruent.csv')
label_dict = {'Pneumonia': 0, 'Pleural Effusion': 1, 'Pneumothorax': 2, "Atelectasis": 3, 'No Finding': 4}
# Filtering for positive cases
positive_cases_pn = set(data[data['Pneumonia'] == 1]['study_id'].astype(int))
positive_cases_pe = set(data[(data['Pleural Effusion'] == 1)]['study_id'].astype(int))
positive_cases_pt = set(data[(data['Pneumothorax'] == 1)]['study_id'].astype(int))
positive_cases_at= set(data[(data['Atelectasis'] == 1)]['study_id'].astype(int))
positive_cases_nf = set(data[(data['No Finding'] == 1)]['study_id'].astype(int))
base_path = 'train'
all_cases = [positive_cases_pn, positive_cases_pe, positive_cases_pt, positive_cases_at, positive_cases_nf ]

# Assuming base_path and positive_cases_pn are defined
# image_dict = find_images_for_study_ids(base_path, set(positive_cases_pn['study_id']))  # Needs definition
model = create_finetuned_model((128, 128, 3), 5)  # Binary classification
# Correct way to instantiate and use the Adam optimizer
optimizer = Adam(learning_rate=0.001)  # You can adjust the learning rate and other parameters
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])


for i in range(len(all_cases)):
    print(i)
    image_dict, study_id_to_label = main.find_images_for_study_ids(base_path, all_cases[i], i)# Example empty dict, replace with actual dictionary from the function
    image_paths, labels = get_images_and_labels(image_dict, study_id_to_label)
    dataset = prepare_dataset(image_paths, labels) 
    # # Train the model
    model.fit(dataset, epochs=10)
# Print to check if dictionaries contain data
print("Number of entries in image_dict:", len(image_dict))
print("Number of entries in study_id_to_label:", len(study_id_to_label))







# Define a function to fetch images and their labels based on study IDs



# Hypothetical function to illustrate checking the type


# Define a function to prepare a dataset from image paths and labels









# Check a sample from the dataset to ensure correct types and shapes
for image_batch, label_batch in dataset.take(1):
    print("Checking batch image type and shape...")
    preprocess_data(image_batch)
    print("Checking batch label type and shape...")
    preprocess_data(label_batch)
    
    
# Initialize the fine-tuned model

  



