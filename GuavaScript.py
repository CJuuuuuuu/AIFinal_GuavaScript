#%% [markdown]
# # **GuavaScript**
  
# %% [markdown]
# ## **Import Modules**

# %%
import os
import tempfile
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import decode_predictions
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from joblib import dump, load
from skimage.feature import hog
from pathlib import Path
from PIL import Image
from rembg import remove, new_session
import gradio as gr

# %% [markdown]
# ## **Preprocess the dataset**

# %% [markdown]
# ### **Remove the background of guava photos**

# %% [markdown]
# #### **Anita**

# %%
session = new_session()

# %%
# Get the absolute paths of the Anita folder
anita_folder = os.path.abspath('Anita')
for folder in Path(anita_folder).glob('*'):
    k = 1
    j = 1
    FileList = []

    for file in Path(folder).glob('*.jpg'):
        FileList.append(file)
    FileList.sort()

    for file in FileList:
        input_path = str(file)
        output_path = str(file.parent.parent.parent / "PhotoData" / (file.parent.parent.stem + '_' + file.parent.stem + '_' + str(k).zfill(2) + '_' + str(j).zfill(2) + ".png"))
        print(input_path)
        print(output_path)
        if int(j) % 3 == 0:
            k += 1
            j = 1
        else:
            j += 1

        with open(input_path, 'rb') as i:
            with open(output_path, 'wb') as o:
                input_data = i.read()
                output = remove(input_data, session=session)
                o.write(output)

# %% [markdown]
# #### **Chloe**

# %%
session = new_session()

# %%
# Get the absolute paths of the Chloe folder
chloe_folder = os.path.abspath('Chloe')
for folder in Path(chloe_folder).glob('*'):
    k = 1
    j = 1
    FileList = []

    for file in Path(folder).glob('*.jpg'):
        FileList.append(file)
    FileList.sort()

    for file in FileList:
        input_path = str(file)
        output_path = str(file.parent.parent.parent / "PhotoData" / (file.stem + ".png"))
        print(input_path)
        print(output_path)
        if int(j) % 3 == 0:
            k += 1
            j = 1
        else:
            j += 1

        with open(input_path, 'rb') as i:
            with open(output_path, 'wb') as o:
                input_data = i.read()
                output = remove(input_data, session=session)
                o.write(output)

# %% [markdown]
# #### **Jenny**

# %%
session = new_session()

# %%
# Get the absolute paths of the Jenny folder
jenny_folder = os.path.abspath('Jenny')
for folder in Path(jenny_folder).glob('*'):
    k = 1
    j = 1
    FileList = []

    for file in Path(folder).glob('*.jpg'):
        FileList.append(file)
    FileList.sort()

    for file in FileList:
        input_path = str(file)
        output_path = str(file.parent.parent.parent / "PhotoData" / (file.parent.parent.stem + '_' + file.parent.stem + '_' + file.stem + ".png"))
        print(input_path)
        print(output_path)
        if int(j) % 3 == 0:
            k += 1
            j = 1
        else:
            j += 1

        with open(input_path, 'rb') as i:
            with open(output_path, 'wb') as o:
                input_data = i.read()
                output = remove(input_data, session=session)
                o.write(output)

# %% [markdown]
# #### **Julia**

# %%
session = new_session()

# %%
# Get the absolute paths of the Julia folder
julia_folder = os.path.abspath('Julia')
for folder in Path(julia_folder).glob('*'):
    k = 1
    j = 1
    FileList = []

    for file in Path(folder).glob('*.jpg'):
        FileList.append(file)
    FileList.sort()

    for file in FileList:
        input_path = str(file)
        output_path = str(file.parent.parent.parent / "PhotoData" / (file.parent.parent.stem + '_' + file.parent.stem + '_' + str(k).zfill(2) + '_' + str(j).zfill(2) + ".png"))
        print(input_path)
        print(output_path)
        if int(j) % 4 == 0:
            k += 1
            j = 1
        else:
            j += 1

        with open(input_path, 'rb') as i:
            with open(output_path, 'wb') as o:
                input_data = i.read()
                output = remove(input_data, session=session)
                o.write(output)

# %% [markdown]
# ### **Create Photo Tags File**

# %%
import csv

# %%
# Open the file in write mode
with open('Photo_Tag.csv', 'w') as f:
    # Create the csv writer
    writer = csv.writer(f)
    header = ['FileName', 'Origin', 'Day']
    writer.writerow(header)

    FileList = []
    # Get the absolute paths of the PhotoData folder
    photo_folder = os.path.abspath('PhotoData')
    for file in Path(photo_folder).glob('*.png'):
        FileList.append(file)
    FileList.sort()

    for file in FileList:
        FileName = str(file.stem)
        Day = FileName.split('_')
        Day = str(Day[-2])
        row = [FileName + '.png', ' ', Day]
        # Write a row to the csv file
        writer.writerow(row)

# %%
print('Done!')

# %% [markdown]
# ### **Get Training Data**

# %%
# Load the CSV files into pandas DataFrames
df1 = pd.read_csv('Machine1.csv')
df2 = pd.read_csv('Machine2.csv')

# %%
# Specify the columns to join on
join_columns = ['Origin', 'Day', '灌溉/澆水頻率', '施肥頻率', '田間栽培管理/耕地管理頻率', '有害生物防治頻率']

# %%
# Perform the join operation
merged_df = pd.merge(df1, df2, on=join_columns)

# %%
# Save the merged DataFrame to a new CSV file
merged_df.to_csv('Machine2_new.csv', index=False)

# %% [markdown]
# ### **Get Testing Data**

# %%
# Load the first CSV file
df1 = pd.read_csv('Machine1.csv')

# %%
# Load the second CSV file
df2 = pd.read_csv('Machine3.csv')

# %%
# Column to check for matching values
column_to_check = 'File Name'  # Replace 'column_name' with the actual column name

# %%
# Find the matching values
matching_values = df1[df1[column_to_check].isin(df2[column_to_check])][column_to_check]

# %%
# Drop the lines with matching values from the first DataFrame
df1 = df1[~df1[column_to_check].isin(matching_values)]

# %%
# Save the remaining lines to a new CSV file
df1.to_csv('TestData.csv', index=False)

# %% [markdown]
# ## **Use guava info to predict guava quality**

# %% [markdown]
# ### **Identify the day of the guava**

# %%
# Read the CSV file
dataset = pd.read_csv('Machine1.csv')

# %%
# Access the columns
file_names = dataset['File Name']
#origin_attributes = dataset['Origin']
day_labels = dataset['Day']
#yield_attributes = dataset['平均產量(每公頃)']
#rain_attributes = dataset['年雨量']
#temper_attributes = dataset['年均溫']
#water_attributes = dataset['灌溉/澆水頻率']
#fertilize_attributes = dataset['施肥頻率']
#manage_attributes = dataset['田間栽培管理/耕地管理頻率']
#creature_attributes = dataset['有害生物防治頻率']

# %%
# Load the guava photos into memory
photo_directory = 'PhotoData'

# %%
guava_photos = []

# %%
for file_name in file_names:
    photo_path = f"{photo_directory}/{file_name}"
    photo = tf.keras.preprocessing.image.load_img(photo_path, target_size=(224, 224))
    photo = tf.keras.preprocessing.image.img_to_array(photo)
    guava_photos.append(photo)

# %%
# Convert the lists to numpy arrays
photo = np.array(guava_photos)
day = np.array(day_labels)

# %%
# Split the data into training and testing sets
X_train_img, X_test_img, y_train, y_test = train_test_split(photo, day, test_size=0.2, random_state=42)

# %%
# Create an ImageDataGenerator for data augmentation
image_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)

# %%
# Generate augmented training data
train_data_augmented = image_datagen.flow(X_train_img, y_train, batch_size=32)

# %%
# Use original validation data for evaluation (no augmentation)
validation_data = image_datagen.flow(X_test_img, y_test, batch_size=32)

# %%
# Create the VGG16 model
base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# %%
# Add custom top layers
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(64, activation='relu')(x)
predictions = tf.keras.layers.Dense(1, activation='linear')(x)

# %%
model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)

# %%
# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# %%
# Train the model with augmented data
model.fit(train_data_augmented, epochs=10, validation_data=validation_data)

# %%
# Evaluate the model
loss = model.evaluate(train_data_augmented)
print("Test Loss:", loss)

# %%
# Save the model as an H5 file
model.save("DayAI.h5")

# %%
# Get the predicted ages
y_pred = model.predict(validation_data)

# %%
# Calculate R-squared value
r2 = r2_score(y_test, y_pred)
print('R-squared:', r2)

# %% [markdown]
# ### **Identify the quality of the guava**

# %%
# Load the data from the CSV file
data = pd.read_csv('Machine3.csv')

# %%
# Load the guava photos into memory
photo_directory = 'PhotoData'

# %%
# Load and preprocess the images
images = []
for image_file in data['File Name']:
    image_file = f"{photo_directory}/{image_file}"
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (128, 128))  # Adjust the size as needed
    features = hog(image)  # Extract image features using HOG (Histogram of Oriented Gradients)
    images.append(features)

# %%
X_images = pd.DataFrame(images)

# %%
# Combine image features with other features
X = pd.concat([X_images, data.drop(['Sweetness', 'Sourness', 'Crunchiness', 'Hardness', 'Flavor', 'File Name'], axis=1)], axis=1)
y = data[['Sweetness', 'Sourness', 'Crunchiness', 'Hardness', 'Flavor']]

# %%
# Convert column names to strings
X.columns = X.columns.astype(str)

# %%
# Encode categorical variables if any
encoder = LabelEncoder()
X['Origin'] = encoder.fit_transform(X['Origin'])

# %%
# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X.iloc[:, -9:])

# %%
# Assign weights to features
feature_weights = [1.5, 3, 1.5, 1.5, 1.5, 2, 2, 2, 2]  # Adjust the weights as desired
X_scaled_weighted = X_scaled * feature_weights

# %%
# Select the non-scaled features
X_non_scaled = 1000000 * X.iloc[:, :-9]

# %%
# Combine the scaled and non-scaled features
X_combined = pd.concat([X_non_scaled, pd.DataFrame(X_scaled_weighted, columns=X.columns[-9:])], axis=1)

# %%
# Scale the combined features
X = scaler.fit_transform(X_combined)

# %%
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
# Train a random forest regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# %%
# Make predictions on the test set
y_pred = model.predict(X_test)

# %%
# Evaluate the model
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)

# %%
# Save the model as a file
dump(model, "TasteAI.joblib")

# %% [markdown]
# ## **Use GuavaScript to predict testing data**

# %%
# Load the data from the CSV file
new_data = pd.read_csv('TestData.csv')

# %%
# Load the guava photos into memory
photo_directory = 'PhotoData'

# %%
# Load and preprocess the images
new_images = []
for image_file in new_data['File Name']:
    image_file = f"{photo_directory}/{image_file}"
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (128, 128))  # Adjust the size as needed
    features = hog(image)  # Extract image features using HOG (Histogram of Oriented Gradients)
    new_images.append(features)

# %%
X_images_new = pd.DataFrame(new_images)

# %%
# Combine image features with other features
X_new = pd.concat([X_images_new, new_data.drop(['File Name'], axis=1)], axis=1)

# %%
# Convert column names to strings
X_new.columns = X_new.columns.astype(str)

# %%
# Encode categorical variables if any
encoder = LabelEncoder()
X_new['Origin'] = encoder.fit_transform(X_new['Origin'])

# %%
# Scale the features
scaler = StandardScaler()
X_new_scaled = scaler.fit_transform(X_new.iloc[:, -9:])

# %%
# Assign weights to features
feature_weights = [1.5, 3, 1.5, 1.5, 1.5, 2, 2, 2, 2]  # Adjust the weights as desired
X_new_scaled_weighted = X_new_scaled * feature_weights

# %%
# Select the non-scaled features
X_new_non_scaled = 1000000 * X_new.iloc[:, :-9]

# %%
# Combine the scaled and non-scaled features
X_new_combined = pd.concat([X_new_non_scaled, pd.DataFrame(X_new_scaled_weighted, columns=X_new.columns[-9:])], axis=1)

# %%
# Scale the combined features
X_new = scaler.fit_transform(X_new_combined)

# %%
# Make predictions on the test set
y_new = model.predict(X_new)

# %%
# Convert predictions to integers
y_new = y_new.astype(int)

# %%
# Add the predicted values as new columns to the DataFrame
new_data['Sweetness'] = y_new[:, 0]
new_data['Sourness'] = y_new[:, 1]
new_data['Crunchiness'] = y_new[:, 2]
new_data['Hardness'] = y_new[:, 3]
new_data['Flavor'] = y_new[:, 4]

# %%
# Save the results to a new CSV file
new_data.to_csv('ResultData.csv', index=False)

# %% [markdown]
# ## **Become Gradio**

# %% [markdown]
# ### **Create Gradio**

# %%
# Load the AI models and necessary data
DayAI = tf.keras.models.load_model("DayAI.h5")
TasteAI = load("TasteAI.joblib")
guava_data = pd.read_csv("CodeData.csv")
origins = guava_data['Traceability Code'].unique().tolist()

# %%
# Function to preprocess the input photo
def preprocess_image(image):
    # Resize the image to the desired input size
    image = image.resize((224, 224))
    # Convert the image to a NumPy array
    image_array = np.array(image)
    # Create an ImageDataGenerator for data augmentation
    image_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )
    # Expand dimensions to match the expected input shape of the model
    image_array = np.expand_dims(image_array, axis=0)
    # Generate augmented image data
    image_array_augmented = image_datagen.flow(image_array, batch_size=32)
    # Access the augmented image data
    augmented_image_batch = next(image_array_augmented)
    # The augmented images are stored in augmented_image_batch
    augmented_image = augmented_image_batch
    return augmented_image

# %%
# Function to predict the day using the first AI model
def predict_day(image):
    # Preprocess the image
    image_array = preprocess_image(image)
    # Predict the day
    day_prediction = DayAI.predict(image_array)
    return int(day_prediction)

# %%
# Function to predict the taste using the second AI model
def predict_taste(image, day, code):
    image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (128, 128))  # Adjust the size as needed
    features = hog(image)  # Extract image features using HOG (Histogram of Oriented Gradients)
    features = np.array(features)
    features = pd.DataFrame([features])
    # Fetch the data for the given code from guava_data
    guava_info = guava_data.loc[guava_data['Traceability Code'] == int(code)]
    if guava_info.empty:
        # Handle the case when no matching data is found
        return "No data found for the given code"
    guava_info = guava_info.iloc[0]
    # Extract the required information
    origin_data = [guava_info['Origin'], guava_info['平均產量(每公頃)'], guava_info['年雨量'], guava_info['年均溫'], guava_info['灌溉/澆水頻率'], guava_info['施肥頻率'], guava_info['田間栽培管理/耕地管理頻率'], guava_info['有害生物防治頻率']]
    # Combine day and origin data
    input_data = np.array([origin_data[:1] + [day] + origin_data[1:]])
    input_data = pd.DataFrame(input_data)
    # Encode categorical variables if any
    encoder = LabelEncoder()
    input_data[0] = encoder.fit_transform(input_data[0])

    # Combine image features with other features
    input_data = pd.concat([features, input_data], axis=1)

    # Convert column names to strings
    input_data.rename(columns={input_data.columns[-9]: 'Origin',
                                        input_data.columns[-8]: 'Day',
                                        input_data.columns[-7]: '平均產量(每公頃)',
                                        input_data.columns[-6]: '年雨量',
                                        input_data.columns[-5]: '年均溫',
                                        input_data.columns[-4]: '灌溉/澆水頻率',
                                        input_data.columns[-3]: '施肥頻率',
                                        input_data.columns[-2]: '田間栽培管理/耕地管理頻率',
                                        input_data.columns[-1]: '有害生物防治頻率'}, inplace=True)
    input_data.columns = input_data.columns.astype(str)

    # Scale the features
    scaler = StandardScaler()
    input_data_scaled = scaler.fit_transform(input_data.iloc[:, -9:])

    # Assign weights to features
    feature_weights = [1.5, 3, 1.5, 1.5, 1.5, 2, 2, 2, 2]  # Adjust the weights as desired
    input_data_scaled_weighted = input_data_scaled * feature_weights

    # Select the non-scaled features
    input_data_non_scaled = 1000000 * input_data.iloc[:, :-9]

    # Combine the scaled and non-scaled features
    input_data_combined = pd.concat([input_data_non_scaled, pd.DataFrame(input_data_scaled_weighted, columns=input_data.columns[-9:])], axis=1)

    # Scale the combined features
    input_data = scaler.fit_transform(input_data_combined)

    # Predict the taste
    taste_prediction = TasteAI.predict(input_data)
    # Convert predictions to integers
    taste_prediction = taste_prediction.astype(int)
    return taste_prediction

# %%
# Define the Gradio interface
def gradio_interface(image, code):
    # Preprocess the image
    image = Image.fromarray(image.astype('uint8'), 'RGB')
    # Predict the day
    day = predict_day(image)
    # Predict the taste
    # Save the image array to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp:
        temp_path = temp.name
        image.save(temp_path)
    taste = predict_taste(temp_path, day, code)
    # Delete the temporary file
    os.remove(temp_path)
    # Return the predicted taste
    return [taste[0, 0], taste[0, 1], taste[0, 2], taste[0, 3], taste[0, 4]]

# %%
# Set up the Gradio interface
origin_dropdown = gr.inputs.Dropdown(choices=origins, label="Traceability Code")
iface = gr.Interface(
    fn=gradio_interface,
    inputs=["image", origin_dropdown],
    outputs=["text", "text", "text", "text", "text"],
    title="Guava Quality Predictor",
    description="Predict the Quality of a guava based on its photo and traceable information. Output0, output1, output2, output3, and output4 correspond to Sweetness, Sourness, Crunchiness, Hardness, and Flavor, respectively."
)

# %%
# Launch the Gradio interface
iface.launch()

