# GuavaScript, a Guava Quality Predictor

## Project Description

GuavaScript is an AI project that aims to predict the quality of guavas based on their photos and traceable information. The project utilizes deep learning techniques, including the Un2net and VGG16 models, to preprocess images and extract features. It also incorporates a random forest regressor model to predict the taste attributes of guavas. By providing accurate information about guava quality, the project addresses the challenges of guava selection, reduces wastage of unsalable fruit, and contributes to the reduction of food waste in Taiwan. Additionally, it aims to mitigate the impact of China's boycott on Taiwanese fruits by increasing domestic sales.

## Getting Started (For User)

To use GuavaScript AI for guava quality prediction, follow these steps:

1. Download the GuavaScript HTML file from this repository.
2. Open the HTML file in a web browser.
3. Upload a guava photo and optionally enter a traceability code.
4. Click "Predict" to generate quality predictions.
5. View the predicted attributes like sweetness, sourness, crunchiness, hardness, and flavor.
6. Please allow some time for processing, and note that prediction accuracy may vary.

Enjoy using GuavaScript AI for accurate guava quality assessment! For any inquiries, consult the documentation or contact our support team.

## Getting Started (For Developer)

To use GuavaScript, follow these steps:

1. Install Python and required libraries:
   - Make sure Python is installed on your system.
   - Use the `pip install -r requirements.txt` command to install the necessary libraries.

2. Obtain the guava dataset:
   - Obtain a dataset containing guava photos and their attributes (sweetness, sourness, crunchiness, hardness, flavor).
   - Note: The dataset is included in the website.

3. Pretrained models or train them:
   - Either obtain pretrained models (Un2net, VGG16, random forest regressor) and place them in the appropriate directories.
   - Alternatively, train the models yourself using the provided code.

With Python, required libraries, guava dataset, and models, you're ready to use GuavaScript for guava quality prediction.

## File Structure 

The file structure of GuavaScript is organized as follows:

**GuavaScript**

- These folders contain guava photos used for preprocessing and analysis
  - Anita <dir>
  - Chloe
  - Jenny
  - Julia
- This folder stores the preprocessed guava photos with background removed
  - PhotoData
- The main code file containing the entire project code
  - GuavaScript.html
  - GuavaScript.ipynb
  - GuavaScript.py
  - requirements.txt
- These CSV files contain the dataset and guava attributes used for training and testing
  - GuavaScript.xlsx (raw data)
  - Photo_Tag.csv
  - Machine1.csv
  - Machine2.csv
  - Machine2_new.csv
  - Machine3.csv
  - CodeData.csv
- These files contain the testing data and the predicted guava attributes, respectively
  - TestData.csv
  - ResultData.csv
- These files store the trained models for day prediction and taste prediction, respectively
  - DayAI.h5
  - TasteAI.joblib

## Development

The development of the GuavaScript AI involves several key steps:

1. **Preprocessing the Dataset**:
   - The first step is to remove the background of the guava photos. This is achieved using the "rembg" tool, which helps separating the guava from its background. The background removal process is performed separately for different folders containing guava photos (Anita, Chloe, Jenny, and Julia).
   - After background removal, a photo tags file is created. This CSV file contains information about the guava photos, including the file name, origin, and day. It provides a structured representation of the guava photo dataset.
   - The training and testing data are obtained by loading and merging data from different CSV files that contain information about guava characteristics and features. The data is split into training and testing sets to facilitate model training and evaluation.

2. **Identifying the Day of the Guava**:
   - The AI employs the VGG16 model, a pre-trained convolutional neural network (CNN) architecture, to predict the day of the guava based on its photo. The VGG16 model is loaded using TensorFlow's Keras API.
   - The guava photos are preprocessed using techniques such as resizing and normalization to match the input requirements of the VGG16 model.
   - Data augmentation is applied using the `ImageDataGenerator` class from TensorFlow to generate augmented training data. Augmentation techniques include rotation, shifting, shearing, zooming, and flipping of images.
   - The VGG16 model is fine-tuned by adding custom top layers that include global average pooling, dense layers, and a linear activation layer. The model is compiled with appropriate optimizer and loss function settings.
   - The model is then trained using the augmented training data, and its performance is evaluated using validation data. The training process involves optimizing the model's weights based on the calculated loss.
   - The trained model is saved in the H5 file format for future use in predicting the day of guavas.

3. **Identifying the Quality of the Guava**:
   - The taste prediction model is built using the Random Forest Regressor algorithm from the scikit-learn library. The model predicts the taste attributes of guavas, including sweetness, sourness, crunchiness, hardness, and flavor, based on guava photos and other features.
   - The guava photos are preprocessed by converting them to grayscale, resizing them to a desired size, and extracting Histogram of Oriented Gradients (HOG) features. The HOG features capture the shape and texture information of the guava photos.
   - The taste prediction model combines the HOG features with other features like origin, average yield, rainfall, temperature, and various agricultural management frequencies. The combined features are preprocessed by scaling and encoding categorical variables.
   - The preprocessed features are used to train the Random Forest Regressor model. The model learns the relationships between the input features and the taste attributes through the ensemble of decision trees.
   - The trained taste prediction model is saved as a serialized file using the joblib library. This allows the model to be loaded and used for predicting guava taste attributes in the future.

4. **Using GuavaScript to Predict Testing Data**:
   - The testing data, which consists of guava photos and corresponding information, is loaded from a CSV file.
   - For each guava photo in the testing data, the day is predicted using the trained VGG16 model. The photo is preprocessed, and the VGG16 model is used to predict the day.
   - The predicted day and other relevant features are combined to create an input dataset for the taste prediction model. The features are preprocessed in the same manner as during model training.
   - The taste prediction model, which was trained using the Random Forest Regressor, is loaded. The model takes the preprocessed features as input and predicts the taste attributes of the guava.
   - The predicted taste attributes are added as new columns to the testing data. The results, including the guava photo, input features, and predicted taste attributes, are saved in a new CSV file.

5. **Becoming Gradio**:
   - The Gradio library is utilized to create an interactive user interface for the GuavaScript AI.
   - The AI models and relevant data are loaded, including the VGG16 model for predicting the day and the taste prediction model for predicting the taste attributes.
   - The Gradio interface allows users to upload a guava photo and select its origin (traceability code) as inputs. The interface provides real-time feedback and visualizations of the predicted taste attributes.
   - The user interface provides a user-friendly way for individuals to interact with the AI and obtain predictions of guava quality based on photos and traceable information.

These steps collectively enable the GuavaScript AI to preprocess the dataset, train models for day and taste prediction, make predictions on testing data, and create a user interface for easy interaction with the AI.

## Results

The results of GuavaScript provide predictions of the taste attributes (sweetness, sourness, crunchiness, hardness, and flavor) for guavas based on their photos and traceable information. By accurately predicting guava quality, the project aims to enable informed purchases, reduce wastage of unsalable fruit, address food waste issues in Taiwan, and mitigate the impact of China's boycott on Taiwanese fruits. The project demonstrates the potential of AI in the agricultural sector and promotes sustainable practices.

## Contributors

[List the contributors to your project and describe their roles and responsibilities.]

## Acknowledgments

[Thank any individuals or organizations who provided support or assistance during your project, including funding sources or data providers.]

## References

[List any references or resources that you used during your project, including data sources, analytical methods, and tools.]
