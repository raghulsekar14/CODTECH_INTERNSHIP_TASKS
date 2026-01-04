CODTECH INTERNSHIP TASKS

COMPANY: CODTECH IT SOLUTIONS

NAME: RAGHUL SEKAR

INTERN ID: CTIS0280

DOMAIN: MACHINE LEARNING

DURATION: 4 WEEKS

MENTOR: NEELA SANTOSH

TASK_1: TITANIC SURVIVAL PREDICTION USING DECISION TREE CLASSIFICATION

✅ Project Description

This project focuses on building a machine learning classification model to predict the survival of passengers aboard the Titanic using the famous Kaggle Titanic Dataset. The goal is to analyze passenger data—such as class, gender, age, and fare—and determine the factors influencing survival.

A Decision Tree Classifier is implemented to train the model, visualize decision rules, and evaluate prediction performance. The project includes data preprocessing, feature engineering, model training, accuracy evaluation, and visualization of the decision tree.

Key steps include:

1. Loading and cleaning the Titanic dataset

2. Handling missing values (e.g., imputing missing ages)

3. Encoding categorical variables such as “Sex” using Label Encoding

4. Splitting the dataset into training and testing sets

5. Building a Decision Tree Classifier with controlled max-depth and leaf size

6. Visualizing the decision-making process using plot_tree and Graphviz

7. Generating readable if-else rules for interpretation

8. Making a sample prediction based on passenger details

This project helps understand how decision trees work, how features affect survival, and how to interpret model outputs. It serves as a foundational machine learning classification example.

OUTPUT:

<img width="696" height="416" alt="527607266-8b35bdc2-5b46-4701-b45e-9e57614d91e4" src="https://github.com/user-attachments/assets/6fd741c9-a21c-4950-a093-2bbf6ec9865c" />

TASK_2: SENTIMENT ANALYSIS USING NLP AND LOGISTIC REGRESSION

✅ Project Description

This project builds a Natural Language Processing (NLP) model to classify the sentiment of tweets using machine learning. The dataset contains tweets labeled with different sentiment categories, and the goal is to predict the sentiment based on text content.

A Logistic Regression (multiclass) model is used for classification after converting tweets into numerical features using TF-IDF Vectorization. The project includes text preprocessing, feature extraction, model training, evaluation, and prediction.

Key steps include:

1. Loading and preprocessing the tweet dataset

2. Cleaning text using lowercasing, regex filtering, and tokenization

3. Removing stopwords and applying stemming

4. Converting text into TF-IDF feature vectors

5. Splitting data into training and testing sets

6. Training a Logistic Regression classifier

7. Evaluating performance using accuracy, classification report, and confusion matrix

8. Predicting sentiment for custom sample tweets

This project demonstrates how to build an end-to-end NLP pipeline for sentiment classification and is a perfect learning example for beginners and intermediate learners interested in text analytics and machine learning.

OUTPUT:

<img width="933" height="673" alt="527613729-8f93f3a3-a826-4160-9500-b9246190f894" src="https://github.com/user-attachments/assets/367d51a7-ffa7-45f8-abb8-7e08792f1cda" />

TASK_3: IMAGE CLASSIFICATION USING CNN

✅ Project Description

This project builds a Convolutional Neural Network (CNN) using PyTorch to classify images from the CIFAR-10 dataset, which contains 10 classes such as airplanes, cars, birds, cats, and more. The model is trained using data augmentation, normalized image tensors, and evaluated on unseen test data.

The CNN includes convolution layers, ReLU activation, max-pooling, dropout, and fully connected layers to perform multi-class image classification.

Key Steps Include:

1. Loading CIFAR-10 dataset for training & testing

2. Applying data augmentation (random crop, flip)

3. Designing a CNN with 3 convolution layers + fully connected layers

4. Training using CrossEntropyLoss + Adam optimizer

5. Evaluating with accuracy, classification report, and confusion matrix

6. Saving the trained model as cnn_model.pth

This project demonstrates how deep learning models process image data and how CNNs can classify visual patterns effectively.

OUTPUT:

<img width="1178" height="900" alt="528133790-fb0ec4ea-ab73-419c-916d-10da3c6f07fd" src="https://github.com/user-attachments/assets/c55c3fe7-4a36-4907-bafb-da14f3bfa32a" />
<img width="1233" height="709" alt="528135933-a313a301-ffc0-42ad-80c4-d4feb5e2c4f1" src="https://github.com/user-attachments/assets/6b63e914-0513-4884-99ff-aeddc41432db" />

TASK_4: MOVIE RECOMMENDATION SYSTEM

✅ Project Description

This project builds a Movie Recommendation System using Singular Value Decomposition (SVD) on user–movie rating data. The goal is to predict missing ratings and recommend movies that a user is most likely to enjoy based on collaborative filtering.

The project includes two parts:

Model Building (Python) – training SVD on the user–item rating matrix and evaluating performance using RMSE and MAE.

Interactive Web App (Streamlit) – allowing users to select a User ID and receive personalized movie recommendations.

Key steps include:

1. Loading and preparing the MovieLens ratings dataset

2. Building a user–item matrix for collaborative filtering

3. Creating a train–test split by masking test ratings

4. Training Truncated SVD to learn latent features

5. Reconstructing predicted user–movie ratings

6. Evaluating the model using RMSE and MAE

Developing a Streamlit UI to:

Select a user

Generate top recommended movies

Display predicted ratings

This project demonstrates how matrix factorization works for recommendation systems and how SVD can effectively predict user preferences. It is ideal for beginners and intermediate learners exploring recommendation engines and collaborative filtering.

STREAMLIT APP: https://movierecomeng0714.streamlit.app/

OUTPUT:

<img width="1885" height="863" alt="527631098-5090fec0-9c50-4558-9e4e-b72bf69fd97a" src="https://github.com/user-attachments/assets/395a41f3-738b-4664-b73e-911275a3d00e" />
