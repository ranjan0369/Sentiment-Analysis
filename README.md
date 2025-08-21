# Sentiment-Analysis Using Logistic Regression Classifier

This is a beginner friendly ML project for those who want to understand the basic ML concepts and lifecycle. This project implements **Sentiment Analysis** using **Logistic Regression Classifier**. It loads a dataset of IMDB Reviews, trains the classifier and evaluates its performance.

---

## Table of Contents

- [Objective](#features)  
- [Dataset](#dataset)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Project Structure](#project-structure)  
- [Training](#training)  
- [Evaluation](#evaluation)  
- [Results](#results)
- [Outpur Sample](#output-sample)  
- [Future Improvements](#future-improvements)  

---

## Objective

The objective of the project is to automatically understand whether a given piece of text expresses positive or negative sentiment. For a given input, it classifies it as a positive or negative. For example:
- "That was a great movie" --> Positive
- "The movie was slow and boring" --> Negative 

---

## Dataset

The project uses a IMDB review dataset of 50000 reviews labeled as positive and negative review. The dataset contains two columns; Reviews and Sentiment. The number of positive and negative reviews are 50-50 meaning it is balanced.

---

## Data Preprocessing

This is the next step after data collection. It is important that our dataset is free from errors to generate a high quality model. The preprocessing for our dataset includes following steps:
- Checking for null values
- Removing the html tags if any
- Removing special characters in the text
- Converting all the characters in lowercase
- Removing stop words that that do not contribute in sentiment like a, the, for, and, is and so on.
- Lemmatizing/Stemming i.e. reducing the words in their root form like surprising to surprise

---

## Feature Extraction

After the pre processing step is complete we are left with a cleaned and ready to train dataset. Then the next step is feature extraction. The goal here is to extract sentiment heavy words as much as possible. Since the model does not understand text, so we need a way to convert the words into numeric value. For this we will use Scikit learn's TF-IDF Vectorizer.

- What is TF-IDF Vectorizer?
It stands for Term Frequency - Inverser Document Frequency Vectorizer. Bascially, it provides balance between how frequently a word is repeating and how much unique is it. This means a frequently appearing word like 'movie' is assigned a low score whereas other words like good, fantastic, boring, slow which are sentiment heavy are assigned high score so that at the end we know which are the most significant features and which are least.

## Training and Evaluation

After necessary features are extracter and converted into vectors, we are ready to feed into a classifier. Since, this is a binary classification problem, I implemented Logistic Regression Classifier in order to train our model.

- What is Logistic Regression Classifier?
Logistic Regression is a linear model commonly used for binary classification problems such as predicting positive or negative sentiment. The model works by estimating the probability that a given input (e.g., a text review) belongs to a particular class. It applies a sigmoid (logistic) function to map the weighted sum of input features into a probability score between 0 and 1:

P(y=1|x) = 1 / (1 + exp(-(w · x + b)))

where:
x = feature vector (e.g., TF-IDF representation of text)
w = learned weights
b = bias term
P(y=1∣x) = probability that the input belongs to the positive class

Based on a chosen threshold (commonly 0.5), the model assigns the input to either the positive or negative sentiment class.

In this project, text data was first converted into numerical feature representations i.e. TF-IDF, and then fed into the Logistic Regression classifier.

---

## Installation

1. **Clone the repository:**

```bash
git clone https://github.com/ranjan0369/Sentiment-Analysis.git
cd Sentiment-Analysis
```

2. **Install dependencies using Pipenv:**
```bash
pip install pipenv
pipenv install --dev
```

3. **Activate the Pipenv shell:**
```bash
pipenv shell
```
---

## Usage
**Train the model:**
```bash
python src/data_preprocess.py
python src/feature_extractore.py
python train_model.py
```

- Trains FisherFaceRecognizer on the dataset.

- Saves the trained model to result/fisher_model.xml.

**Evaluate the model:**
```bash
python src/evaluate_model.py
```

- Loads the saved model and evaluates accuracy on the validation set.

- Prints validation accuracy to the console.
---

## Project Structure
```bash
project/
├── src/
│   ├── train_model.py       # Script to train and save the model
│   └── evaluate_model.py    # Script to evaluate the model
├── facial-dataset/          # Training dataset
│   └── train/
│       ├── happy/
│       ├── sad/
│       └── ...
├── result/
│   └── fisher_model.xml     # Saved model
├── Pipfile
├── Pipfile.lock
└── README.md
```
---

## Training

- Uses cv2.face.FisherFaceRecognizer_create().

- Preprocesses all images as grayscale arrays.

- Automatically shuffles and splits dataset into training and validation sets.

---

## Evaluation

- Loads the trained model (fisher_model.xml).

- Runs predictions on the validation set.

- Computes accuracy:

```bash
Accuracy (%) = (Correct Predictions / Total Validation Images) * 100
```
---

## Results

Example output after running evaluate_model.py:

```bash
Validation Accuracy: 78.5%
```
Accuracy may vary depending on dataset quality and size.

---

## Output Sample

![Webcam Test](images/output.jpg)

---

## Future Improvements

- Replace FisherFace with CNN-based models (MobileNet, ResNet) for higher accuracy.

- Implement face alignment using MTCNN/dlib landmarks.

- Add data augmentation (rotations, flips, brightness variations).

- Create a web demo using Streamlit or Flask.

- Include explainability visualizations (Grad-CAM) for model predictions.

- Dockerize the full pipeline for reproducibility.
---

## License

MIT License – free to use and modify for personal, academic, or professional purposes.

---

## Contact

For questions or collaborations, reach out to Ranjan Shrestha: [ranjan.shrestha0369@gmail.com]
