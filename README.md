# Sentiment Analysis Training Model
This repository contains code to train the sentiment analysis model using Sentiment140 dataset. The training phase contains data loading, preprocessing, and training with certain hyperparameters. Plus logging and tracking metrics and models into Neptune.

## Data
In this project, we use preprocessed and vectorized dataset from [preprocessing repository](https://github.com/emhaihsan/twitter-sentiment-preprocessing). This `.pickle` data contains a total of 1.6 million tweets collected using Twitter API. The data includes one hot encoded tweet with a label containing positive sentiment (1) and negative sentiment (0). For the first version, we just used 5% of training data to train a model to optimize the speed.

## Methods

Total of 3 models used and implemented to train a model for this project:
- **Logistic Regression**
- **Random Forest**
- **Support Vector Machine (SVM)**
  
Model Hyperparameters for these models configured using `config.yaml` file.

## Metrics
The metrics that are tracked and logged to Neptune are:
- Accuracy
- Precision
- Recall
- F1-Score

## Versioning
Model version set on `config.yaml` and tracked with Neptune for comparison and reproduction purposes. Every model version is logged with its configuration parameters and evaluation metrics. Every version of model logged with namespace included model and name version, ensure that every experiment can tracked and compared with clarity.


## Setup and Installation
Clone this repository to your local machine:

```bash
git clone https://github.com/emhaihsan/twitter-sentiment-train.git
cd twitter-sentiment-train
```


### Environment Variable
Create file `.env` in your directory and add your project name and API token from Neptune:
```env
NEPTUNE_PROJECT_NAME=your_neptune_project_name
NEPTUNE_API_TOKEN=your_neptune_api_token
```

### Data Directory
Make sure you already have .pickle file from the data preprocessing phase. You can just simply set up the data path on the script to the data directory from your own preprocessing process on your local machine. 

### Configuration

Edit `config.yaml` file to specify model configurations and hyperparameters:

```yaml
model_versions: "v1.0.0"
choosen_method: "Random Forest"
methods:
  - name: "Regresi Logistik"
    config:
      max_iter: 1000
  - name: "Random Forest"
    config:
      n_estimators: 100
      max_depth: null
  - name: "SVM"
    config:
      C: 1.0
      kernel: "rbf"
```

### Running Training Script
Running training script to train model and log into Neptune.

```bash
python train.py
```

### Results logged in Neptune.ai
Login to your [neptune](neptune.ai) account then you will see your recent running logged on your project page. You can deep dive into details for every experiment that you configured before.
![](https://github.com/emhaihsan/twitter-sentiment-train/tree/main/img/img1.png)
![](https://github.com/emhaihsan/twitter-sentiment-train/tree/main/img/img2.png)
![](https://github.com/emhaihsan/twitter-sentiment-train/tree/main/img/img3.png)
![](https://github.com/emhaihsan/twitter-sentiment-train/tree/main/img/img4.png)

## Additional Reference
- [Neptune Documentations](https://docs.neptune.ai/)
- [Preprocessing Repo](https://github.com/emhaihsan/twitter-sentiment-preprocessing)
