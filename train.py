import os
import yaml
import joblib
import neptune
from sklearn.svm import SVC
from dotenv import load_dotenv
from dataloader import load_pickle
from utils import stratified_sample, convert_params
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Load environment variables
load_dotenv()

# Neptune setup
project_name = os.getenv('NEPTUNE_PROJECT_NAME')
api_key = os.getenv('NEPTUNE_API_TOKEN')

# Initialize Neptune
run = neptune.init_run(project=project_name, api_token=api_key)

train_data = load_pickle("train_data.pkl")
test_data = load_pickle("test_data.pkl")

train_x = train_data['X_train']
train_y = train_data['y_train']
test_x = test_data['X_test']
test_y = test_data['y_test']

print(f"Shape before sampling: {train_x.shape, train_y.shape}")


train_x, train_y = stratified_sample(train_x, train_y, sample_size=0.05) # Use 5% data

# Load methods config and model version from YAML
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

model_version = config["model_versions"]  # Get model version from YAML

model_mapping = {
    "Logistic Regression": LogisticRegression,
    "Random Forest": RandomForestClassifier,
    "SVM": SVC,
}

# Load methods config and model version from YAML
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

model_version = config["model_versions"]  # Get model version from YAML

# Dictionary to map method names to sklearn models
model_mapping = {
    "Logistic Regression": LogisticRegression,
    "Random Forest": RandomForestClassifier,
    "SVM": SVC,
}

print(f"Shape after sampling: {train_x.shape, train_y.shape}")

for method in config["methods"]:
    model_name = method["name"]
    model_config = convert_params(method["config"])
    
    # Model namespace including version
    model_namespace = f"models/{model_name}/{model_version}"
    
    # Log model parameters to Neptune
    for param_name, param_value in model_config.items():
        run[f"{model_namespace}/parameters/{param_name}"] = param_value
    
    # Instantiate and train the model
    ModelClass = model_mapping[model_name]
    model = ModelClass(**model_config)
    model.fit(train_x, train_y)

    # Save the model to a file
    model_filename = f"{model_name}_{model_version}.joblib"
    joblib.dump(model, model_filename)

    # Log model artifact to Neptune
    run[f"{model_namespace}/artifact"].upload(model_filename)
    
    # Predict and evaluate
    predicted_y = model.predict(test_x)
    report = classification_report(test_y.values, predicted_y, output_dict=True)
    
    # Log classification report to Neptune
    run[f"{model_namespace}/classification_report"] = report
    
    print(f"Classification Report for {model_name} (Version {model_version}):\n{report}\n")
# Stop the Neptune run once all models are logged
run.stop()