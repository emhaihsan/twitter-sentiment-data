from dataloader import load_pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from utils import stratified_sample

train_data = load_pickle("train_data.pkl")
test_data = load_pickle("test_data.pkl")

train_x = train_data['X_train'].copy()
train_y = train_data['y_train'].copy()
test_x = test_data['X_test'].copy()
test_y = test_data['y_test'].copy()


model = LogisticRegression()
model.fit(train_x, train_y)

y_pred = model.predict(test_x)
print(classification_report(test_y.values, y_pred))