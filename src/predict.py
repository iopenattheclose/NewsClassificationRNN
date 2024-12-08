from library import *
from RNN import *
from utils import *



def predict():
    model = load_object("artifacts/model.pkl")
    test_text_X = load_object("artifacts/test_text_X.pkl")
    validation = load_object("artifacts/validation.pkl")
    prediction = model.predict(test_text_X)
    prediction = prediction.argmax(axis=1)
    print(f"Accuracy: {accuracy_score(prediction, validation)}")

if __name__ == "__main__":
    predict()