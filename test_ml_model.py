from ml_model import train_model

def test_train_model():
    model, accuracy = train_model()

    assert model is not None
    assert 0 <= accuracy <= 1

def test_model_prediction():
    model, accuracy = train_model()

    sample = [[5.1, 3.5, 1.4, 0.2]]
    prediction = model.predict(sample)

    assert len(prediction) == 1
    assert prediction[0] in [0, 1, 2]
