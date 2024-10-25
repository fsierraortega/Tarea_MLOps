from app import model_pred

new_data = {
  "Major_Axis_Length": 525,
  "Perimeter": 229,
  "Area": 85,
  "Convex_Area": 0.92,
  "Eccentricity": 0.57
}


def test_predict():
    prediction = model_pred(new_data)
    assert prediction == 1
    print(prediction)