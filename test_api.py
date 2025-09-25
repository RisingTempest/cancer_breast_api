import requests

url = "http://127.0.0.1:5001/predict"

examples = [
    [14.5, 20.1, 90.3, 600.0, 0.1, 0.2, 0.3, 0.1, 0.25, 0.08,
     0.3, 1.0, 2.0, 25.0, 0.01, 0.02, 0.02, 0.005, 0.02, 0.003,
     12.0, 25.0, 80.0, 400.0, 0.15, 0.2, 0.1, 0.05, 0.3, 0.09],

    [20.0, 30.0, 140.0, 1000.0, 0.2, 0.3, 0.4, 0.2, 0.3, 0.1,
     0.5, 2.0, 3.0, 40.0, 0.02, 0.03, 0.04, 0.01, 0.03, 0.01,
     25.0, 35.0, 150.0, 1100.0, 0.25, 0.3, 0.15, 0.08, 0.35, 0.1],

    [13.5, 19.8, 85.0, 550.0, 0.09, 0.18, 0.28, 0.09, 0.23, 0.07,
     0.28, 0.9, 1.9, 24.0, 0.009, 0.018, 0.019, 0.004, 0.018, 0.002,
     11.5, 24.5, 78.0, 390.0, 0.14, 0.19, 0.11, 0.045, 0.28, 0.085]
]

# for i, features in enumerate(examples, 1):
#     response = requests.post(url, json={"features": features})
#     print(f"Ejemplo {i}: {response.json()}")
for i, features in enumerate(examples, 1):
    response = requests.post(url, json={"features": features})
    pred = response.json().get("prediction")

    # Mapear a texto
    if pred == 0:
        class_name = "Maligno"
    elif pred == 1:
        class_name = "Benigno"
    else:
        class_name = "Desconocido"

    print(f"Ejemplo {i}: {{'prediction': {pred}}} - {class_name}")