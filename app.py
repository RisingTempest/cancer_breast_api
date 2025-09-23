# app.py
from flask import Flask, request, jsonify
import joblib
import numpy as np
import logging

# ----------------------
# Configuración de logging
# ----------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ----------------------
# Crear la aplicación Flask
# ----------------------
app = Flask(__name__)

# ----------------------
# Cargar modelo
# ----------------------
model = joblib.load("modelo.pkl")
logging.info("Modelo cargado correctamente.")

# ----------------------
# Ruta principal (GET)
# ----------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "API lista. Endpoint POST /predict"}), 200

# ----------------------
# Endpoint de predicción
# ----------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        if data is None:
            return jsonify({"error": "JSON inválido o body vacío"}), 400
        if "features" not in data:
            return jsonify({"error": "Falta la clave 'features' en el JSON"}), 400

        features = data["features"]
        if not isinstance(features, list):
            return jsonify({"error": "'features' debe ser una lista de números"}), 400

        features = np.array(features).reshape(1, -1)
        expected = getattr(model, "n_features_in_", None)
        if expected is not None and features.shape[1] != expected:
            return jsonify({
                "error": f"Se esperaban {expected} características, pero se recibieron {features.shape[1]}"
            }), 400

        # Logging de la petición
        logging.info(f"Recibida petición con features: {features.tolist()}")

        # Predicción y probabilidades
        prediction = model.predict(features)
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)[0].tolist()
        else:
            probabilities = None

        response = {"prediction": int(prediction[0])}
        if probabilities:
            response["probabilities"] = probabilities

        logging.info(f"Predicción: {response}")
        return jsonify(response), 200

    except ValueError as ve:
        logging.error(f"ValueError: {ve}")
        return jsonify({"error": f"ValueError: {str(ve)}"}), 400
    except Exception as e:
        logging.error(f"Error inesperado: {e}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

# ----------------------
# Ejecutar app localmente
# ----------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
