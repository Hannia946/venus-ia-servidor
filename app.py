import os
import numpy as np
import joblib
from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow import keras

app = Flask(__name__)

# ── Cargar modelo y scaler al iniciar el servidor ─────────────────
# Render busca los archivos en la raíz del proyecto
MODEL_PATH  = os.path.join(os.path.dirname(__file__), 'modelo_venus_V2.keras')
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'scaler_venus.pkl')

try:
    model  = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Modelo y scaler cargados correctamente")
except Exception as e:
    print(f"Error al cargar el modelo: {e}")
    model  = None
    scaler = None

ESTADOS = ['Óptimo', 'Estrés Ambiental', 'Estrés Hídrico', 'Crítico']

# ── Ruta de verificación ──────────────────────────────────────────
# Útil para saber si el servidor está vivo sin mandar datos reales
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'servidor': 'Venus Atrapamoscas - IA',
        'estado':   'activo',
        'modelo':   'cargado' if model else 'error al cargar'
    })

# ── Ruta principal de predicción ──────────────────────────────────
@app.route('/predecir', methods=['POST'])
def predecir():
    if model is None or scaler is None:
        return jsonify({'error': 'Modelo no disponible'}), 503

    datos = request.get_json()
    if not datos:
        return jsonify({'error': 'No se recibieron datos'}), 400

    # Validamos que vengan todos los campos necesarios
    campos = ['temperatura', 'humedad', 'luz', 'tds', 'nivel_agua', 'estacion']
    faltantes = [c for c in campos if c not in datos]
    if faltantes:
        return jsonify({
            'error':    'Campos faltantes',
            'faltantes': faltantes
        }), 400

    try:
        X = np.array([[
            datos['temperatura'],
            datos['humedad'],
            datos['luz'],
            datos['tds'],
            datos['nivel_agua'],
            datos['estacion']
        ]])

        X_norm = scaler.transform(X)
        probs  = model.predict(X_norm, verbose=0)[0]
        clase  = int(np.argmax(probs))

        return jsonify({
            'estado':          ESTADOS[clase],
            'clase':           clase,
            'probabilidades': {
                ESTADOS[i]: round(float(probs[i]) * 100, 1)
                for i in range(4)
            },
            'datos_recibidos': datos
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ── Ruta para verificar el modelo con un dato de prueba ──────────
@app.route('/prueba', methods=['GET'])
def prueba():
    if model is None:
        return jsonify({'error': 'Modelo no disponible'}), 503

    lectura_prueba = np.array([[28.5, 65.0, 25000.0, 30.0, 2.0, 1]])
    X_norm = scaler.transform(lectura_prueba)
    probs  = model.predict(X_norm, verbose=0)[0]
    clase  = int(np.argmax(probs))

    return jsonify({
        'mensaje':  'Prueba con lectura óptima de verano',
        'resultado': ESTADOS[clase],
        'clase':     clase,
        'probabilidades': {
            ESTADOS[i]: round(float(probs[i]) * 100, 1)
            for i in range(4)
        }
    })

if __name__ == '__main__':
    # El puerto lo asigna Render automáticamente via variable de entorno
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)