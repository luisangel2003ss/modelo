from flask import Flask, render_template, request, send_from_directory
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
from tensorflow.keras.losses import MeanSquaredError

app = Flask(__name__)  # ✅ Mover esto arriba

# Ruta para servir imágenes desde carpeta "img"
@app.route('/img/<path:filename>')
def custom_static(filename):
    return send_from_directory('img', filename)

# Cargar modelo y utilidades
model = load_model("modelo_trained.h5", custom_objects={"mse": MeanSquaredError})
preprocessor = joblib.load("preprocessor.pkl")
le_causa = joblib.load("labelencoder_causa.pkl")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predecir', methods=['POST'])
def predecir():
    year = int(request.form['year'])
    month = int(request.form['month'])
    operator = request.form['operator']
    county = request.form['county']
    operation = request.form['operation']
    source = request.form['source']

    df_input = pd.DataFrame([{
        'year': year,
        'month': month,
        'operator_edit': operator,
        'county_edit': county,
        'type_operation': operation,
        'source': source
    }])

    X_input = preprocessor.transform(df_input)
    y_reg_pred, y_clf_pred = model.predict(X_input)
    volumen_estimado = np.expm1(y_reg_pred[0][0])
    causa = le_causa.inverse_transform([np.argmax(y_clf_pred[0])])[0]

    return render_template('index.html',
                           prediction=True,
                           volumen=round(volumen_estimado, 2),
                           causa=causa)

if __name__ == '__main__':
    app.run(debug=True)
