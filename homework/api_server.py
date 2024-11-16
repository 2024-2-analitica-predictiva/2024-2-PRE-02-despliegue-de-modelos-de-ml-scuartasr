## api_server.py


# Estas líneas que están muestras dónde se está conectando el servidor y cuáles son
# los datos (indicados con -d) que se están enviando al servidor.

#
# Usage from command line:
# curl http://127.0.0.1:5000 -X POST -H "Content-Type: application/json" \
# -d '{"bathrooms": "2", "bedrooms": "3", "sqft_living": "1800", \
# "sqft_lot": "2200", "floors": "1", "waterfront": "1", "condition": "3"}'
#

# ============================================================================

#
# Importación de paquetes importantes
#
import pickle
import pandas as pd  # type: ignore
from flask import Flask, request  # type: ignore

# Con pickle se guarda el modelo, con pandas se trabajan con marcos de datos
# y flask sirve para la construcción de la aplicación.

# ============================================================================

#
# Inicialización de la aplicación
#

app = Flask(__name__)
app.config["SECRET_KEY"] = "you-will-never-guess"

#
# Lista con todas las características que se van a utilizar
#

FEATURES = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "condition",
]

#
# Se define la ruta de la aplicación. En este caso, se estará en la página
# principal de la aplicación, lo cual se define con "/"


@app.route("/", methods=["POST"])
def index():
    # Esta es la función que se ejecuta cuando se hace una petición POST	
    # a la página principal de la aplicación. En este caso, se está
    # enviando un JSON con los datos de la casa que se quiere predecir.
    # ¡Esta es la API!

    """API function"""

    args = request.json                                         # Se obtienen los datos del JSON que se envió
    filt_args = {key: [int(args[key])] for key in FEATURES}     # Se crea un diccionario con los datos
    df = pd.DataFrame.from_dict(filt_args)                      # Se transforma en marco de datos para dárselo al modelo

    with open("homework/house_predictor.pkl", "rb") as file:    # Se abre el modelo
        loaded_model = pickle.load(file)

    prediction = loaded_model.predict(df)                       # Se abre el modelo

    return str(prediction[0][0])                                # Se calcula la predicción y se entrega como texto


if __name__ == "__main__":                                      # Si se está ejecutando este archivo                                      
    app.run(debug=True)



