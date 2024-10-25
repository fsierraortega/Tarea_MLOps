from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import pickle
import numpy as np
import pandas as pd

# Cargar el modelo y el scaler desde los archivos .pkl
with open('modelo_arbol_decision.pkl', 'rb') as archivo_modelo:
    modelo = pickle.load(archivo_modelo)

with open('scaler.pkl', 'rb') as archivo_scaler:
    scaler = pickle.load(archivo_scaler)

# Definir las características esperadas
columnas = ['Major_Axis_Length', 'Perimeter', 'Area', 'Convex_Area', 'Eccentricity']

# Crear la aplicación FastAPI
app = FastAPI(title="Clasificación de Granos de Arroz")

# Definir el endpoint para predicción
@app.get("/prediccion/")
async def predecir_fraude(Major_Axis_Length: float,
    Perimeter: float,
    Area: int,
    Convex_Area: int,
    Eccentricity: float):
    try:
        # Convertir la entrada en un DataFrame
        datos_entrada = pd.DataFrame([[Major_Axis_Length,Perimeter,Area,Convex_Area,Eccentricity]], columns=columnas)
        
        # Escalar las características
        datos_entrada_scaled = scaler.transform(datos_entrada)
        
        # Realizar la predicción
        prediccion = modelo.predict(datos_entrada_scaled)
        probabilidad = modelo.predict_proba(datos_entrada_scaled)[:, 1]
        
        # Construir la respuesta
        resultado = {
            "Tipo de Arroz": 'Osmancik' if bool(prediccion[0]) else 'Cammeo',
            "Probabilidad Osmancik": float(probabilidad[0])
        }
        
        return resultado
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, port=8080, host="0.0.0.0")