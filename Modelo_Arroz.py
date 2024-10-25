# # `Análisis de modelo de clasificación supervisado`

# Participantes:
# - Fernando Sierra
# - Diego Valdivia
# - Cristian Valenzuela

# ### `Importar las bibliotecas necesarias `
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.io.arff as arff
from scipy import stats
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score


# ### `Funciones`
def graficos_univariados(df, variable):
    # Configurar el tamaño de la figura
    plt.figure(figsize=(10, 4))

    # Crear subplots: 1 fila, 2 columnas
    plt.subplot(1, 2, 1)
    # Histograma usando Seaborn
    sns.histplot(df[variable], kde=True, color="lightblue", alpha=0.8)
    plt.title('Histograma de ' + variable)

    plt.subplot(1, 2, 2)
    # Boxplot usando Seaborn
    sns.boxplot(y=df[variable], color="lightblue")
    plt.title('Boxplot de ' + variable)

    # Mejorar layout y mostrar la figura
    plt.tight_layout()
    plt.show()


def elimina_outliers(df, columnas, umbral):
    """
    Elimina outliers del Dataset basado en el Z-Score.
    Parámetros:
    - df: Dataset a tratar.
    - columnas: Lista con los nombres de las columnas a analizar.
    - umbral: Umbral del Z-Score para identificar outliers.
    Retorna:
    - datos_filtrados: DataFrame sin outliers.
    """
    indices_a_eliminar = set()
    
    for columna in columnas:
        # Calcular el Z-Score para cada columna
        z_scores = stats.zscore(df[columna])
        
        # Identificar índices de outliers y agregarlos al conjunto
        indices_a_eliminar.update(df[abs(z_scores) > umbral].index)
        #print(f"Outliers eliminados de {columna}: {len(indices_a_eliminar)}")
    
    # Imprimir el número de outliers identificados
    print("Total de valores atípicos eliminados por Z-Score:", len(indices_a_eliminar))
    
    # Elimina outliers del DataFrame usando los índices identificados
    df_filtrados = df.drop(index=indices_a_eliminar)
    
    return df_filtrados


def graf_matriz_confusion(pred_default, pred_hp, name_model):
    # Calcula matrices de confusión
    conf_matrix_default = confusion_matrix(y_test, pred_default)
    conf_matrix_hp = confusion_matrix(y_test, pred_hp)
    # Configurar la visualización
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    # Títulos de los subplots
    ax[0].set_title(name_model + ' Sin Ajuste de Hiperparámetros')
    ax[1].set_title(name_model + ' Con Ajuste de Hiperparámetros')

    class_labels = ['Cammeo', 'Osmancik']
    # Visualización de la matriz de confusión para el modelo sin ajuste de hiperparámetros
    sns.heatmap(conf_matrix_default, annot=True, fmt='d', cmap='Blues', ax=ax[0], cbar=False, xticklabels=class_labels, yticklabels=class_labels)
    ax[0].set_xlabel('Predicciones')
    ax[0].set_ylabel('Verdaderos')

    # Visualización de la matriz de confusión para el modelo con ajuste de hiperparámetros
    sns.heatmap(conf_matrix_hp, annot=True, fmt='d', cmap='Greens', ax=ax[1], cbar=False, xticklabels=class_labels, yticklabels=class_labels)
    ax[1].set_xlabel('Predicciones')
    ax[1].set_ylabel('Verdaderos')

    plt.show()
    

def graf_feature_importances(df):
    # Orden por importancia
    features_df = df.sort_values(by='Importance', ascending=False)

    # Porcentaje
    features_df['Importance_Percentage'] = 100 * (features_df['Importance'] / features_df['Importance'].sum())

    # Suma acumulada
    features_df['Cumulative_Importance'] = features_df['Importance_Percentage'].cumsum()

    # Gráfico
    plt.figure(figsize=(12, 4))
    plt.plot(features_df['Feature'], features_df['Cumulative_Importance'], marker='o', drawstyle="steps-post")
    plt.xticks(rotation=90) 
    plt.xlabel('Características')
    plt.ylabel('Importancia Acumulada (%)')
    plt.title('Importancia Acumulada de las Características')
    plt.grid(True)

    # Linea roja al 80%
    plt.axhline(y=80, color='r', linestyle='--')
    plt.xlim([-1, len(features_df)])

    plt.show()


# ### `Lectura de base de datos y primeros valores del dataframe`
ruta = os.getcwd()
print(f"La ruta actual es: {ruta}")
archivo_arff = ruta + '/Rice_Cammeo_Osmancik.arff'
datos = pd.DataFrame(arff.loadarff(open(archivo_arff, 'rt'))[0])
datos.describe()


# Las 7 variables numéricas tienen valores en rangos razonables desde su media, indicando que todos los datos parecen tener datos bien distribuidos y razonable, esto es señal de que son útiles modelar. La variable `Class` (que indica si el grano es `Cammeo` u `Osmancik`) será nuestra `variable objetivo` para la clasificación.

# ### `Tratamiento de duplicados y valores faltantes`: 
# Revisión de datos faltantes y tipo de variables
datos.info()
#Revisamos si hay duplicados
print()
print("Número de datos duplicados:", len(datos[datos.duplicated()]), ", de un total de", len(datos))

# #### Dado que no hay `datos faltantes` y tampoco `datos duplicados` procedemos al siguiente paso.
# 

# Se corrige los valores de la columna 'Class' eliminando los caracteres extraños
datos['Class'] = datos['Class'].str.decode("utf-8").str.replace("b'", "").str.replace("'", "")

#Estadisticas de variable respuesta.
print(datos.Class.value_counts(), datos.Class.value_counts('%'))

# plt.figure(figsize=(6, 6))
# datos['Class'].value_counts().plot(kind='bar', color="lightblue")
# plt.xlabel('Tipo de arroz')
# plt.ylabel('Frecuencia')
# plt.title('Distribución de arroces')
# plt.xticks(rotation=0)
# plt.show()

# En los gráficos se puede observar que existe más data de Osmancik que de Cammeo, pero la diferencia no es tan extrema (57% y 43% respectivamente). Por ende, por ahora no se requerirá técnicas de balanceo de clases, pero esto se puede retomar al evaluar el rendimiento del modelo.  

# ### `Tratamiento de outliers`

# Vamos a realizar un tratamiento de outliers para las variables `Eccentricity` y `Minor_Axis_Length`. Se ejecuto el `Z-score` con un umbral de `2.5`, con el cual se eliminaron `111 outliers` (2.9% del total del dataset).

# Z score para Eccentricity y Minor_Axis_Length
datos_filtrados = elimina_outliers(datos, ['Eccentricity', 'Minor_Axis_Length'], 2.5)


# ### `Ingeniera de características`
# Crearemos algunas variables que quiza nos puedan aportar al modelo:
# - La `relación de aspecto` puede indicar la elongación de los granos de arroz. Podría calcularse dividiendo el Major_Axis_Length y Minor_Axis_Length.
# - La `compacidad` podría indicar cuán compacto es un grano de arroz, y se puede calcular como el cuadrado del perímetro dividido por el área.
# - La `circularidad` es una medida de cuán cercano es el grano de arroz a una forma circular, y se podría calcular usando el área y el perímetro.

# Agregando nuevas características
datos_filtrados['Aspect_Ratio'] = datos_filtrados['Major_Axis_Length'] / datos_filtrados['Minor_Axis_Length']
datos_filtrados['Compactness']  = (datos_filtrados['Perimeter']**2) / datos_filtrados['Area']
datos_filtrados['Circularity']  = (4 * np.pi * datos_filtrados['Area']) / (datos_filtrados['Perimeter']**2)


# Preparar los datos para el modelo
X = datos_filtrados.drop(['Class'], axis=1)  # Excluyendo la variable objetivo
y = datos_filtrados['Class']  # Variable objetivo

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Entrenar un modelo Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Obtener la importancia de las características
feature_importances = rf.feature_importances_

# Visualizar la importancia de las características
importances_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances})
importances_df = importances_df.sort_values(by='Importance', ascending=False)

# Graficamos
# graf_feature_importances(importances_df)


# Uno de los análisis que se pueden realizar, es la ingeniería de características, de donde realizamos la creación de algunas variables y un gráfico de `features importances`, donde se puede concluir que con 5 variables podemos explicar un poco más del 80 % del modelo.
# 
# Este criterio será impuesto a los de correlación, ya que el aporte de la variable `Extent` se considera mínimo. Por otro lado, lo comentado de la multicolinealidad lo analizaremos con el performance del modelo.
# 

# ### `Selección de las variables, división de conjunto de datos y Estandarización `
# La pregunta importante que hay que responder es: "¿Qué tipo de modelo debemos utilizar?". Esto nos ayudará a manejar la complejidad de las variables. Puede que tengamos que buscar modelos como `SVM` o `árboles de decisión`, que pueden manejar bien estos casos.  
# 
# - `Selección de Variables`: Nos centraremos en `Major_Axis_Length`, `Perimeter`, `Area`, `Convex_Area` y `Eccentricity` como variables independientes.
# - `Codificación de la Variable Objetivo`: Convertiremos `Class` a una variable a 0 y 1 (0 = “Cammeo” y 1= “Osmancik”).
# - `División de Datos`: Separaremos los datos en conjuntos de entrenamiento y prueba (`80%` y `20%` respectivamente).
# - `Estandarizaremos`: Las variables nombradas en los puntos anteriores, esto es con el fin de buscar que los modelos se benefician de este proceso, como SVM.


# Selección de variables
X = datos_filtrados[['Major_Axis_Length', 'Perimeter', 'Area', 'Convex_Area', 'Eccentricity']]
y = datos_filtrados['Class']

# 0 = Cammeo y 1 =  de la variable objetivo
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Set de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=2024)

# Estadarización
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)


# Dimensiones
X_train_scaled.shape, X_test_scaled.shape, y_train.shape, y_test.shape
X_train_scaled.describe()
X_test_scaled.describe()

# Las proporciones de los conjuntos de datos de entrenamiento y testeo parecen estar semejantes estadísticamente.


# ### `Modelo Árbol de decisión`
tree_model = DecisionTreeClassifier(random_state=2023)
tree_model.fit(X_train_scaled, y_train)
pred_values_tree = tree_model.predict(X_test_scaled)

print(classification_report(y_test,pred_values_tree, target_names=label_encoder.classes_))

print(f"Tree accuracy: {round(accuracy_score(y_test, pred_values_tree), 3)}")

# La comparación preliminar de modelos usando `accuracy` como métrica de desempeño arrojó los siguientes resultados:
# - `SVM`: 92.8%
# - `Decision Tree`: 87.6%
# 
# 
# El modelo de `SVM` mostró el mejor rendimiento inicial (`92.8%`). Dado este resultado, parece ser el candidato más prometedor para proceder a la optimización de hiper-parámetros y preliminarmente mejorar aún más su desempeño. Por otro lado, tenemos el `árbol de decisión` que si bien, fue el que tuvo el peor resultado (`87.6%`), pero dadas sus capacidades, puede ser muy bueno si se le asignan los hiper-parámetros adecuados y computacionalmente no es tan costoso como SVM.
# 

# ### `Hiper-parámetros modelo Árbol de decisión` 

# Espacio de hiperparámetros con distribuciones para muestreo
parametros_tree = {
    'max_depth': randint(1, 10), 
    'min_samples_split': randint(2, 50), 
    'min_samples_leaf': randint(1, 50),
    'ccp_alpha': uniform(0.0, 0.1),  # Valores entre 0.0 y 0.1 para la poda de complejidad de costo
    'criterion': ['gini', 'entropy']  # Criterios para medir la calidad de una división
}

# Inicializar RandomizedSearchCV
random_search = RandomizedSearchCV(DecisionTreeClassifier(random_state=42), parametros_tree, n_iter=500, scoring='accuracy', cv=5, random_state=42, verbose=0)

# Entrenar RandomizedSearchCV
random_search.fit(X_train_scaled, y_train)

# Mejores hiperparámetros y su puntaje
print("Mejores Hiperparámetros:", random_search.best_params_)
print("Mejor Puntaje de Validación Cruzada:", random_search.best_score_)

# La optimización de hiper-parámetros mediante `RandomizedSearchCV` para el modelo árbol de decisión ha encontrado la siguiente mejor configuración:
# - `max_depth`: 5
# - `min_samples_leaf`: 30
# - `min_samples_split`: 40
# - `criterion`: 'entropy'
# - `ccp_alpha`: 0.0005145170754714768
# 
# Con un Accuracy promedio de 92% en la validación cruzada. Esta configuración será entrenada y la evaluaremos en el conjunto de prueba para obtener las métricas finales de desempeño y la matriz de confusión.
# 

# Entrenar el modelo Arbol de decisión con los mejores hiper-parámetros encontrados
best_params = random_search.best_params_
best_model_tree = DecisionTreeClassifier(**best_params)

best_model_tree.fit(X_train_scaled, y_train)

# Predecir en el conjunto de prueba
pred_values_tree_hp = best_model_tree.predict(X_test_scaled)

# Calcular métricas de desempeño y matriz de confusión
print(classification_report(y_test, pred_values_tree_hp, target_names=label_encoder.classes_))
#print(confusion_matrix(y_test, pred_values_tree_hp))


# Grafico de comparación matriz de confusión
# graf_matriz_confusion(pred_values_tree, pred_values_tree_hp, "Árbol de decisión")


print("Arbol de decisión sin hiper-parámetros: ")
print(f"Tree accuracy: {round(accuracy_score(y_test, pred_values_tree), 4)}")
print(f"Tree f1 score: {round(f1_score(y_test, pred_values_tree), 4)}")
print(f"Tree precision: {round(precision_score(y_test, pred_values_tree), 4)}")
print(f"Tree recall: {round(recall_score(y_test, pred_values_tree), 4)}")
print(" ")
print("Arbol de decisión con hiper-parámetros: ")
print(f"Tree HP accuracy: {round(accuracy_score(y_test, pred_values_tree_hp), 4)}")
print(f"Tree HP f1 score: {round(f1_score(y_test, pred_values_tree_hp), 4)}")
print(f"Tree HP precision: {round(precision_score(y_test, pred_values_tree_hp), 4)}")
print(f"Tree HP recall: {round(recall_score(y_test, pred_values_tree_hp), 4)}")


# Guardamos el modelo entrenado.
import pickle

with open(ruta + 'modelo_arbol_decision.pkl', 'wb') as archivo_salida:
    pickle.dump(best_model_tree, archivo_salida)

with open('scaler.pkl', 'wb') as archivo_salida:
    pickle.dump(scaler, archivo_salida)

print("Modelo_arbol_decision.pkl generado exitosamente!")