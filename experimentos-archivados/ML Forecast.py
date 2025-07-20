# %% [markdown]
# # Construcción del dataset

# %%
import pandas as pd
from mlforecast import MLForecast
from lightgbm import LGBMRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
import numpy as np

from window_ops.rolling import rolling_mean
import optuna
from sklearn.model_selection import TimeSeriesSplit
from utilsforecast.feature_engineering import fourier

# %%

df = pd.read_csv('../data/sell-in.txt', sep='\t', encoding='utf-8')
df.head()

# %%
df_productos_predecir = pd.read_csv('../data/product_id_apredecir201912.txt', sep='\t', encoding='utf-8')
df_productos_predecir.head()

# %%
df['periodo'].sort_values().unique()

# %%
df_pivot = df.pivot_table(
    index=['product_id', 'customer_id'],
    columns='periodo',
    values='tn',
    aggfunc='sum',
    fill_value=None
)
df_pivot = df_pivot.reset_index()
df_pivot.columns.name = None
df_pivot.head()

# %%
# Remove from df_pivot the products that are not in df_productos_predecir
df_pivot = df_pivot[df_pivot['product_id'].isin(df_productos_predecir['product_id'])]

# %%
# df_mlforecast = df_pivot[df_pivot['customer_id'] == 10001].copy()
df_mlforecast = df_pivot.copy()

# %%
df_mlforecast.head()

# %%
df_mlforecast.shape

# %%
# --- PASO 1: TRANSFORMACIÓN DE DATOS A FORMATO LARGO ---
# Este es el formato conveniente que usaremos en ambos casos.
print("\n--- 1. Transformando datos a formato largo ---")
df_long = df_mlforecast.melt(
    id_vars=['product_id', 'customer_id'],
    var_name='periodo',
    value_name='y' # MLForecast usa 'y' como nombre de la variable objetivo
)

df_long.head()

# %%
FECHA_CORTE = '2019-10-01'
horizonte_prediccion = 2
# Lista de product_id únicos a procesar
product_ids = df_mlforecast['product_id'].unique()

# DataFrame para acumular resultados finales
df_pred_final = pd.DataFrame()

for pid in product_ids:
    # 1. Filtrar datos para el product_id actual
    df_pivot_prod = df_pivot[df_pivot['product_id'] == pid]
    if df_pivot_prod.empty:
        continue

    # 2. Transformar a formato largo
    df_long_prod = df_pivot_prod.melt(
        id_vars=['product_id', 'customer_id'],
        var_name='periodo',
        value_name='y'
    )
    df_long_prod = df_long_prod.fillna(0)
    df_long_prod['unique_id'] = df_long_prod['product_id'].astype(str) + "_" + df_long_prod['customer_id'].astype(str)
    df_long_prod['ds'] = pd.to_datetime(df_long_prod['periodo'], format='%Y%m')
    df_final_prod = df_long_prod[['unique_id', 'ds', 'y']].sort_values(by=['unique_id', 'ds']).reset_index(drop=True)

    # 3. Dividir en entrenamiento y validación
    df_final_prod = df_final_prod.loc[:, ~df_final_prod.columns.duplicated()]
    df_entrenamiento_prod = df_final_prod[df_final_prod['ds'] <= FECHA_CORTE]
    # Entrenar modelo
    fcst_prod = MLForecast(
        models=LGBMRegressor(random_state=42, n_estimators=100),
        freq='MS',
        lags=[1, 2, 3, 6, 12],
        date_features=['month', 'year'],
    )
    fcst_prod.fit(df_entrenamiento_prod, static_features=[])

    # 4. Predecir 2 pasos adelante (201911 y 201912)
    pred_prod = fcst_prod.predict(h=horizonte_prediccion)
    pred_prod['product_id'] = pid

    # 5. Filtrar solo 201912 y extraer customer_id
    pred_prod_201912 = pred_prod[pred_prod['ds'] == '2019-12-01'].copy()
    pred_prod_201912['customer_id'] = pred_prod_201912['unique_id'].str.split('_').str[1].astype(int)
    pred_prod_201912.rename(columns={'LGBMRegressor': 'tn'}, inplace=True)

    # 6. Acumular resultados
    df_pred_final = pd.concat([df_pred_final, pred_prod_201912[['product_id', 'customer_id', 'tn']]], ignore_index=True)

# Sumar tn por producto
df_pred_sum = df_pred_final.groupby('product_id', as_index=False)['tn'].sum()
print(df_pred_sum)


# %%
df_pred_sum.head()

# %%
# 1. Extraer los valores reales de df_long para 201912 y sumarlos por producto
df_validacion = (
    df_long[df_long['periodo'] == 201912]  # sin comillas, como int
    .groupby('product_id', as_index=False)['y']
    .sum()
    .rename(columns={'y': 'tn_real'})
)

# 2. Unir con las predicciones
df_eval = df_validacion.merge(df_pred_sum, on='product_id', how='inner')

y_real = df_eval['tn_real']
y_pred = df_eval['tn']

# 3. Calcular las métricas
mae = mean_absolute_error(y_real, y_pred)
mse = mean_squared_error(y_real, y_pred)
rmse = np.sqrt(mse)
mape = mean_absolute_percentage_error(y_real, y_pred)
r2 = r2_score(y_real, y_pred)

# 4. Imprimir resultados
print("\n" + "="*40)
print(" MÉTRICAS DE RENDIMIENTO DEL MODELO")
print("="*40)
print(f"Error Absoluto Medio (MAE):       {mae:.2f} unidades")
print(f"Raíz del Error Cuadrático (RMSE): {rmse:.2f} unidades")
print(f"Error Porcentual Absoluto (MAPE): {mape:.2%}")
print(f"Coeficiente de Determinación (R²): {r2:.2f}")
print("="*40)

print("\nInterpretación:")
print(f"- En promedio, el modelo se equivoca en {mae:.2f} toneladas (o la unidad que estés usando).")
print(f"- El error porcentual promedio es de {mape:.2%}.")
print(f"- Un R² de {r2:.2f} indica qué proporción de la varianza de los datos es explicada por el modelo (más cercano a 1 es mejor).")


# %%
FECHA_CORTE_FINAL = '2019-12-01'
horizonte_prediccion_final = 2
df_pred_final_202002 = pd.DataFrame()

product_ids = df_mlforecast['product_id'].unique()

for pid in product_ids:
    df_pivot_prod = df_pivot[df_pivot['product_id'] == pid]
    if df_pivot_prod.empty:
        continue

    df_long_prod = df_pivot_prod.melt(
        id_vars=['product_id', 'customer_id'],
        var_name='periodo',
        value_name='y'
    )
    df_long_prod = df_long_prod.fillna(0)
    df_long_prod['unique_id'] = df_long_prod['product_id'].astype(str) + "_" + df_long_prod['customer_id'].astype(str)
    df_long_prod['ds'] = pd.to_datetime(df_long_prod['periodo'], format='%Y%m')
    df_final_prod = df_long_prod[['unique_id', 'ds', 'y']].sort_values(by=['unique_id', 'ds']).reset_index(drop=True)

    df_final_prod = df_final_prod.loc[:, ~df_final_prod.columns.duplicated()]
    df_entrenamiento_prod = df_final_prod[df_final_prod['ds'] <= FECHA_CORTE_FINAL]

    fcst_prod = MLForecast(
        models=LGBMRegressor(random_state=42, n_estimators=100),
        freq='MS',
        lags=[1, 2, 3, 6, 12],
        date_features=['month', 'year'],
    )
    fcst_prod.fit(df_entrenamiento_prod, static_features=[])

    pred_prod = fcst_prod.predict(h=horizonte_prediccion_final)
    pred_prod['product_id'] = pid

    # Filtrar solo 2020-02-01 y extraer customer_id
    pred_prod_202002 = pred_prod[pred_prod['ds'] == '2020-02-01'].copy()
    pred_prod_202002['customer_id'] = pred_prod_202002['unique_id'].str.split('_').str[1].astype(int)
    pred_prod_202002.rename(columns={'LGBMRegressor': 'tn'}, inplace=True)

    df_pred_final_202002 = pd.concat([df_pred_final_202002, pred_prod_202002[['product_id', 'customer_id', 'tn']]], ignore_index=True)

# Sumar tn por producto para 202002
df_pred_sum_202002 = df_pred_final_202002.groupby('product_id', as_index=False)['tn'].sum()
print(df_pred_sum_202002)


# %%
df_pred_sum_202002.head()

# %%
df_pred_sum_202002.shape

# %%
df_pred_sum_202002.to_csv('prediccion_tn_por_producto_3.csv', index=False)


