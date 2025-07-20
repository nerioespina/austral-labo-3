# ML Forecast con categorías
## Importamos las librerías

import pandas as pd
from mlforecast import MLForecast
from lightgbm import LGBMRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
import numpy as np

from window_ops.rolling import rolling_mean
import optuna
from sklearn.model_selection import TimeSeriesSplit
from utilsforecast.feature_engineering import fourier

# Cargamos los datos
print("\n--- 0. Cargando datos ---")
df = pd.read_csv('./sell-in.txt', sep='\t', encoding='utf-8')
df_productos_predecir = pd.read_csv('./product_id_apredecir201912.txt', sep='\t', encoding='utf-8')
df = df[df['product_id'].isin(df_productos_predecir['product_id'])]

# Hacemos pivot para que se generen todos los valores por cada periodo, producto y cliente
print("\n--- 1. Transformando datos a formato ancho ---")
df_pivot = df.pivot_table(
    index=['product_id', 'customer_id'],
    columns='periodo',
    values='tn',
    aggfunc='sum',
    fill_value=None
)
df_pivot = df_pivot.reset_index()
df_pivot.columns.name = None


# Hacemos un melt para transformar el DataFrame a formato largo
print("\n--- 2. Transformando datos a formato largo ---")
df = df_pivot.melt(
    id_vars=['product_id', 'customer_id'],
    var_name='periodo',
    value_name='y' # MLForecast usa 'y' como nombre de la variable objetivo
)


# Estableciendo configuraciones y parámetros para train y test
print("\n--- 3. Configurando datos de entrenamiento y prueba ---")
np.random.seed(42)
df_real_201912 = df[df['periodo'] == 201912][['product_id', 'customer_id', 'y']]

FECHA_CORTE = '2019-10-01'
horizonte_prediccion = 2
product_ids = df['product_id'].unique()
df_pred_final = pd.DataFrame()
df_best_params = pd.DataFrame()

# iniciamos el train y test
print("\n--- 4. Entrenando modelos y optimizando hiperparámetros ---")
# Crear un DataFrame para guardar los mejores hiperparámetros de cada serie
contador = 0
cantidad_total = len(product_ids)
for pid in product_ids:
    contador += 1
    print(f"\nProcesando product_id: {pid} ({contador}/{cantidad_total})")
    df_prod = df[df['product_id'] == pid].copy()
    if df_prod.empty:
        continue

    df_prod['unique_id'] = df_prod['product_id'].astype(str) + "_" + df_prod['customer_id'].astype(str)
    df_prod['ds'] = pd.to_datetime(df_prod['periodo'], format='%Y%m')
    df_prod['y'] = df_prod['y'].fillna(0)
    
    df_final_prod = df_prod.groupby(['unique_id', 'ds'], as_index=False)['y'].sum()
    df_final_prod = df_final_prod.sort_values(by=['unique_id', 'ds']).reset_index(drop=True)
    df_entrenamiento_prod = df_final_prod[df_final_prod['ds'] <= FECHA_CORTE]
    duplicates = df_entrenamiento_prod.duplicated(subset=['unique_id', 'ds']).sum()
    if duplicates > 0:
        df_entrenamiento_prod = df_entrenamiento_prod.drop_duplicates(
            subset=['unique_id', 'ds'], keep='last'
        ).reset_index(drop=True)

    def objective(trial):
        params = {
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'random_state': 42
        }

        tscv = TimeSeriesSplit(n_splits=3)
        maes = []
        for train_idx, val_idx in tscv.split(df_entrenamiento_prod):
            train_data = df_entrenamiento_prod.iloc[train_idx].copy()
            val_data = df_entrenamiento_prod.iloc[val_idx].copy()
            val_data = val_data.drop_duplicates(subset=['unique_id', 'ds'], keep='last')
            try:
                fcst = MLForecast(
                    models=LGBMRegressor(**params, device='gpu'),
                    freq='MS',
                    lags=list(range(1, 25)),
                    date_features=['month', 'year'],
                )
                fcst.fit(train_data, static_features=[])
                h = val_data['ds'].nunique()
                preds = fcst.predict(h=h)
                preds = preds.drop_duplicates(subset=['unique_id', 'ds'], keep='last')
                comparison_df = pd.merge(
                    val_data, 
                    preds, 
                    on=['unique_id', 'ds'], 
                    how='inner'
                )
                if len(comparison_df) > 0:
                    maes.append(mean_absolute_error(comparison_df['y'], comparison_df['LGBMRegressor']))
                else:
                    maes.append(1000)
            except Exception as e:
                maes.append(1000)
        return np.mean(maes)

    try:
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=20, show_progress_bar=False)
        best_params = study.best_params
        best_params['random_state'] = 42

        # Guardar los mejores hiperparámetros en el DataFrame
        row = {'product_id': pid}
        row.update(best_params)
        df_best_params = pd.concat([df_best_params, pd.DataFrame([row])], ignore_index=True)

        fcst_prod = MLForecast(
            models=LGBMRegressor(**best_params),
            freq='MS',
            lags=list(range(1, 25)),
            date_features=['month', 'year'],
        )
        fcst_prod.fit(df_entrenamiento_prod, static_features=[])
        pred_prod = fcst_prod.predict(h=horizonte_prediccion)
        pred_prod['product_id'] = pid
        pred_prod_201912 = pred_prod[pred_prod['ds'] == '2019-12-01'].copy()
        if not pred_prod_201912.empty:
            pred_prod_201912['customer_id'] = pred_prod_201912['unique_id'].str.split('_').str[1].astype(int)
            pred_prod_201912.rename(columns={'LGBMRegressor': 'tn'}, inplace=True)
            df_pred_final = pd.concat([
                df_pred_final, 
                pred_prod_201912[['product_id', 'customer_id', 'tn']]
            ], ignore_index=True)
    except Exception as e:
        print(f"Error procesando product_id {pid}: {e}")
        continue

# Resumen final
if not df_pred_final.empty:
    df_pred_sum = df_pred_final.groupby('product_id', as_index=False)['tn'].sum()
    print(df_pred_sum)
else:
    print("No se generaron predicciones")

# Mostrar los mejores hiperparámetros por serie
print(df_best_params)


print(df_real_201912.shape, df_pred_final.shape)

# Calcula el error cuadrático medio (MSE) entre las predicciones y los valores reales
# Para esto, necesitamos los valores reales correspondientes a las predicciones

# Unimos las predicciones con los valores reales
df_eval = pd.merge(df_pred_final, df_real_201912, on=['product_id', 'customer_id'], how='inner')

# Calculamos el error cuadrático medio
mse = mean_squared_error(df_eval['y'].fillna(0), df_eval['tn'].fillna(0))
print(f'Error cuadrático medio (MSE): {mse}')

# Guardamos los mejores hiperparámetros en un archivo CSV
print("\n--- 5. Guardando los mejores hiperparámetros ---")

df_best_params.to_csv('df_best_params.csv', index=False)

# generamos las predicciones para febrero de 2020
print("\n--- 6. Generando predicciones para febrero de 2020 ---")
FECHA_CORTE = '2019-12-01'
horizonte_prediccion = 2  # febrero 2020

product_ids = df['product_id'].unique()
df_pred_final = pd.DataFrame()

contador = 0
cantidad_total = len(product_ids)
    
for pid in product_ids:

    contador += 1
    print(f"\nProcesando product_id: {pid} ({contador}/{cantidad_total})")

    # Obtener los mejores hiperparámetros para el producto actual
    params_row = df_best_params[df_best_params['product_id'] == pid]
    if params_row.empty:
        print(f"No se encontraron hiperparámetros para el product_id {pid}. Saltando.")
        continue
    
    best_params_prod = params_row.drop(columns=['product_id']).to_dict('records')[0]
    
    # Asegurar que los parámetros que deben ser enteros lo sean
    for p in ['num_leaves', 'max_depth', 'n_estimators']:
        if p in best_params_prod:
            best_params_prod[p] = int(best_params_prod[p])

    df_prod = df[df['product_id'] == pid].copy()
    if df_prod.empty:
        continue

    df_prod['unique_id'] = df_prod['product_id'].astype(str) + "_" + df_prod['customer_id'].astype(str)
    df_prod['ds'] = pd.to_datetime(df_prod['periodo'], format='%Y%m')
    df_prod['y'] = df_prod['y'].fillna(0)
    df_final_prod = df_prod[['unique_id', 'ds', 'y']].sort_values(by=['unique_id', 'ds']).reset_index(drop=True)
    df_final_prod = df_final_prod.loc[:, ~df_final_prod.columns.duplicated()]
    df_entrenamiento_prod = df_final_prod[df_final_prod['ds'] <= FECHA_CORTE]

    fcst_prod = MLForecast(
        models=LGBMRegressor(**best_params_prod, device='gpu'),
        freq='MS',
        lags=list(range(1, 25)),
        date_features=['month', 'year'],
    )
    fcst_prod.fit(df_entrenamiento_prod, static_features=[])

    pred_prod = fcst_prod.predict(h=horizonte_prediccion)
    pred_prod['product_id'] = pid

    pred_prod_202002 = pred_prod[pred_prod['ds'] == '2020-02-01'].copy()
    if not pred_prod_202002.empty:
        pred_prod_202002['customer_id'] = pred_prod_202002['unique_id'].str.split('_').str[1].astype(int)
        pred_prod_202002.rename(columns={'LGBMRegressor': 'tn'}, inplace=True)

        df_pred_final = pd.concat([df_pred_final, pred_prod_202002[['product_id', 'customer_id', 'tn']]], ignore_index=True)

if not df_pred_final.empty:
    df_pred_sum = df_pred_final.groupby('product_id', as_index=False)['tn'].sum()
    print(df_pred_sum)
else:
    print("No se generaron predicciones para febrero de 2020.")

# Guardamos las predicciones finales
print("\n--- 7. Guardando las predicciones finales ---")
df_pred_sum.to_csv('df_pred_sum_b.csv', index=False)


