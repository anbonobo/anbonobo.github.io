import dill
from sklearn.pipeline import Pipeline
import os
import json
import pandas as pd
from datetime import datetime

path = os.environ.get('PROJECT_PATH', '.')
def load_pipeline(file_path):
    try:
        with open(file_path, 'rb') as file:
            return dill.load(file)
    except Exception as e:
        print(f"Failed to load with dill: {e}")
        return None
def use_pipeline_for_prediction(pipeline, data):
    try:
        predictions = pipeline.predict(data)
        return predictions
    except Exception as e:
        print(f"Failed to predict: {e}")
        return None

def load_json_files_to_dataframe(folder_path):
    all_data = []
    for json_file in os.listdir(folder_path):
        if json_file.endswith('.json'):
            file_path = os.path.join(folder_path, json_file)
            with open(file_path, 'r') as file:
                data = json.load(file)
                df = pd.json_normalize(data)
                all_data.append(df)
    return pd.concat(all_data, ignore_index=True)

# Функция для сохранения предсказаний в CSV
def save_predictions_to_csv(predictions, file_path):
    try:
        df = pd.DataFrame(predictions)
        df.to_csv(file_path, index=False)
        print(f"Predictions saved to {file_path}")
    except Exception as e:
        print(f"Failed to save predictions: {e}")

def predict() -> None:
    # Путь к модели и JSON-файлам
    models_dir = os.path.join(path, 'data', 'models')
    latest_model = None
    latest_time = None

    for filename in os.listdir(models_dir):
        if filename.startswith('cars_pipe') and filename.endswith('.pkl'):
            model_path = os.path.join(models_dir, filename)

            # Извлекаем временную метку из имени файла
            try:
                timestamp_str = filename[len('cars_pipe_'):-len('.pkl')]
                timestamp = datetime.strptime(timestamp_str, "%Y%m%d%H%M")

                # Проверяем, является ли текущий файл самым последним
                if latest_time is None or timestamp > latest_time:
                    latest_time = timestamp
                    latest_model = model_path
            except ValueError:
                # Если формат имени файла не соответствует ожиданиям, пропускаем его
                continue

    if latest_model:
        print(f"The latest model is: {latest_model}")
    else:
        print("No models found.")


    json_folder_path = os.path.join(path, 'data', 'test')

    # Загрузка модели
    pipeline = load_pipeline(latest_model)
    if pipeline:
        # Загрузка данных из JSON-файлов
        data_df = load_json_files_to_dataframe(json_folder_path)
        # Предполагаем, что данные не содержат целевых переменных и являются входными данными
        if not data_df.empty:
            predictions = use_pipeline_for_prediction(pipeline, data_df)
            if predictions is not None:
                # Сохранение предсказаний в CSV
                save_predictions_to_csv(predictions,
                                        f'{path}/data/predictions/prediction{datetime.now().strftime("%Y%m%d%H%M")}.csv')
    else:
        print("Pipeline not loaded.")


# Основной блок кода
if __name__ == "__main__":
    predict()