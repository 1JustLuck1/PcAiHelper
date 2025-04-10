import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from flask import Flask, render_template, jsonify, request
from database.db_connector import get_db_connection
import joblib
from tensorflow import keras
from tensorflow.keras.models import load_model
import pandas as pd
from service import get_components_data
import json


app = Flask(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'static', 'models')

def load_model_app(model_name):
    """Безопасная загрузка модели разных форматов (.pkl, .keras, .joblib)"""
    # Проверяем все возможные расширения
    extensions = ['.pkl', '.keras', '.joblib']
    model_path = None
    
    # Ищем файл с подходящим расширением
    for ext in extensions:
        possible_path = os.path.join(MODELS_DIR, f"{model_name}{ext}")
        if os.path.exists(possible_path):
            model_path = possible_path
            break
    
    if not model_path:
        raise FileNotFoundError(
            f"Model file {model_name} with extensions {extensions} not found in {MODELS_DIR}"
        )
    
    # Загружаем модель в зависимости от расширения
    if model_path.endswith('.pkl') or model_path.endswith('.joblib'):
        return joblib.load(model_path)
    elif model_path.endswith('.keras'):
        return keras.models.load_model(model_path)
    else:
        raise ValueError(f"Unsupported model format: {os.path.splitext(model_path)[1]}")

# Загрузка моделей
try:
    cpu_preload_model = load_model_app('cpu_preload_predictor') # Обогащение опросных данных дополнительными
    gpu_preload_model = load_model_app('gpu_preload_predictor')

    cpu_main_preprocessor =  load_model_app('cpu_main_preprocessor') #препроцессинг (энкодинг)
    gpu_main_preprocessor =  load_model_app('gpu_main_preprocessor')

    cpu_main_model = load_model_app('cpu_main_model') #предсказание mark показателя
    gpu_main_model = load_model_app('gpu_main_model')

    mark_to_cpu_model = load_model_app('cpu_mark_to_cpu_model') # перевод mark показателя в модель
    mark_to_gpu_model = load_model_app('gpu_mark_to_gpu_model')

    print(f"INFO ---> Модели успешно загружены!")
except Exception as e:
    print(f"Ошибка загрузки модели: {str(e)}")
    cpu_preload_model = None
    gpu_preload_model = None
    cpu_main_preprocessor = None
    gpu_main_preprocessor = None
    cpu_main_model = None
    gpu_main_model = None
    mark_to_cpu_model = None
    mark_to_gpu_model = None

def get_links_for_components(cpu, gpu):
    cpu_links = {
        'DNS': f'https://www.dns-shop.ru/search/?q={cpu}',
        'OZON': f'https://www.ozon.ru/search/?text={cpu}',
        'Yandex market': f'https://market.yandex.ru/search?text={cpu}',
        'Citilink': f'https://www.citilink.ru/search/?text={cpu}',
        'Regard': f'https://www.regard.ru/catalog?search={cpu}'
    }

    gpu_links = {
        'DNS': f'https://www.dns-shop.ru/search/?q={gpu}',
        'OZON': f'https://www.ozon.ru/search/?text={gpu}',
        'Yandex market': f'https://market.yandex.ru/search?text={gpu}',
        'Citilink': f'https://www.citilink.ru/search/?text={gpu}',
        'Regard': f'https://www.regard.ru/catalog?search={gpu}'
    }

    return cpu_links, gpu_links

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/configure', methods=["GET","POST"])
def configure():
    return render_template('configure.html')

@app.route('/evaluate')
def evaluate():
    return render_template('evaluate.html')

@app.route('/info')
def info():
    return render_template('info.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/api/configure', methods=["POST"])
def api_configure():
    if request.method == 'POST':
        if gpu_preload_model is None:
            return jsonify({"error": "Model not loaded"}), 500
        try:
            data = request.get_json()
            # print("Получены данные:", data)  # Логируем ответы
            fields = [answer['answerId'] for answer in data['answers']]

            data_cpu_preload = pd.DataFrame([[
                fields[0], #sphere
                fields[1], #budget
                fields[2], #resolution
                fields[3]  #vendor(cpu)
            ]], columns=["sphere", "budget", "resolution", "vendor"])

            data_gpu_preload = pd.DataFrame([[
                fields[0], #sphere
                fields[1], #budget
                fields[2], #resolution
                fields[4], #vendor(gpu)
                fields[5]  #perfMargin
            ]], columns=["sphere", "budget", "resolution", "vendor","perfMargin"])
            #Модель предобработки входных данных
            cpu_input_preprocessing = cpu_preload_model.predict(data_cpu_preload)[0].tolist()
            #0 - cpuPrior
            #1 - cpuTDP
            #2 - cpuCores
            gpu_input_preprocessing = gpu_preload_model.predict(data_gpu_preload)[0].tolist()
            #0 - gpuPrior
            #1 - gpuTDP
            #2 - gpuMemory
            data_cpu_main = pd.DataFrame([[
                fields[0], #sphere
                cpu_input_preprocessing[0], #cpuPrior
                fields[1], #budget
                fields[2], #resolution
                fields[3], #vendor(cpu)
                cpu_input_preprocessing[1], #cpuTDP
                cpu_input_preprocessing[2], #cpuCores
            ]], columns=["sphere","cpuPrior", "budget", "resolution", "vendor", "tdp", "cores"])

            data_gpu_main = pd.DataFrame([[
                fields[0], #sphere
                gpu_input_preprocessing[0], #gpuPrior
                fields[1], #budget
                fields[2], #resolution
                fields[4], #vendor(gpu)
                int(fields[5]), #perfMargin
                gpu_input_preprocessing[2], #gpuMemory
                gpu_input_preprocessing[1] #gpuTDP
            ]], columns=["sphere","gpuPrior", "budget", "resolution", "vendor", "perfMargin", "memSize", "tdp"])
            
            try:
                cpu_main_processed_data = cpu_main_preprocessor.transform(data_cpu_main)

                cpu_main_data = cpu_main_model.predict(cpu_main_processed_data)[0][0].tolist()

                if fields[5] == '1':
                    cpu_main_data = cpu_main_data * 1.25

                cpu_vendor_encoder = mark_to_cpu_model['vendor_encoder'].transform([fields[3]])

                df_cpu_mark_value = pd.DataFrame({
                    'cpuMark': [cpu_main_data],
                    'vendor_encoded': cpu_vendor_encoder
                    })

                final_cpu_model = mark_to_cpu_model['model'].predict(df_cpu_mark_value)
                cpu = mark_to_cpu_model['cpuName_encoder'].inverse_transform(final_cpu_model)[0]

                # print(cpu)
            except Exception as e:
                print("CPU MODEL ERROR ---> ", str(e))

            try:
                gpu_main_processed_data = gpu_main_preprocessor.transform(data_gpu_main)

                gpu_main_data = gpu_main_model.predict(gpu_main_processed_data)[0][0].tolist()
                if fields[5] == '1':
                    gpu_main_data = gpu_main_data * 1.25

                gpu_vendor_encoder = mark_to_gpu_model['vendor_encoder'].transform([fields[4]])

                df_gpu_mark_value = pd.DataFrame({
                    'gpuMark': [gpu_main_data],
                    'vendor_encoded': gpu_vendor_encoder
                    })

                final_gpu_model = mark_to_gpu_model['model'].predict(df_gpu_mark_value)
                gpu = mark_to_gpu_model['gpuName_encoder'].inverse_transform(final_gpu_model)[0]

                # print(gpu)
            except Exception as e:
                print("GPU MODEL ERROR ---> ", str(e))

            cpudata, gpudata, total_tdp = get_components_data(cpu, gpu)
            
            cpudata = json.dumps(cpudata)
            gpudata = json.dumps(gpudata)
            cpu_json = json.loads(cpudata)
            gpu_json = json.loads(gpudata)

            cpu_links, gpu_links = get_links_for_components(cpu, gpu)

            cpu_links = json.loads(json.dumps(cpu_links))
            gpu_links = json.loads(json.dumps(gpu_links))

            return jsonify({
                "cpu": cpu,
                "gpu": gpu,
                "total_tdp": total_tdp,
                "cpud": cpu_json,
                "gpud": gpu_json,
                "cpu_links": cpu_links,
                "gpu_links": gpu_links
            })
        except Exception as e:
            # print(str(e))
            return jsonify({"error": str(e)}), 500

@app.route('/api/cpus')
def get_cpus():
    conn = get_db_connection()
    with conn.cursor() as cursor:
        cursor.execute('SELECT * FROM cpus')
        cpus = cursor.fetchall()
    cursor.close()
    conn.close()
    return jsonify(cpus)

@app.route('/api/gpus')
def get_gpus():
    conn = get_db_connection()
    with conn.cursor() as cursor:
        cursor.execute('SELECT * FROM gpus')
        gpus = cursor.fetchall()
    cursor.close()
    conn.close()
    return jsonify(gpus)

if __name__ == '__main__':
    app.run(debug=True)