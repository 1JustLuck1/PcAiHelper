import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from flask import Flask, render_template, jsonify, request, send_file
from database.db_connector import get_db_connection
import joblib
from tensorflow import keras
from tensorflow.keras.models import load_model
import pandas as pd
from service import *
import json
from openpyxl import Workbook
from openpyxl.styles import Font
from io import BytesIO
from datetime import datetime

app = Flask(__name__)

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

    vendor_psu_model = load_model_app('vendor_psu_model')
    vendor_mb_model = load_model_app('vendor_mb_model')
    vendor_cpu_cooler_model = load_model_app('vendor_cpu_cooler_model')

    print("\033[32m{}\033[0m".format("INFO ---> Модели успешно загружены!"))
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
                cpu_main_data = cpu_main_model.predict(cpu_main_processed_data)

                if fields[5] == '1':
                    cpu_main_data = cpu_main_data * 1.125

                cpu_vendor_encoder = mark_to_cpu_model['vendor_encoder'].transform([fields[3]])
                df_cpu_mark_value = pd.DataFrame({
                    'cpuMark': [cpu_main_data],
                    'vendor_encoded': cpu_vendor_encoder
                    })

                final_cpu_model = mark_to_cpu_model['model'].predict(df_cpu_mark_value)
                cpu = mark_to_cpu_model['cpuName_encoder'].inverse_transform(final_cpu_model)[0]

                # print(cpu)
            except Exception as e:
                print(f"\033[31mСPU MODEL ERROR ---> {e}\033[0m")

            try:
                gpu_main_processed_data = gpu_main_preprocessor.transform(data_gpu_main)
                # print(gpu_main_processed_data)
                gpu_main_data = gpu_main_model.predict(gpu_main_processed_data)
                
                if fields[5] == '1':
                    gpu_main_data = gpu_main_data * 1.125
                
                gpu_vendor_encoder = mark_to_gpu_model['vendor_encoder'].transform([fields[4]])
                gpu_type_encoder = mark_to_gpu_model['type_encoder'].transform([fields[-1]])

                df_gpu_mark_value = pd.DataFrame({
                    'gpuMark': [gpu_main_data],
                    'vendor_encoded': gpu_vendor_encoder,
                    'type_encoded' : gpu_type_encoder
                    })

                final_gpu_model = mark_to_gpu_model['model'].predict(df_gpu_mark_value)
                gpu = mark_to_gpu_model['gpuName_encoder'].inverse_transform(final_gpu_model)[0]

                # print(gpu)
            except Exception as e:
                print(f"\033[31mGPU MODEL ERROR ---> {e}\033[0m")
                return jsonify({"error": str(e)}), 500
            
            cpudata, gpudata, cpu_tdp, total_tdp = get_components_data(cpu, gpu)
            
            cpudata = json.dumps(cpudata)
            gpudata = json.dumps(gpudata)

            cpu_json = json.loads(cpudata)
            gpu_json = json.loads(gpudata)

            cpu_links = get_links_for_components(cpu)
            gpu_links = get_links_for_components(gpu)

            cpu_links = json.loads(json.dumps(cpu_links))
            gpu_links = json.loads(json.dumps(gpu_links))

            ram = ""
            if fields[1] == "min":
                ram = "ADATA XPG Lancer Blade 8"
            elif fields[1] == "base":
                ram = "ADATA XPG Lancer Blade 16"
            elif fields[1] == "medium":
                ram = "ADATA XPG Lancer Blade 32"
            elif fields[1] == "max":
                ram = "ADATA XPG Lancer Blade 64"

            try:
                psu_vendor_predict = pd.DataFrame({
                    'price': [fields[1]],
                    'watt': [total_tdp]
                })
                psu_vendor = vendor_psu_model.predict(psu_vendor_predict)
            except Exception as e:
                print(f"\033[31mPSU MODEL ERROR ---> {e}\033[0m")
            
            psu_links = get_links_for_components(psu_vendor[0])
            psu_links = json.loads(json.dumps(psu_links))
            
            try:
                mb_vendor_predict = pd.DataFrame({
                    'cpuMark': [cpu_main_data], #ПРОПИСАТЬ ЛОГИКУ
                    'socket': [cpu_json["socket"]],
                    'price': [fields[1]]
                })
                print("SSSS", mb_vendor_predict)
                mb_vendor = vendor_mb_model.predict(mb_vendor_predict)
            except Exception as e:
                print(f"\033[31mMOTHERBOARD MODEL ERROR ---> {e}\033[0m")

            mb_links = get_links_for_components(mb_vendor[0])
            mb_links = json.loads(json.dumps(mb_links))

            try:
                cpu_cooler_predict = pd.DataFrame({
                    'mtp': [cpu_tdp],
                    'price': [fields[1]],
                    'type': ['air'] #ПРОПИСАТЬ ЛОГИКУ
                })
                cpu_cooler_vendor = vendor_cpu_cooler_model.predict(cpu_cooler_predict)
                print("cpu_cooler_vendor: ", cpu_cooler_vendor)
            except Exception as e:
                print(f"\033[31mCPU COOLER MODEL ERROR ---> {e}\033[0m")

            cpu_cooler_links = get_links_for_components(cpu_cooler_vendor[0])
            cpu_cooler_links = json.loads(json.dumps(cpu_cooler_links))

            return jsonify({
                "cpu": cpu,
                "gpu": gpu,
                "ram": ram,
                "total_tdp": total_tdp,
                "psu_vendor": psu_vendor[0],
                "mb_vendor": mb_vendor[0],
                "cpu_cooler_vendor": cpu_cooler_vendor[0],
                "cpud": cpu_json,
                "gpud": gpu_json,
                "cpu_links": cpu_links,
                "gpu_links": gpu_links,
                "psu_links": psu_links,
                "mb_links": mb_links,
                "cpu_cooler_links": cpu_cooler_links
            })
        except Exception as e:
            print(str(e))
            return jsonify({"error": str(e)}), 500

@app.route('/api/evaluate', methods=["POST"])
def api_evaluate():
    data = request.get_json()
    cpu = data.get('cpu')
    gpu = data.get('gpu')
    sphere = data.get('sphere')
    lvl = data.get('lvl')

    cpu_compare, gpu_compare = get_components_compare(sphere, lvl) 

    cpu_mark = 0
    cpu_compare_mark = 0
    gpu_mark = 0
    gpu_compare_mark = 0

    conn = get_db_connection()
    with conn.cursor() as cursor:
        cursor.execute('SELECT cpuMark FROM hist_cpus where cpuName=%s',([cpu]))
        cpu_mark = cursor.fetchall()
        cursor.execute('SELECT cpuMark FROM hist_cpus where cpuName=%s',([cpu_compare]))
        cpu_compare_mark = cursor.fetchall()
        cursor.execute('SELECT gpuMark FROM hist_gpus where gpuName=%s',([gpu]))
        gpu_mark = cursor.fetchall()
        cursor.execute('SELECT gpuMark FROM hist_gpus where gpuName=%s',([gpu_compare]))
        gpu_compare_mark = cursor.fetchall()
    cursor.close()
    conn.close()

    cpu_mark = json.loads(json.dumps(cpu_mark))
    cpu_compare_mark = json.loads(json.dumps(cpu_compare_mark))
    gpu_mark = json.loads(json.dumps(gpu_mark))
    gpu_compare_mark = json.loads(json.dumps(gpu_compare_mark))

    cpu_percent = round(round(cpu_mark[0][0] / cpu_compare_mark[0][0], 2) * 100)
    gpu_percent = round(round(gpu_mark[0][0] / gpu_compare_mark[0][0], 2) * 100)

    cpu_quote, gpu_quote = components_quote(cpu_percent, gpu_percent)

    return jsonify({
        "cpu_percent": cpu_percent,
        "gpu_percent": gpu_percent,
        "cpu_quote" : cpu_quote,
        "gpu_quote" : gpu_quote
        })

@app.route('/download-excel', methods=['POST'])
def download_excel():
    try:
        components = request.json
        
        wb = Workbook()
        ws = wb.active
        ws.title = "Компоненты ПК"
        
        headers = ["Компонент", "Модель/Характеристики"]
        ws.append(headers)
        
        bold_font = Font(bold=True)
        for cell in ws[1]:
            cell.font = bold_font
        
        ws.append(["Процессор", components.get('cpu', 'Не указано')])
        ws.append(["Видеокарта", components.get('gpu', 'Не указано')])
        ws.append(["Оперативная память", components.get('ram', 'Не указано')])
        ws.append(["Блок питания", f"{components.get('psu', '?')}W ({components.get('psu_vendor', 'Не указано')})"])
        ws.append(["Материнская плата", components.get('mb_vendor', 'Не указано')])
        ws.append(["Кулер процессора", components.get('cpu_cooler_vendor', 'Не указано')])
        
        for col in ws.columns:
            max_length = 0
            for cell in col:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(cell.value)
                except:
                    pass
            adjusted_width = (max_length + 2)
            ws.column_dimensions[col[0].column_letter].width = adjusted_width
        
        buffer = BytesIO()
        wb.save(buffer)
        buffer.seek(0)
        
        return send_file(
            buffer,
            as_attachment=True,
            download_name='pc_components.xlsx',
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    
    except Exception as e:
        return {"error": str(e)}, 500

@app.route('/api/cpus')
def get_cpus():
    conn = get_db_connection()
    with conn.cursor() as cursor:
        cursor.execute('SELECT cpuName FROM hist_cpus')
        cpus = cursor.fetchall()
    cursor.close()
    conn.close()
    return jsonify([
        {"id": idx, "name": cpu[0]}
        for idx, cpu in enumerate(cpus, 1)
    ])

@app.route('/api/gpus')
def get_gpus():
    conn = get_db_connection()
    with conn.cursor() as cursor:
        cursor.execute('SELECT gpuName FROM hist_gpus')
        gpus = cursor.fetchall()
    cursor.close()
    conn.close()
    return jsonify([
        {"id": idx, "name": gpu[0]}
        for idx, gpu in enumerate(gpus, 1)
    ])

if __name__ == '__main__':
    app.run(debug=True)