import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from database.db_connector import get_db_connection
from flask import jsonify
import joblib
from tensorflow import keras
from tensorflow.keras.models import load_model
import pandas as pd

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'static', 'models')
def load_model_app(model_name):
    extensions = ['.pkl', '.keras', '.joblib']
    model_path = None
    
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

def get_components_data(cpu, gpu):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("USE pchelperdb")

    cursor.execute(
        """
        SELECT * FROM act_cpus WHERE cpuName = %s
        """,
        ([cpu])
    )

    user_cpu = cursor.fetchall()

    cursor.execute(
        """
        SELECT * FROM act_gpus WHERE gpuName = %s
        """,
        ([gpu])
    )

    user_gpu = cursor.fetchall()

    if user_cpu:
        cpu_data_list = {
            "cores": user_cpu[0][2],
            "threads": user_cpu[0][3],
            "base_clock": user_cpu[0][4],
            "boost_clock": user_cpu[0][5],
            "tdp": user_cpu[0][9],
            "socket": user_cpu[0][11]
        }

    if user_gpu:
        gpu_data_list = {
            "vram": user_gpu[0][6],
            "tdp": user_gpu[0][-1]
        }    

    cpu_tdp, total_tdp = calculate_total_tdp(user_cpu[0][9],user_gpu[0][-1])

    cursor.close()
    conn.close()

    return cpu_data_list, gpu_data_list, cpu_tdp, total_tdp

def calculate_total_tdp(cpu_tdp, gpu_tdp):
    main = int(cpu_tdp) + int(gpu_tdp) #Компоненты с основным энергопотреблением.
    reserve = main * 0.25 # 25% от основного потребления "запаса по мощности" БП в нагрузке.
    add = 100 # объем мощности для второстепенных компонентов (ОЗУ, Накопители, охлажение).
    # NVME - 10 W
    # RAM - 5 W
    # SATA - 15 W
    # СЖО - 15 W
    # Вентелятор - 5 W
    return cpu_tdp, main + reserve + add

def components_quote(cpu_percent, gpu_percent):

    cq1 = abs(100 - cpu_percent) <= 20
    cq2 = -20 > cpu_percent - 100 > -50
    cq3 = -50 >= cpu_percent - 100
    cq4 = 20 < cpu_percent - 100 < 50
    cq5 = 50 <= cpu_percent - 100
    
    gq1 = abs(100 - gpu_percent) <= 20
    gq2 = -20 > gpu_percent - 100 > -50
    gq3 = -50 >= gpu_percent - 100
    gq4 = 20 < gpu_percent - 100 < 50
    gq5 = 50 <= gpu_percent - 100

    cpu_quotes = {
        cq3 : "Существенная нехватка производительности. Следует рассмотреть новый процессор/платформу для апгрейда.",
        cq2 : "Ощутимая разница в производительности процессора. Стоит задуматься о скором апгейде.",
        cq1 : "Производительность процессора достаточно хорошая для выбранного уровня задач.",
        cq4 : "Хороший запас по производительности на будущее. Апгрейд не требуется.",
        cq5 : "Отличный запас по производительности на будущее. Апгрейд не потребуется еще доглое время на текущем уровне задач."
    }

    gpu_quotes = {
        gq3 : "Существенная нехватка производительности. Следует рассмотреть новую видеокарту для апгрейда.",
        gq2 : "Ощутимая разница в производительности видеокарты. Стоит задуматься о ее скором апгейде.",
        gq1 : "Производительность видеокарты достаточно хорошая для выбранного уровня задач.",
        gq4 : "Хороший запас на будущее по производительности видеокарты. Апгрейд не требуется.",
        gq5 : "Отличный запас на будущее по производительности видеокарты. Апгрейд не потребуется еще доглое время на текущем уровне задач."
    }

    cpu_quote = next((v for k, v in cpu_quotes.items() if k), "Не можем оценить уровень производительности.")
    
    gpu_quote = next((v for k, v in gpu_quotes.items() if k), "Не можем оценить уровень производительности.")

    return cpu_quote, gpu_quote

def get_links_for_components(obj):
    links = {
        'DNS': f'https://www.dns-shop.ru/search/?q={obj}',
        'OZON': f'https://www.ozon.ru/search/?text={obj}',
        'Yandex market': f'https://market.yandex.ru/search?text={obj}',
        'Citilink': f'https://www.citilink.ru/search/?text={obj}',
        'Regard': f'https://www.regard.ru/catalog?search={obj}'
    }

    return links

def get_components_compare(sphere, lvl):
    data = {
        "base": {
            "gaming": {
                "cpu": "Intel Core i5-12400",
                "gpu": "NVIDIA GeForce RTX 4060"
            },
            "3d": {
                "cpu": "AMD Ryzen 7 5800X",
                "gpu": "NVIDIA GeForce RTX 3060 Ti"
            },
            "video": {
                "cpu": "Intel Core i5-13600K",
                "gpu": "NVIDIA GeForce RTX 3060 12GB"
            },
            "ml": {
                "cpu": "AMD Ryzen 7 7700X",
                "gpu": "NVIDIA GeForce RTX 4070"
            },
        },
        "advanced": {
            "gaming": {
                "cpu": "AMD Ryzen 7 7800X3D",
                "gpu": "NVIDIA GeForce RTX 4070"
            },
            "3d": {
                "cpu": "Intel Core i7-13700K",
                "gpu": "NVIDIA GeForce RTX 4070 Ti"
            },
            "video": {
                "cpu": "AMD Ryzen 9 7900X",
                "gpu": "NVIDIA GeForce RTX 4080"
            },
            "ml": {
                "cpu": "Intel Core i9-13900K",
                "gpu": "NVIDIA GeForce RTX 4090"
            },
        },
        "pro": {
            "gaming": {
                "cpu": "Intel Core i9-13900K",
                "gpu": "NVIDIA GeForce RTX 4080"
            },
            "3d": {
                "cpu": "AMD Ryzen 9 7950X3D",
                "gpu": "NVIDIA GeForce RTX 4090"
            },
            "video": {
                "cpu": "Intel Core i9-14900K",
                "gpu": "NVIDIA GeForce RTX 4090"
            },
            "ml": {
                "cpu": "AMD Threadripper PRO 7970X",
                "gpu": "NVIDIA GeForce RTX 4090" # x2
            },
        }
    }

    return data[lvl][sphere]["cpu"], data[lvl][sphere]["gpu"]