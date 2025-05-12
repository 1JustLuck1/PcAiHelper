from service import load_model_app

def load_models():
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
        vendor_psu_model = None
        vendor_mb_model = None
        vendor_cpu_cooler_model = None

    return cpu_preload_model, gpu_preload_model, cpu_main_preprocessor, gpu_main_preprocessor, cpu_main_model, gpu_main_model, mark_to_cpu_model, mark_to_gpu_model, vendor_psu_model, vendor_mb_model, vendor_cpu_cooler_model