from database.db_connector import get_db_connection
from flask import jsonify

def get_components_data(cpu, gpu):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("USE pchelperdb")

    cursor.execute(
        """
        SELECT * FROM cpus WHERE name = %s
        """,
        ([cpu])
    )

    user_cpu = cursor.fetchall()

    cursor.execute(
        """
        SELECT * FROM gpus WHERE name = %s
        """,
        ([gpu])
    )

    user_gpu = cursor.fetchall()

    if user_cpu:
        cpu_data_list = {
            "cores": user_cpu[0][1],
            "threads": user_cpu[0][2],
            "base_clock": user_cpu[0][3],
            "boost_clock": user_cpu[0][4],
            "tdp": user_cpu[0][-1]
        }

    if user_gpu:
        gpu_data_list = {
            "vram": user_gpu[0][6],
            "tdp": user_gpu[0][-1]
        }    

    total_tdp = calculate_total_tdp(user_cpu[0][-1],user_gpu[0][-1])

    cursor.close()
    conn.close()

    return cpu_data_list, gpu_data_list, total_tdp

def calculate_total_tdp(cpu_tdp, gpu_tdp):
    main = int(cpu_tdp) + int(gpu_tdp) #Компоненты с основным энергопотреблением.
    reserve = main * 0.25 # 25% от основного потребления "запаса по мощности" БП в нагрузке.
    add = 100 # объем мощности для второстепенных компонентов (ОЗУ, Накопители, охлажение).
    # NVME - 10 W
    # RAM - 5 W
    # SATA - 15 W
    # СЖО - 15 W
    # Вентелятор - 5 W
    return main + reserve + add

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