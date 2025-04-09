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
    return (int(cpu_tdp) + int(gpu_tdp)) * 1.5 + 50