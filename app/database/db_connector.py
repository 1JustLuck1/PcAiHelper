import mysql.connector
from mysql.connector import Error

def get_db_connection():
    try:
        conn = mysql.connector.connect(
            host='localhost',
            database='pchelperdb',
            user='root',
            password='Mo$cow2025'
        )
        return conn
    except Error as e:
        print(e)
        return None