
import os
import mysql.connector

class Database:
    def __init__(self):
        self.connection = mysql.connector.connect(
            host=os.getenv('DB_HOST'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            database=os.getenv('DB_NAME')
        )
        self.cursor = self.connection.cursor()

    # def last_detection_time(self, name):
    #     query = "SELECT datetime FROM DetectedFaces WHERE name=%s ORDER BY datetime DESC LIMIT 1"
    #     self.cursor.execute(query, (name,))
    #     result = self.cursor.fetchone()
    #     return result[0] if result else None
    
    def insert_detected_face(self, datetime, name, image_path):
        query = "INSERT INTO DetectedFaces (datetime, name, image_path) VALUES (%s, %s, %s)"
        values = (datetime, name, image_path)
        self.cursor.execute(query, values)
        self.connection.commit()

    def close(self):
        self.cursor.close()
        self.connection.close()
