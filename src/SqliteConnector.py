import sqlite3

class SqliteConnector():
    def __init__(self) -> None:
        self.conn = sqlite3.connect('icloud_features.db')
        self.cursor = self.conn.cursor()

    def close(self):
        self.conn.close()

    def query(self, query):
        self.cursor.execute(query)
        self.conn.commit()
        return self.cursor.fetchall()
