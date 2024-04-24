import sqlite3
from datetime import datetime, timedelta

def settInnData():
    def les_tekstfil(filnavn):
        with open(filnavn, 'r') as fil:
            data = fil.readlines()
        return data
        
    def sett_inn_data(status, name, timestamp):
        conn = sqlite3.connect('stampData.db')
        cursor = conn.cursor()
        
        cursor.execute('''CREATE TABLE IF NOT EXISTS stampdata
                    (status TEXT,
                    name TEXT,
                    timestamp TEXT)''')
                    
        cursor.execute('''INSERT INTO stampdata (status, name, timestamp)
                    VALUES (?, ?, ?)''', (status, name, timestamp))
        
        conn.commit()
        conn.close()
        
    tekstfil_innhold = les_tekstfil('timecard.txt')

    for i in range(0, len(tekstfil_innhold), 3):
        status = tekstfil_innhold[i].strip().strip(',')
        navn = tekstfil_innhold[i].strip().strip(',')
        #dato = datetime.strptime(tekstfil_innhold[i+1].strip(), '%Y-%m-%d %H:%M:%S')
        dato = tekstfil_innhold[i].strip().strip('\n')
        sett_inn_data(status, navn, dato)

    print("Dataen er satt inn i databasen.")

