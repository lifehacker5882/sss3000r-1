import sqlite3
from datetime import datetime, timedelta

def les_tekstfil(filnavn):
	with open(filnavn, 'r') as fil:
		data = fil.readlines()
	return data
	
def sett_inn_data(navn, innstempling, utstempling):
	conn = sqlite3.connect('stampData.db')
	cursor = conn.cursor()
	
	cursor.execute('''CREATE TABLE IF NOT EXISTS innstempling_data
				(id INTEGER PRIMARY KEY AUTOINCREMENT,
				navn TEXT,
				innstempling TEXT,
				utstempling TEXT)''')
				
	cursor.execute('''INSERT INTO innstempling_data (navn, innstempling, utstempling)
				VALUES (?, ?, ?)''', (navn, innstempling, utstempling))
	
	conn.commit()
	conn.close()		
	
tekstfil_innhold = les_tekstfil('data.txt')

for i in range(0, len(tekstfil_innhold), 3):
	navn = tekstfil_innhold[i].strip().strip('"')
	innstempling = datetime.strptime(tekstfil_innhold[i+1].strip(), '%Y-%m-%d %H:%M:%S')	
	utstempling = datetime.strptime(tekstfil_innhold[i+2].strip(), '%Y-%m-%d %H:%M:%S')
	sett_inn_data(navn, innstempling, utstempling)

print("Dataen er satt inn i databasen.")
