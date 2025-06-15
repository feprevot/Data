import psycopg2
from matplotlib import use
import matplotlib.pyplot as plt

use('TkAgg')
conn = psycopg2.connect(
    dbname="piscineds",
    user="fprevot",
    password="mysecretpassword",
    host="localhost",
    port=5432
)

cur = conn.cursor()
cur.execute("""
    SELECT event_type, COUNT(*) 
    FROM customers 
    GROUP BY event_type
""")
results = cur.fetchall()

cur.close()
conn.close()

labels = [row[0] for row in results]
sizes = [row[1] for row in results]

plt.figure(figsize=(7, 7))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
plt.title("User Activity on the Site")
plt.axis('equal')
plt.show()
