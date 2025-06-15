import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine
from matplotlib import use

engine = create_engine('postgresql://fprevot:mysecretpassword@localhost:5432/piscineds')
use('TkAgg')
query = """
SELECT
    user_id,
    COUNT(*) AS number_of_purchases,
    COUNT(DISTINCT user_session) AS number_of_sessions,
    SUM(price) AS total_spent
FROM customers
WHERE event_type = 'purchase' AND price IS NOT NULL
GROUP BY user_id
"""
df = pd.read_sql(query, engine)

X = df[["number_of_purchases", "number_of_sessions", "total_spent"]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertia = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(6, 4))
plt.plot(k_range, inertia, marker='o')
plt.title("The Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("Inertia")
plt.tight_layout()
plt.show()
