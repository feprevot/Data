import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from datetime import datetime
from matplotlib import use
use("TkAgg")


engine = create_engine(
    "postgresql://fprevot:mysecretpassword@localhost:5432/piscineds"
)
query = """
SELECT user_id, user_session, price, event_time
FROM customers
WHERE event_type = 'purchase' AND price > 0
"""
df = pd.read_sql(query, engine)
df["event_time"] = pd.to_datetime(df["event_time"])


latest = df["event_time"].max()
customer_df = (
    df.groupby("user_id")
      .agg(number_of_purchases=("price", "count"),
           total_spent=("price", "sum"),
           last_purchase_date=("event_time", "max"))
      .reset_index()
)
customer_df["recency_days"]   = (latest - customer_df["last_purchase_date"]).dt.days
customer_df["average_basket"] = customer_df["total_spent"] / customer_df["number_of_purchases"]


customer_df["log_purchases"] = np.log1p(customer_df["number_of_purchases"])
customer_df["log_spent"]     = np.log1p(customer_df["total_spent"])

features = ["log_purchases", "log_spent", "recency_days", "average_basket"]
X_scaled = StandardScaler().fit_transform(customer_df[features])

kmeans = KMeans(n_clusters=5, random_state=42)
customer_df["cluster"] = kmeans.fit_predict(X_scaled)

profiles = customer_df.groupby("cluster")[features].mean().round(2)
sizes    = customer_df["cluster"].value_counts().sort_index()
profiles["count"] = sizes


cluster_to_label = {
    0: "Inactive",
    1: "New Customers",
    2: "Silver",
    3: "Platinum",
    4: "Lambdas",
}

customer_df["label"] = customer_df["cluster"].map(cluster_to_label)

plt.figure(figsize=(8,5))
sns.countplot(
    data=customer_df,
    y="label",
    hue="label",                   
    palette="Set2",
    order=customer_df["label"].value_counts().index,
    legend=False 
)

plt.title("segment repartition (KMeans, k=5, log-transform)")
plt.xlabel("Nombre de clients"); plt.ylabel("Segment")
plt.tight_layout(); plt.show()




bubble = (customer_df.groupby("label")
          .agg(med_recency=("recency_days", lambda x: x.median()/30),
               med_monetary=("total_spent", "median"),
               avg_spent=("total_spent", "mean"),
               n=("user_id", "size"))
          .reset_index())
bubble["size"] = bubble["avg_spent"] / bubble["avg_spent"].max() * 1200

plt.figure(figsize=(8,6))
for _, row in bubble.iterrows():
    plt.scatter(row["med_recency"], row["med_monetary"],
                s=row["size"], alpha=.6, label=row["label"])
    plt.text(row["med_recency"]+0.05, row["med_monetary"]+0.4,
             f'{row["label"]}', fontsize=9)
plt.xlabel("Median Recency (months)")
plt.ylabel("Median Monetary Value")
plt.title("Customer segments  Bubble chart (recency vs monetary)")
plt.grid(True, ls="--", alpha=.4)
plt.tight_layout(); plt.show()
