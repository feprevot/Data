import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import use
from sqlalchemy import create_engine
use('TkAgg')

engine = create_engine('postgresql://fprevot:mysecretpassword@localhost:5432/piscineds')

query = """
SELECT DISTINCT user_id, user_session, price
FROM customers
WHERE event_type = 'purchase'
"""
df = pd.read_sql(query, engine)

purchase_counts = df.groupby("user_id")["price"].count()

bins = [0, 10, 20, 30, 40, float("inf")]
labels = ["0-10", "11-20", "21-30", "31-40", "41+"]

purchase_bins = pd.cut(purchase_counts, bins=bins, labels=labels, include_lowest=True)
purchase_distribution = purchase_bins.value_counts().sort_index()

purchase_distribution.plot(kind="bar", color="mediumseagreen")
plt.xlabel("Frequency of purchases")
plt.ylabel("NUmber of users")
plt.title("Number of users by frequency of purchases")
plt.tight_layout()
plt.show()

total_spent_per_user = df.groupby("user_id")["price"].sum()

bins = [0, 50, 100, 150, 200, float("inf")]

binned_totals = pd.cut(total_spent_per_user, bins=bins, include_lowest=True)

spend_distribution = binned_totals.value_counts().sort_index()

spend_distribution.plot(kind="bar", color="cornflowerblue")
plt.xlabel("Total spend")
plt.ylabel("NUmber of users")
plt.title("Distribution of total spend per user")
plt.tight_layout()
plt.show()