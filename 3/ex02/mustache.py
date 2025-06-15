import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import use
import seaborn as sns
from sqlalchemy import create_engine
use('TkAgg')

engine = create_engine('postgresql://fprevot:mysecretpassword@localhost:5432/piscineds')

query = """
SELECT user_id, user_session, price
FROM customers
WHERE event_type = 'purchase' AND price IS NOT NULL
"""
df = pd.read_sql(query, engine)

count = df['price'].count()
print(f"count {count}")
mean_price = df['price'].mean()
print(f"mean  {mean_price:.2f}")
std_variance = df['price'].std()
print(f"std  {std_variance:.2f}")
min_price = df['price'].min()
print(f"min  {min_price:.2f}")
tf_percentile = df['price'].quantile(0.25)
print(f"25%  {tf_percentile:.2f}")
f_percentile = df['price'].quantile(0.50)
print(f"50%  {f_percentile:.2f}")
sf_percentile = df['price'].quantile(0.75)
print(f"75%  {sf_percentile:.2f}")
max_price = df['price'].max()
print(f"max  {max_price:.2f}")

sns.set(style="whitegrid")
plt.figure(figsize=(8, 5))
sns.boxplot(x=df['price'], color="gray", width=0.3, fliersize=1, linewidth=1)
plt.title("Box plot of purchase prices", fontsize=13)
plt.ylabel("Price")
plt.xlabel("")
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


sns.set(style="whitegrid")
plt.figure(figsize=(8, 5))
sns.boxplot(x=df['price'], color="gray", width=0.3, fliersize=2, linewidth=1, showfliers=False) 
plt.title("Box plot of purchase prices", fontsize=13)
plt.ylabel("Price")
plt.xlabel("")
plt.xlim(-1, 13)
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

session_totals = df.groupby(["user_id", "user_session"])["price"].sum().reset_index()
session_totals.rename(columns={"price": "session_total"}, inplace=True)
avg_basket_per_user = session_totals.groupby("user_id")["session_total"].mean()

sns.set(style="whitegrid")
plt.figure(figsize=(8, 5))
sns.boxplot(
    x=avg_basket_per_user,
    color="gray",
    width=0.3,
    fliersize=2,
    linewidth=1,
    showfliers=True
)
plt.title("Box plot of average basket total per session (per user)", fontsize=13)
plt.xlabel("Average basket total")
plt.yticks([])
plt.xlim(-20, 150)
plt.grid(True, axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
