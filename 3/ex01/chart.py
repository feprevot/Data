from matplotlib import use
import matplotlib.pyplot as plt
import pandas as pd
from sqlalchemy import create_engine
import matplotlib.dates as mdates

use('TkAgg')

engine = create_engine("postgresql+psycopg2://fprevot:mysecretpassword@localhost:5432/piscineds")

query = """
SELECT event_time::date AS date, price, user_id
FROM customers
WHERE event_type = 'purchase'
  AND event_time BETWEEN '2022-09-30' AND '2023-03-01'
"""
df = pd.read_sql(query, engine)
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.strftime('%b')

daily_counts = df.groupby('date').size()

plt.figure(figsize=(8, 5))
plt.plot(daily_counts, linewidth=1)
plt.title("Number of customers")
plt.ylabel("Number of customers")
plt.xlabel("Date")
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))
plt.tight_layout()
plt.show()

monthly_sales = df.groupby('month')['price'].sum().reindex(['Oct', 'Nov', 'Dec', 'Jan', 'Feb'])

plt.figure(figsize=(8, 5))
plt.bar(monthly_sales.index, monthly_sales / 1_000_000, color='lightblue')
plt.title("Monthly Sales")
plt.ylabel("Total sales in million of A")
plt.xlabel("Month")
plt.tight_layout()
plt.show()

daily_avg = df.groupby('date').agg({
    'price': 'sum',
    'user_id': pd.Series.nunique
})
daily_avg['avg_spend'] = daily_avg['price'] / daily_avg['user_id']

plt.figure(figsize=(8, 5))
plt.fill_between(daily_avg.index, daily_avg['avg_spend'], alpha=0.4)
plt.title("Average Spend per Customer")
plt.ylabel("Average spend/customers in A")
plt.xlabel("Date")
plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b'))
plt.tight_layout()
plt.show()
