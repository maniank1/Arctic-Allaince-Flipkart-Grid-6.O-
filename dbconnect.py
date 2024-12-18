import sqlite3

# Connect to SQLite database or create it if it doesn't exist
conn = sqlite3.connect('Flipkart_Database.db')

# Create a cursor object using the cursor() method
cursor = conn.cursor()

# Drop table if it already exists
cursor.execute("DROP TABLE IF EXISTS inventory")

# Create table as per requirement
sql ='''CREATE TABLE inventory(
   id INTEGER PRIMARY KEY,
   product_name TEXT,
   product_brand TEXT,
   pack_size TEXT,
   mrp REAL,
   expiry_date TEXT,
   freshness_percentage INTEGER,
   last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)'''
cursor.execute(sql)

# Commit your changes in the database
conn.commit()

# Close the connection
conn.close()
