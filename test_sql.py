import duckdb
import pandas as pd

# Connect to the database file
con = duckdb.connect(database='yelp_dw.db', read_only=True) # Use read_only=True for querying

# Example: List tables
tables = con.execute("SHOW TABLES;").fetchdf()
print("Tables in the database:")
print(tables)

# Example: Query data and fetch as a Pandas DataFrame
query = "SELECT * FROM fact_user_friend LIMIT 5;"
user_info = con.execute(query).fetchdf()
print("\nUser Info (Sample):")
print(user_info)

# Close the connection
con.close()
