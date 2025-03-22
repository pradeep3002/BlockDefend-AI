import pandas as pd
from pymongo import MongoClient
import logging

# Initialize logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DataExport")

# Connect to MongoDB
def connect_to_mongodb(host, port, db_name, collection_name):
    try:
        client = MongoClient(host, port)
        db = client[db_name]
        collection = db[collection_name]
        return client, collection
    except Exception as e:
        logger.error(f"❌ Error connecting to MongoDB: {e}")
        return None, None

# Retrieve data from MongoDB
def retrieve_data(collection):
    try:
        cursor = collection.find({}, {"_id": 0})  # Exclude MongoDB's `_id` field
        df = pd.DataFrame(list(cursor))
        return df
    except Exception as e:
        logger.error(f"❌ Error retrieving data from MongoDB: {e}")
        return None

# Save data to CSV
def save_to_csv(df, filename):
    try:
        df.to_csv(filename, index=False)
        logger.info(f"✅ Data successfully exported to {filename}")
    except Exception as e:
        logger.error(f"❌ Error saving data to {filename}: {e}")

# Main function
def main():
    host = "localhost"
    port = 27017
    db_name = "network_analysis"
    collection_name = "processed_data"
    filename = "network_data.csv"

    client, collection = connect_to_mongodb(host, port, db_name, collection_name)
    if client is None:
        return

    df = retrieve_data(collection)
    if df is None:
        return

    if df.empty:
        logger.warning("No data found in MongoDB. Ensure that processed_data contains records.")
    else:
        save_to_csv(df, filename)

    client.close()
    logger.info("MongoDB connection closed.")

if __name__ == "__main__":
    main()