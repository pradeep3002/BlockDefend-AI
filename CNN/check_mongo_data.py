from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient('localhost', 27017)
db = client['network_analysis']
collection = db['processed_data']

# Fetch and print the first document
print("First document in MongoDB:")
print(collection.find_one())

# Check if the 'is_malicious' field exists
if 'is_malicious' not in collection.find_one():
    print("Error: 'is_malicious' field is missing in the data.")
else:
    print("Data is ready for training.")