from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
 
uri = "mongodb+srv://jonab89:thomas123456@sss3000.sdur60z.mongodb.net/?retryWrites=true&w=majority&appName=SSS3000"
 
# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))
 
# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)