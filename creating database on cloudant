from cloudant.client import Cloudant
from cloudant.error import CloudantException
from cloudant.result import Result, ResultByKey
client = Cloudant.iam("b3e03381-f624-4db8-a3da-3588bface309-bluemix", "sckyMGqNGv8CX9aIcTDbrhYZYhYBDUfEXAJuXuN8SB1D")
client.connect()
databaseName = "attendance_toqa"
myDatabase = client.create_database(databaseName)
if myDatabase.exists():
   print "'{0}' successfully created.\n".format(databaseName)

   
   
sampleData = [
   [1, "Gabr", "Hazem", 100],
   [2, "Adel", "Muhammad", 40],
   [3, "omar", "Mekawy", 20],
   [4, "mustafa", "azazy", 10],
 ]

# Create documents by using the sample data.
# Go through each row in the array
for document in sampleData:
 # Retrieve the fields in each row.
 number = document[0]
 name = document[1]
 description = document[2]
 temperature = document[3]

 # Create a JSON document that represents
 # all the data in the row.
 jsonDocument = {
     "numberField": number,
     "nameField": name,
     "descriptionField": description,
     "temperatureField": temperature
 }

 # Create a document by using the database API.
 newDocument = myDatabase.create_document(jsonDocument)

 # Check that the document exists in the database.
 if newDocument.exists():
     print "Document '{0}' successfully created.".format(number)
