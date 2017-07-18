from pymongo import MongoClient
from datetime import datetime

# client = MongoClient()
# db = client.test
#
# result = db.restaurants.insert_one(
#     {
#         "address": {
#             "street": "2 Avenue",
#             "zipcode": "10075",
#             "building": "1480",
#             "coord": [-73.9557413, 40.7720266]
#         },
#         "borough": "Manhattan",
#         "cuisine": "Italian",
#         "grades": [
#             {
#                 "date": datetime.strptime("2014-10-01", "%Y-%m-%d"),
#                 "grade": "A",
#                 "score": 11
#             },
#             {
#                 "date": datetime.strptime("2014-01-16", "%Y-%m-%d"),
#                 "grade": "B",
#                 "score": 17
#             }
#         ],
#         "name": "Vella",
#         "restaurant_id": "41704620"
#     }
# )
#
# print result.inserted_id
#
# for entry in db.restaurants.find():
#     print entry
#
# print type(db.restaurants.find())
# print len([entry for entry in db.restaurants.find()])

# print [en for en in db.restaurants.find({'_id': result.inserted_id})]


client = MongoClient()
db = client.twitter_data

result = db.test_coll.insert_one(
    {
        "address": {
            "street": "2 Avenue",
            "zipcode": "10075",
            "building": "1480",
            "coord": [-73.9557413, 40.7720266]
        },
        "borough": "Manhattan",
        "cuisine": "Italian",
        "grades": [
            {
                "date": datetime.strptime("2014-10-01", "%Y-%m-%d"),
                "grade": "A",
                "score": 11
            },
            {
                "date": datetime.strptime("2014-01-16", "%Y-%m-%d"),
                "grade": "B",
                "score": 17
            }
        ],
        "name": "Vella",
        "restaurant_id": "41704620"
    }
)
