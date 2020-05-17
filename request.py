import requests
import json

url = 'http://127.0.0.1:5000/dc'
data = json.dumps({})
send_request = requests.post(url, data)
print(send_request.json())
