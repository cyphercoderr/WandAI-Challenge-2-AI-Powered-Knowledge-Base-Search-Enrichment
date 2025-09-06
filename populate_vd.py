import requests

BASE_URL = "http://localhost:8001"

# Read entire book from file
with open("fastapi_tutorial.txt", "r", encoding="utf-8") as f:
    book_text = f.read()

# Create request payload
doc = {
    "id": "python_book",
    "text": book_text
}

# Send to /ingest
resp = requests.post(f"{BASE_URL}/ingest", json=doc)
print("Ingest response:", resp.json())
