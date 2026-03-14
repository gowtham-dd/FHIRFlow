import json
import requests

# Test with a simple claim
test_claim = {
    "patient_id": "pat001",
    "code": "J3420",
    "dose": 1500
}

# Save to a temp file
with open('test_claim.json', 'w') as f:
    json.dump(test_claim, f)

# Upload to your app
with open('test_claim.json', 'rb') as f:
    files = {'file': ('test_claim.json', f, 'application/json')}
    response = requests.post('http://localhost:5000/api/upload', files=files)
    
print(f"Status: {response.status_code}")
print(f"Response: {response.text}")