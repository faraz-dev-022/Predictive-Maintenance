# test_all.py
import requests, numpy as np, json

print("=" * 50)
print("TEST 1 — Health check")
try:
    r = requests.get("http://localhost:8000/")
    print("Status:", r.status_code)
    print("Response:", r.json())
except Exception as e:
    print(f"Error connecting to API: {e}")

print("\nTEST 2 — Healthy engine (low risk)")
# Engine early in its life — should return LOW risk
with open("models/config.json") as f:
    config = json.load(f)

n_features = len(config["sequence_cols"])
seq_len    = config["sequence_length"]

# Simulate healthy sensor readings (all zeros = normalised healthy baseline)
healthy_window = np.zeros((seq_len, n_features)).tolist()
r = requests.post("http://localhost:8000/predict",
                  json={"cycles": healthy_window})
print("Status:", r.status_code)
print("Response:", r.json())

print("\nTEST 3 — Failing engine (high risk)")
# Simulate degraded readings (ones = high sensor values)
failing_window = np.ones((seq_len, n_features)).tolist()
r = requests.post("http://localhost:8000/predict",
                  json={"cycles": failing_window})
print("Status:", r.status_code)
print("Response:", r.json())

print("\nTEST 4 — Wrong input shape (should return 422 error)")
bad_payload = {"cycles": [[1, 2, 3]]}  # wrong shape
r = requests.post("http://localhost:8000/predict",
                  json=bad_payload)
print("Status:", r.status_code, "(expected 422)")

print("\n" + "=" * 50)
print("All tests done!")
