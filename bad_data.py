import json

with open("ShareGPT_V3_unfiltered_cleaned_split.json", "r") as f:
    data = json.load(f)

def get_size(conversation):
    size = len(conversation[0]["value"]) + len(conversation[1]["value"])
    return size

small_requests = [v for v in data if len(v["conversations"]) > 1 and get_size(v["conversations"]) < 50 ]

big_requests = [v for v in data if len(v["conversations"]) > 1 and get_size(v["conversations"]) > 15000 and get_size(v["conversations"]) < 20000]

print(len(small_requests))
print(len(big_requests))

assert len(small_requests) > 900
small_data = small_requests[0:900]

assert len(big_requests) > 100
big_data = big_requests[0:100]

data = small_data + big_data
assert len(data) == 1000

with open("adversarial_sharegpt.json", "w") as f:
    json.dump(data, f)

# print(len(small_requests))
# print(len(big_requests))