import json
from statistics import mean
from copy import deepcopy

with open("ShareGPT_V3_unfiltered_cleaned_split.json", "r") as f:
    data = json.load(f)

def get_size(conversation):
    size = len(conversation[0]["value"]) + len(conversation[1]["value"])
    return size

small_requests = [v for v in data if len(v["conversations"]) > 1 and get_size(v["conversations"]) < 60 ]
big_requests = [v for v in data if len(v["conversations"]) > 1 and get_size(v["conversations"]) > 15000 and get_size(v["conversations"]) < 20000]


really_big_requests = deepcopy(big_requests)
# For the _really_ big requests
for b in really_big_requests:
    prompt_size = len(b["conversations"][0]["value"])
    # Make prompt close to 200k chars
    multiplier = 200000 // prompt_size
    b["conversations"][0]["value"] = b["conversations"][0]["value"] * multiplier
    # print(len(b["conversations"][0]["value"]))

print("Average sizes")
print(mean(len(v["conversations"][0]["value"]) for v in small_requests))
print(mean(len(v["conversations"][0]["value"]) for v in big_requests))
print(mean(len(v["conversations"][0]["value"]) for v in really_big_requests))

print("Lengths")
print(len(small_requests))
print(len(big_requests))

assert len(small_requests) > 990
assert len(big_requests) > 100

# Medium case
medium_data = small_requests[0:900] + big_requests[0:100]
# Large case
large_data = small_requests[0:990] + really_big_requests[0:10]
# Mixed case
mixed_data = small_requests[0:850] + big_requests[0:140] + really_big_requests[0:10]

with open("medium_sharegpt.json", "w") as f:
    json.dump(medium_data, f)
with open("large_sharegpt.json", "w") as f:
    json.dump(large_data, f)
with open("mixed_sharegpt.json", "w") as f:
    json.dump(mixed_data, f)

