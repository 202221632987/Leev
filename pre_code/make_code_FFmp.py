import json
import os

from tqdm import tqdm
# open JSON file
with open('../function.json', 'r') as f:
    data_total = json.load(f)
i = 0
print("\n"+str(len(data_total)))
if not os.path.exists("raw_code_FFmp"):
    os.mkdir("raw_code_FFmp")
for data in tqdm(data_total):
    filename = str(data["commit_id"])+"_"+str(data["target"])+".c"
    # 文件有重名现象
    if os.path.exists("raw_code_FFmp"+"/"+filename):
        with open("raw_code_FFmp"+"/"+filename, 'r') as f:
            func = f.read()
            if func == data["func"]:
                print(filename)
                continue
            else:
                with open("raw_code_FFmp" + "/" + str(i)+"_"+str(data["target"])+".c", 'w') as f:
                    f.write(data["func"])
            i = i + 1
    with open("raw_code_FFmp"+"/"+filename, 'w') as f:
        f.write(data["func"])
print(i)

