import pandas as pd
import os
from tqdm import tqdm
data = pd.read_csv("../MSR_data_cleaned.csv")
data_length = data.shape[0]
print(data_length)
if not os.path.exists("../data/raw_code_Fan"):
    os.mkdir("../data/raw_code_Fan")
vul_num = 0
for i in tqdm(range(data_length)):
    func_after = data.at[i, "func_after"]
    func_before = data.at[i,"func_before"]
    vul = data.at[i,"vul"]
    if vul ==1:
        vul_num = vul_num+1
    data_name = str(i)+"_"+str(vul)+".c"
    if func_after != func_before and vul != 1:
        print(data_name)
    filename = data_name
    # 文件有重名现象
    if os.path.exists("../data/raw_code_Fan" + "/" + filename):
        with open("../data/raw_code_Fan" + "/" + filename, 'r') as f:
            func = f.read()
            if func == func_after:
                print(filename)
                continue
            else:
                with open("../data/raw_code_Fan" + "/" +filename, 'w') as f:
                    f.write(func_before)
            i = i + 1
    with open("../data/raw_code_Fan" + "/" + filename, 'w') as f:
        f.write(func_before)
print(vul_num)
