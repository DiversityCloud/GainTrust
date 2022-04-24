import os
import shutil

save_dir = "./result"
for a_dir in os.listdir("./save"):
    os.makedirs(f"{save_dir}/{a_dir}", exist_ok=True)
    for file in os.listdir(f"./save/{a_dir}"):
        if file.endswith("txt"):
            shutil.copy(f"./save/{a_dir}/{file}", f"{save_dir}/{a_dir}/{file}")