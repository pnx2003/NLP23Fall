import os
import subprocess

# 指定wandb文件夹的路径
wandb_folder_path = "../wandb"

# 获取wandb文件夹下的所有文件名
files = os.listdir(wandb_folder_path)

# 遍历每个文件
for file in files:
    # 构建命令行命令
    command = f"wandb sync {file}"

    # 执行命令行命令
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"命令执行失败：{e}")
