import os
from tqdm import tqdm

folders = ['Vul', 'No-Vul']

# 要写入的文件路径
output_file = 'sard.txt'

# 遍历每个文件夹
with open(output_file, 'w') as outfile:
    for folder in tqdm(folders):
        folder_path = os.path.join('./sardnvd', folder)  # 替换为实际的文件夹路径
        if os.path.exists(folder_path):
            # 遍历文件夹中的所有文件
            for root, dirs, files in tqdm(os.walk(folder_path)):
                for file in files:
                    if file.endswith('.c'):  # 仅处理.c文件
                        file_path = os.path.join(root, file)
                        with open(file_path, 'r') as infile:
                            outfile.write(infile.read() + '\n')
