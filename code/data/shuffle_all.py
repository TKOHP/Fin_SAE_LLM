import os
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from sklearn.utils import shuffle
from tqdm import tqdm

# 假设你已经安装了 safetensors 和 sklearn 库
# pip install safetensors scikit-learn
device="cpu"
def load_tensors_from_folder(folder_path):
    tensors = []
    for file_name in tqdm(os.listdir(folder_path),desc="读取数据"):
        if file_name.endswith('.safetensors'):
            file_path = os.path.join(folder_path,file_name)
            with safe_open(file_path, framework='pt', device=device) as f:
                activations=f.get_tensor("activations")
                tensors.append(activations)
    tensors=torch.cat(tensors, dim=0)
    return tensors


def save_tensors_to_folder(tensors, folder_path, original_file_names):
    num_files = len(original_file_names)
    tensor_split = torch.chunk(tensors, num_files, dim=0)
    for i, tensor_part in tqdm(enumerate(tensor_split),desc="保存数据"):
        save_path = os.path.join(folder_path, f"{original_file_names[i]}")
        save_file({'activations': tensor_part}, save_path)


def main(input_folder, output_folder):
    # 读取并合并所有 tensor
    tensors = load_tensors_from_folder(input_folder)

    # 打乱 tensor
    tensors = tensors[torch.randperm(tensors.shape[0])]


    os.makedirs(output_folder,exist_ok=False)
    # 获取原始文件名用于保存
    original_file_names = [file_name for file_name in os.listdir(input_folder) if file_name.endswith('.safetensors')]

    # 将打乱后的 tensor 分割并保存到新的目录
    save_tensors_to_folder(tensors, output_folder, original_file_names)

if __name__ == '__main__':
    # 设置输入文件夹和输出文件夹路径
    input_folder = 'D:/project/dataset/activations/unshuffle'
    output_folder = 'D:/project/dataset/activations/shuffle1'

    # 执行主函数
    main(input_folder, output_folder)
