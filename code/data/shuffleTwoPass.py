import os
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm
device="cpu"
def load_tensor(file_path):
    with safe_open(file_path, framework='pt',device=device) as f:
        buffer = f.get_tensor("activations")
    return buffer


def create_empty_safetensors_files(output_folder, num_files, tensor_shape):
    os.makedirs(output_folder, exist_ok=False)
    for i in range(num_files):
        file_path = os.path.join(output_folder, f"{i}.safetensors")
        empty_tensor = torch.empty(tensor_shape, dtype=torch.float32)  # 使用全零张量初始化
        save_file({'activations': empty_tensor}, file_path)


def append_to_safetensors_file(file_path, tensor_slice):
    with safe_open(file_path, framework='pt') as f:
        existing_tensor = f.get_tensor('activations')
    new_tensor = torch.cat((existing_tensor, tensor_slice.unsqueeze(0)), dim=0)
    save_file({'activations': new_tensor}, file_path)


def distribute_tensor_elements(input_files,output_folder):
    num_files = len(input_files)
    # 逐个文件处理
    for file_name in tqdm(input_files,desc="pass1"):
        file_path = os.path.join(input_folder, file_name)
        tensor = load_tensor(file_path)
        num_elements = tensor.shape[0]

        # 分配 tensor 元素到输出 tensor
        indices = torch.randperm(num_elements)  # 随机排列索引
        for i, element in enumerate(tensor[indices]):
            target_file_index = i % num_files
            target_file_path = os.path.join(output_folder, f"{target_file_index}.safetensors")
            append_to_safetensors_file(target_file_path, element)
def final_shuffle(output_folder):
    output_files = [file_name for file_name in os.listdir(output_folder) if file_name.endswith('.safetensors')]
    # num_files = len(output_files)
    for file_name in tqdm(output_files,desc="pass2"):
        file_path = os.path.join(output_folder, file_name)
        tensor = load_tensor(file_path)
        tensor = tensor[torch.randperm(tensor.shape[0])]
        save_file({'activations': tensor}, file_path)


def main(input_folder, output_folder):
    # 读取文件名
    input_files = [file_name for file_name in os.listdir(input_folder) if file_name.endswith('.safetensors')]
    num_files = len(input_files)

    # 假设所有 tensor 的形状相同，从第一个文件获取形状
    first_tensor = load_tensor(os.path.join(input_folder, input_files[0]))
    tensor_shape = list(first_tensor.shape)
    tensor_shape[0]=0
    # 创建空白的 safetensors 文件
    create_empty_safetensors_files(output_folder, num_files, tensor_shape)
    del first_tensor
    distribute_tensor_elements(input_files,output_folder)
    final_shuffle(output_folder)

if __name__ == '__main__':

    # 设置输入文件夹和输出文件夹路径
    input_folder = 'D:/project/dataset/activations/unshuffle'
    output_folder = 'D:/project/dataset/activations/shuffle2'

    # 执行函数
    main(input_folder, output_folder)
