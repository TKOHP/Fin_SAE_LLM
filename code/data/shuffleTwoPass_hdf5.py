import os
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm
import h5py

device="cpu"
def load_tensor(file_path):
    with safe_open(file_path, framework='pt',device=device) as f:
        buffer = f.get_tensor("activations")
    return buffer
def load_h5(file_path):
    with safe_open(file_path, framework='pt',device=device) as f:
        buffer = f.get_tensor("activations")
    return buffer

def create_empty_hdf5_files(output_folder, num_files, tensor_shape):
    os.makedirs(output_folder, exist_ok=False)
    for i in range(num_files):
        file_path = os.path.join(output_folder, f"{i}.h5")  # 使用全零张量初始化
        with h5py.File(file_path, 'a') as f:
            if 'mydataset' not in f:
                # 创建一个可以扩展的数据集
                # f.create_dataset('mydataset', (0,), maxshape=(None,))
                f.create_dataset('mydataset', (0,tensor_shape[1],tensor_shape[2]), maxshape=(None,tensor_shape[1],tensor_shape[2]))


def append_to_hdf5_file(file_path, tensor_slice):
    num=tensor_slice.shape[0]
    with h5py.File(file_path, 'r+') as f:
        dset = f['mydataset']
        current_shape = dset.shape
        # 扩展数据集
        dset.resize((dset.shape[0] + num,current_shape[1],current_shape[2]))

        # 写入新的数据
        dset[-num:] = tensor_slice.numpy()

def distribute_tensor_elements(input_files,output_folder):
    num_files = len(input_files)
    # 逐个文件处理

    for file_name in tqdm(input_files,desc="pass1"):
        file_path = os.path.join(input_folder, file_name)
        tensor = load_tensor(file_path)
        num_elements = tensor.shape[0]
        # 查看每个tensor的大小
        total_tensors = len(tensor)
        tensors_per_file = total_tensors // num_files
        remainder = total_tensors % num_files
        # 分配 tensor 元素到输出 tensor
        indices = torch.randperm(num_elements)  # 随机排列索引
        tensor=tensor[indices]
        current_start_index = 0
        for j in range(num_files):
            current_end_index = current_start_index + tensors_per_file + (1 if j < remainder else 0)
            current_tensors = tensor[current_start_index:current_end_index]
            current_start_index = current_end_index
            target_file_path = os.path.join(output_folder, f"{j}.h5")
            append_to_hdf5_file(target_file_path, current_tensors)

def final_shuffle(output_folder):
    output_files = [file_name for file_name in os.listdir(output_folder) if file_name.endswith('.h5')]
    # num_files = len(output_files)
    for file_name in tqdm(output_files,desc="pass2"):
        h5_path = os.path.join(output_folder, file_name)
        file_path=h5_path.replace(".h5",".safetensors")
        with h5py.File(h5_path, 'r') as f:
            # 假设你要读取的数据集是三维的
            data = f["mydataset"][...]  # 读取整个数据集

        # 将数据转换为 PyTorch Tensor
        tensor = torch.tensor(data)
        tensor = tensor[torch.randperm(tensor.shape[0])]
        save_file({'activations': tensor}, file_path)
        os.remove(h5_path)


def main(input_folder, output_folder):
    # 读取文件名
    input_files = [file_name for file_name in os.listdir(input_folder) if file_name.endswith('.safetensors')]
    num_files = len(input_files)

    # 假设所有 tensor 的形状相同，从第一个文件获取形状
    first_tensor = load_tensor(os.path.join(input_folder, input_files[0]))
    tensor_shape = list(first_tensor.shape)
    tensor_shape[0]=0
    # 创建空白的 safetensors 文件
    create_empty_hdf5_files(output_folder, num_files, tensor_shape)
    del first_tensor
    distribute_tensor_elements(input_files,output_folder)
    final_shuffle(output_folder)

if __name__ == '__main__':

    # 设置输入文件夹和输出文件夹路径
    input_folder = '/root/data/sae/dataset/test_unshuffle'
    output_folder = '/root/data/sae/dataset/test_shuffle1'

    # 执行函数
    main(input_folder, output_folder)
