# 环境安装
两种情况，不成功可以换另一种情况尝试
## 1 一口气安装
pip install git+https://github.com/TKOHP/SAELens.git git+https://github.com/TKOHP/TransformerLens.git circuitsvis
## 2 先安装再安装
pip install git+https://github.com/TKOHP/SAELens.git

先安装sae_lens，再卸载原有的Transformer_lens安装

pip install git+https://github.com/TKOHP/TransformerLens.git 

pip install circuitsvis

# 存储，代码code/make_dataset
虚拟环境为mxl，因为后面直接在依赖包的源码上写的，且强制将一些部分变为了cpu，和github上不一样，所有这里就先不用安装github上的sae_lens呢。可以通过pycharm和虚拟环境里的sae_lens包同步来修改源码。
需要多GPU，一个GPU不够加载
## 默认参数
python make_dataset.py --batch_size 1024 --traning_step 3000 --context_size 512 --n_batches_in_buffer 64 --store_batch_size_prompts 16 --save_path "/root/data/sae/dataset" --n_devices 3
这是默认参数，不配置的话按默认参数走，其中save_path不用配置，按照默认的来就行。
device看情况配置。
## 更大的batch_size和训练step
python make_dataset.py --batch_size 4096 --traning_step 30000 --context_size 512 --n_batches_in_buffer 64 --store_batch_size_prompts 16 --save_path "/root/data/sae/dataset" --n_devices 3