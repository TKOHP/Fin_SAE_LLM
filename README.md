# 环境安装
两种情况，不成功可以换另一种情况尝试
## 1 一口气安装
pip install git+https://github.com/TKOHP/SAELens.git git+https://github.com/TKOHP/TransformerLens.git circuitsvis
## 2 先安装再安装
pip install git+https://github.com/TKOHP/SAELens.git

先安装sae_lens，再卸载原有的Transformer_lens安装。
pip uninstall transformer_lens

pip install git+https://github.com/TKOHP/TransformerLens.git 

pip install circuitsvis
# 计算显存占用，代码code/calculate_ram.py
calculate()计算SAE显存占用
shuffle_ram()计算shuffle的占用，为固定开销=19*buffer_size
# 存储，代码code/make_dataset
可以通过pycharm和虚拟环境里的sae_lens包同步来修改源码。
需要多GPU，一个GPU不够加载。
## 虚拟环境配置
pip install git+https://github.com/TKOHP/SAElens_cache.git

先安装sae_lens，再卸载原有的Transformer_lens安装。
pip uninstall transformer_lens

pip install git+https://github.com/TKOHP/TransformerLens.git 

pip install circuitsvis
## 使用
直接编辑代码中的cfg，文件名自定义，起个好记的就行，会创建一个和文件名相同的json文件，保存在同一个父目录下。
new_cached_activations_path路径不要以"/"结尾就行，因为要拼接.json。
## code/make_dataset_old，之前旧的，使用命令行参数运行的代码，现在不适用
### 默认参数
python make_dataset.py --batch_size 1024 --traning_step 3000 --context_size 512 --n_batches_in_buffer 64 --store_batch_size_prompts 16 --save_path "/root/data/sae/dataset" --n_devices 3
这是默认参数，不配置的话按默认参数走，其中save_path不用配置，按照默认的来就行。
device看情况配置。
### 更大的batch_size和训练step
python make_dataset.py --batch_size 4096 --traning_step 30000 --context_size 512 --n_batches_in_buffer 64 --store_batch_size_prompts 16 --save_path "/root/data/sae/dataset" --n_devices 3
# 可视化HTML生成，code/vis_maker
## 默认参数配置如下
python vis_maker.py --model_name /root/data/sae/LLMmodel/XuanYuan-6B-Chat --sae /root/data/sae/sae_checkpoint/2eizws4q --sae_b /root/data/sae/sae_checkpoint/2eizws4q --hook_point blocks.0.hook_mlp_out --save_html_path /root/data/sae/vis_html/vis.html 
save_html_path 每次定义一个新的文件名称，不让会覆盖旧的
保存文件命名是XuanYuan_6B_Chat_a_2eizws4q_b_2eizws4q_blocks.0.hook_mlp_out
# 评估文件，code/evals
SAElens源码中的文件，稍微修改了加载模型部分，以适用金融LLM和多GPU。
