# 环境安装->只用于训练，存储和可视化参考别处
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
# 数据部分，代码code/data
可以通过pycharm和虚拟环境里的sae_lens包同步来修改源码。
需要多GPU，一个GPU不够加载。
## 虚拟环境配置
pip install git+https://github.com/TKOHP/SAElens_cache.git

先安装sae_lens，再卸载原有的Transformer_lens安装。
pip uninstall transformer_lens

pip install git+https://github.com/TKOHP/TransformerLens.git 

pip install circuitsvis
## 数据生成使用，code/data/make_dataset
直接编辑代码中的cfg，文件名自定义，起个好记的就行，会创建一个和文件名相同的json文件，保存在同一个父目录下。
new_cached_activations_path路径不要以"/"结尾就行，因为要拼接.json。

## 数据shuffle1，code/data/shuffle_all
所有buffer加载到内存，组成一个tensor后shuffle，再分开
## 数据shuffle2，code/data/shuffleTwoPass
pass1：
* 创建n个空白目标buffer。
* 对源shuffler的每一个源buffer进行以下处理：
  * shuffle这个buffer
  * 按顺序遍历n个空白目标buffer，将源buffer中的元素依次拼接上去。
* 对目标buffer进行遍历，在每一个buffer内shuffle。

pass2：每个buffer独立shuffle

# 可视化HTML生成，code/vis_maker
## 虚拟环境配置
在环境安装的基础上，安装包：

pip install git+https://github.com/TKOHP/sae_vis.git

然后卸载transformer_lens。pip uninstall transformer_lens

pip install git+https://github.com/TKOHP/TransformerLens_sae_vis.git
note:transformers版本为4.38.1.
## 默认参数配置如下
python vis_maker.py --model_name /root/data/sae/LLMmodel/XuanYuan-6B-Chat --sae /root/data/sae/sae_checkpoint/2eizws4q --sae_b /root/data/sae/sae_checkpoint/2eizws4q --hook_point blocks.0.hook_mlp_out --save_html_path /root/data/sae/vis_html/vis.html 
save_html_path 每次定义一个新的文件名称，不让会覆盖旧的
保存文件命名是XuanYuan_6B_Chat_a_2eizws4q_b_2eizws4q_blocks.0.hook_mlp_out
# 评估文件，code/evals
SAElens源码中的文件，稍微修改了加载模型部分，以适用金融LLM和多GPU。
# 模型引导
sae_lens
pip install git+https://github.com/TKOHP/TransformerLens_steering.git
# 模型测评
后台运行测评程序
nohup bash chatglm.sh > chatglm.log 2>&1 &