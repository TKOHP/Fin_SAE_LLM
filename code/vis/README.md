# 可视化HTML生成，code/vis_maker
## 虚拟环境配置
在环境安装的基础上，安装包：

pip install git+https://github.com/TKOHP/sae_vis.git

然后卸载transformer_lens。pip uninstall transformer_lens

pip install git+https://github.com/TKOHP/TransformerLens_sae_vis.git 这个包其实是对TransformerLens的多GPU进行了一个指定，并对齐了一些变量
note:transformers版本为4.38.1.

# vis_maker
## 以特征为中心
包括生成全部特征，生成单个特征的（将is_single设为True）