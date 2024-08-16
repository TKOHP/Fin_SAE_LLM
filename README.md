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

# 主要修改过的源码
## 模型测评
https://github.com/TKOHP/XuanYuan.git
后台运行测评程序
nohup bash chatglm.sh > chatglm.log 2>&1 &
## SAElens
https://github.com/TKOHP/SAELens.git
## 可视化
https://github.com/TKOHP/sae_vis
## Transformer_lens
https://github.com/TKOHP/TransformerLens.git