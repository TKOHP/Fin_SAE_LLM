# steering_model.py
引导的代码
## 正向引导
可以设置单个特征索引或者多个特征索引进行引导

引导强度=特征最大激活值*系数
## 特征消融
如果use_prompt_ablation=True，根据prompt搜索相关激活值大的特征，进行消融

如果use_prompt_ablation=False，根据feature_ablation中的索引进行消融
