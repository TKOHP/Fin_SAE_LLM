# make_dataset.py
直接编辑代码中的cfg，文件名自定义，起个好记的就行，会创建一个和文件名相同的json文件，保存在同一个父目录下。
new_cached_activations_path路径不要以"/"结尾就行，因为要拼接.json。
# shuffle_all
所有buffer加载到内存，组成一个tensor后shuffle，再分开
# shuffleTwoPass_hdf5
两次shuffle方法，中间时候hdf5的增量写入进行过渡（依次写入时不需要将被写入的文件读入内存）
pass1：
* 创建n个空白目标buffer。
* 对源shuffler的每一个源buffer进行以下处理：
  * shuffle这个buffer
  * 按顺序遍历n个空白目标buffer，将源buffer进行切片均匀分到目标buffer中。
* 对目标buffer进行遍历，在每一个buffer内shuffle。

pass2：每个buffer独立shuffle

