# make_dataset.py
直接编辑代码中的cfg，文件名自定义，起个好记的就行，会创建一个和文件名相同的json文件，保存在同一个父目录下。
new_cached_activations_path路径不要以"/"结尾就行，因为要拼接.json。
# shuffle_all
对于给定文件夹内的数据，统一读入到内存中进行shuffle
# shuffleTwoPass_hdf5
两次shuffle方法，中间时候hdf5的增量写入进行过渡（依次写入时不需要将被写入的文件读入内存）
