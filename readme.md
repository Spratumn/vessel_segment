## 环境准备

```sh
conda create -n vgn python=3.6
conda activate vgn
pip install -r requirements.txt
```


## 基于gt建图

1. 参考现有目录结构，将数据放入到指定的文件目录；
2. 编辑 train.txt 或 test.txt（指定用于建图的文件列表）；
3. 运行建图程序（按照需要设置合适的命令行参数）；
4. 结果文件将自动保存在相应的 ``graph``目录下。

例如：
```sh
python -m make_graph.make_graph --dataset CHASE_DB1 --win_size 8 --source_type gt --use_multiprocessing

# or

python -m make_graph.make_graph --dataset Artery --win_size 8 --source_type gt --use_multiprocessing
```