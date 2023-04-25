## 1.环境准备

```sh
# 1.创建并激活conda独立环境
conda create -n vgn python=3.6
conda activate vgn
# 2.根据电脑cuda环境自行安装合适的pytorch, 参考(cuda11.1+pytorch-1.8.0)
# pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
# 3.安装其他依赖包
pip install -r requirements.txt
```


## 2.基于gt建图

1. 参考现有目录结构，将数据放入到指定的文件目录；
2. 编辑 train.txt 或 test.txt（指定用于建图的文件列表）；
3. 运行建图程序（按照需要设置合适的命令行参数）；
4. 结果文件将自动保存在相应的 ``graph``目录下。

例如：
```sh
python -m make_graph.make_graph --dataset Artery --win_size 8 --source_type gt --multiprocess 16

```


## 3.CNN训练

1. 参考现有目录结构，将数据放入到指定的文件目录；
2. 运行CNN训练程序（按照需要设置合适的命令行参数）；
3. 相关训练结果自动保存在``log/${dataset_name}/CNN``目录下。

例如：
```sh
python train_CNN.py --dataset Artery
```


## 4.基于result建图

1. 参考现有目录结构，将数据放入到指定的文件目录；
2. 运行test_CNN.py（按照需要设置合适的参数）, 生成``*_prob.png``；
3. 运行建图程序（按照需要设置合适的命令行参数），生成``*.graph_res``；
4. 结果文件将自动保存在相应的 ``datasets/{dataset_name}/graph``目录下。


## 5.VGN训练

1. 参考现有目录结构，将数据放入到指定的文件目录；
2. 运行VGN训练程序（按照需要设置合适的命令行参数）；
3. 相关训练结果自动保存在``log/${dataset_name}/VGN``目录下。

例如：
```sh
# 加载CNN训练权重
python train_CNN.py --dataset Artery --pretrained_model log/${dataset_name}/CNN/weights/${weights_name}.pth

# or
# 加载imagenet权重
python train_CNN.py --dataset Artery --pretrained_model pretrained_model/VGG_imagenet.npy

# or
# 重新训练
python train_CNN.py --dataset Artery
```

## 6.VGN 测试

运行test_CNN.py（按照需要设置合适的参数）, 生成``*_prob.png``；


## 7. 训练和测试结果

[cnn](./doc/cnn.md)

[vgn](./doc/vgn.md)