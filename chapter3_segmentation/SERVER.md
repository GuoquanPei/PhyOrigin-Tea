# Linux 服务器运行：5 张 RTX 4090

代码使用 Meta 原版 SAM3。默认启动 5 个 GPU 进程，每张卡 batch size 为 1，
全局 batch size 为 5 个“图像/类别文本查询”数据点，不能等同为 5 个独立生物样本；
同一图像的四个类别查询会形成四个数据点。每张卡仍分别受 24 GB 显存限制。默认分辨率为官方的
1008px。本地完成了数据处理与软件测试，尚未在这台五卡服务器上完成实际训练。

## 1. 拉取代码并安装独立环境

```bash
git clone https://github.com/GuoquanPei/PhyOrigin-Tea.git
cd PhyOrigin-Tea
nvidia-smi

# 此依赖组合使用 Python 3.12；需要系统已安装对应的 venv 支持。
PYTHON_BIN=python3.12 bash chapter3_segmentation/setup_sam3.sh
```

安装脚本固定官方 SAM3 源码版本、PyTorch 2.7.1、torchvision 0.22.1 和 CUDA 12.6
轮子；服务器驱动需要支持该 CUDA 运行时。服务器不需要安装 Windows 环境。
脚本默认创建 `.venv-sam3`，不修改系统 Python 或已有 Conda 环境；重复执行会复用
属于当前仓库的环境。遇到同名的无标记环境会退出，可改用新目录：

```bash
PYTHON_BIN=/opt/python3.12/bin/python3.12 \
  bash chapter3_segmentation/setup_sam3.sh /data/software/sam3 /data/venvs/phytea-sam3
```

后续命令同时传入 `--sam3-root /data/software/sam3`
和 `--python /data/venvs/phytea-sam3/bin/python`。默认位置不需要这两个参数。
未设置 `PYTHON_BIN` 时先查找 `python3.12`，再检查版本为 3.12 的 `python3`。
本依赖组合固定 numpy 1.26.4，因此安装脚本明确拒绝 Python 3.13 及以上版本。

## 2. 传输数据和原始权重

数据、预训练权重和训练结果不进入 Git。推荐传输已经生成的规范数据包，
这样保留既定的 **799/100/100** 划分和 EXIF 校正结果。

在当前 Windows 机器的仓库根目录执行，`python` 需已安装数据处理依赖：

```powershell
python chapter3_segmentation/export_data.py --data-dir chapter3_segmentation/data --output tea_data.tar.gz
scp tea_data.tar.gz tea_data.tar.gz.sha256 user@server:/data/tea/
scp weights/sam3.pt user@server:/data/tea/sam3.pt
```

如果本机已生成 `chapter3_segmentation/runs/transfer/tea_data.tar.gz`，可直接传输
该文件及同目录下的 `.sha256` 文件，无需重新导出。先在服务器创建接收目录。

在服务器校验后解压到独立的数据位置；不要解压到原始数据目录。归档含顶层
`tea_data/`，其中包含 `images/`、`annotations/`、`metadata/`、`manifest.csv`
与 `split_stats.json`：

```bash
(cd /data/tea && sha256sum -c tea_data.tar.gz.sha256)
mkdir -p /data
tar -xzf /data/tea/tea_data.tar.gz -C /data
```

manifest 中原始 Windows 路径用于保留来源；训练读取 `images/` 和 COCO 注释，
不依赖服务器存在 `F:` 盘。原始权重应是官方 SAM3 检查点，并已满足其访问许可。

也可以传输原始 LabelMe 数据，保留“采集设备/采集点/RGB/Q编号/图片及同名JSON”
目录结构，再在服务器准备：

```bash
# 在保存原始数据的机器上，末尾斜线表示复制 data 内的内容。
rsync -av --progress /path/to/raw/data/ user@server:/data/tea/raw/

# 在服务器仓库根目录执行。
bash chapter3_segmentation/run_server.sh --stage prepare \
  --data-root /data/tea/raw --data-dir /data/tea_data --seed 42
```

默认以设备和采集点分层，固定 seed=42；像素完全重复的图像保留在同一划分。
如果同一枝条有多张图，需要在准备时提供 `--groups-csv /data/tea/groups.csv`，
列为 `relative_path,group_id`。未提供身份映射时，只能声称图像级划分。
已有完整准备数据会复用；改变分组时需明确增加 `--force-prepare --data-root ...`。

## 3. 预检、独立冒烟测试、正式训练

```bash
# 根据需要指定实际分配给当前作业的五张卡。
export CUDA_VISIBLE_DEVICES=0,1,2,3,4

bash chapter3_segmentation/run_server.sh --stage check \
  --data-dir /data/tea_data --checkpoint /data/tea/sam3.pt --gpus 5

bash chapter3_segmentation/run_server.sh --stage smoke \
  --data-dir /data/tea_data --checkpoint /data/tea/sam3.pt --gpus 5

bash chapter3_segmentation/run_server.sh --stage train \
  --data-dir /data/tea_data --checkpoint /data/tea/sam3.pt \
  --gpus 5 --batch-size 1 --epochs 50 --output-dir /data/tea/runs/sam3-full
```

冒烟测试使用 10 张训练图、5 张验证图和 1 个 epoch，单独写入
`chapter3_segmentation/runs/smoke/<新目录>/`。正式训练始终使用完整准备数据，
不会恢复冒烟测试的权重或优化器。脚本检查冒烟输出确实含有检查点后才继续。
五卡下预检、完整前向、反向、验证、保存仍需要服务器实际执行才能确认。

也可以一次完成准备数据复核、预检、冒烟测试、正式训练和 val/test 评估：

```bash
bash chapter3_segmentation/run_server.sh --stage all \
  --data-dir /data/tea_data --checkpoint /data/tea/sam3.pt \
  --gpus 5 --batch-size 1 --epochs 50 --output-dir /data/tea/runs/sam3-full
```

脚本遇到失败立即退出。总日志位于 `chapter3_segmentation/runs/server_logs/`；
正式训练的预检、Hydra 配置、TensorBoard 和检查点位于 `--output-dir`。
上游训练器会自动恢复该目录的 `checkpoints/checkpoint.pt`；新实验使用新的输出目录。
如果服务器显存不足，可显式使用 `--freeze-backbone` 冻结原版视觉和语言主干；
该选项不增加或替换模型结构，应作为单独实验记录。

## 4. 评估和后续特征导出

```bash
bash chapter3_segmentation/run_server.sh --stage evaluate \
  --data-dir /data/tea_data --checkpoint /data/tea/sam3.pt \
  --output-dir /data/tea/runs/sam3-full
```

该阶段依次评估 val 和 test，默认读取正式训练的
`checkpoints/checkpoint.pt`；可用 `--eval-checkpoint /path/to/trained.pt` 指定训练权重。
评估为单卡推理，输出分别位于 `evaluation/val/` 和 `evaluation/test/`，
包括预测掩膜 `predictions.json`、指标与元数据。测试集只用于最终报告。

ResNet 多尺度特征不自动运行，以便明确控制其 ImageNet 权重来源。需要时先把
对应 torchvision ResNet50 ImageNet 权重放到服务器，再运行：

```bash
.venv-sam3/bin/python chapter3_segmentation/extract_features.py \
  --data-dir /data/tea_data --split test \
  --predictions /data/tea/runs/sam3-full/evaluation/test/predictions.json \
  --weights /data/tea/resnet50-imagenet.pth \
  --output-dir /data/tea/runs/sam3-full/features/test --device cuda
```

完整训练接口见 [TRAINING.md](TRAINING.md)，数据、评价和表型流程见 [README.md](README.md)。
