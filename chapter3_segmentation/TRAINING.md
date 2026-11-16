# 原版 SAM3 微调

本目录训练的是 Meta 原版 SAM3 图像模型，直接调用官方 `sam3.train.train` 与
`sam3.train.trainer.Trainer`；没有增加 SAM3-Tea、适配器或替换解码器。
Ultralytics 的 SAM 接口只有预测器，不能用 `SAM(...).train(...)` 实现这项训练。

训练配置改编自[官方 RF100-VL 配置](https://github.com/facebookresearch/sam3/blob/eef6839bcfc3458dd52ea80535dc6bd62aea86b1/sam3/train/configs/roboflow_v100/roboflow_v100_full_ft_100_images.yaml)，
固定源码版本 `eef6839bcfc3458dd52ea80535dc6bd62aea86b1`，授权文本见
[SAM3_LICENSE](configs/SAM3_LICENSE)。启动器在输出目录生成完整 Hydra YAML，
不会往上游源码目录复制或修改文件。

## 环境与当前机器

[官方环境要求](https://github.com/facebookresearch/sam3#installation)：Python 3.12+、
PyTorch 2.7+、CUDA 12.6+。本训练入口使用上游 Triton 与 NCCL，要求 Linux 或 WSL2。
本机发现的 `torch_env` 是 Python 3.9.19、torch 2.3.1+cu121，不能直接使用。
RTX 4060 只有 8 GiB；完整 1008px 微调可能显存不足，现阶段没有完成 GPU 训练。

用户服务器配置已补充为5张RTX4090，每卡24GiB。当前默认`--num-gpus 5`，由官方启动器在单机创建5个DDP进程；不使用外层torchrun。预检会检查可见卡数、所选各卡显存和BF16能力。服务器运行入口见[SERVER.md](SERVER.md)。

在安装了 NVIDIA 驱动、WSL2 GPU 支持及 Python 3.12 的 Linux/WSL2 中运行：

```bash
bash chapter3_segmentation/setup_sam3.sh
source .venv-sam3/bin/activate
python chapter3_segmentation/train.py --sam3-root external/sam3 --check
```

脚本使用 torch 2.7.1 / torchvision 0.22.1 的 CUDA 12.6 wheel，避免覆盖原有
Windows 环境。官方 SAM3 检查点已经放在 `weights/sam3.pt` 时，不需要在线下载权重。
自己下载时应先完成[官方访问授权](https://huggingface.co/facebook/sam3)。

此版本SAM3仍导入`pkg_resources`，依赖文件约束`setuptools<82`；
[setuptools82已移除此模块](https://setuptools.pypa.io/en/latest/deprecated/pkg_resources.html)。
预检也会检查该模块，避免缺失时进入训练。

## 数据与命令

数据入口为 `data/images/<sample_id>.jpg` 和 `data/annotations/{train,val,test}.json`。
COCO `file_name` 为图片文件名；类别 ID 为 0–3，名称依次为 `tea bud`、
`first tea leaf`、`second tea leaf`、`tea stem`。类别名称直接作为原版模型的文本提示。
标注需含有效的 COCO 多边形或 RLE 实例分割。训练和每轮验证只读取 train/val；
test 留给最终独立评估。

```bash
# 在现有 Python 环境中检查配置，不加载模型，不要求 GPU 或上游包可导入
python chapter3_segmentation/train.py --sam3-root external/sam3 --dry-run

# 正式微调：50 epochs，5GPU，每卡batch=1，workers=2，seed=42
python chapter3_segmentation/train.py --sam3-root external/sam3

# 可选：冻结原版视觉和语言骨干，训练原版其余模块，降低梯度和优化器内存
python chapter3_segmentation/train.py --sam3-root external/sam3 --freeze-backbone
```

冻结模式仍是原版模型，但与全参数微调是不同实验；应记录该设置。它没有经过本机
8 GiB 显存实测，不能保证运行成功。默认使用官方 1008px 设置，不通过降低分辨率
悄悄改变实验。`--epochs`、`--batch-size`、`--workers`、`--seed`、`--data-dir`、
`--checkpoint`、`--output-dir` 可显式配置。

`--batch-size`是每卡的image-query数据项数；当前每个类别单独形成查询，不能将全局batch5解释为5个独立新梢。增加卡数不会自动放大学习率，梯度累积保持1。

`--dry-run` 输出 `sam3_train.yaml` 和 `preflight.json`，即使环境不满足也只做配置导出；
`--check` 遇到阻碍返回退出码 2；正常训练同样先检查，不会将 dry-run 当作训练成功。
日志和最后权重分别位于 `runs/sam3/logs`、`runs/sam3/checkpoints/checkpoint.pt`；
配置启用了实例 mask、Dice 和语义 mask 损失、训练/验证 RLE 解码以及 COCO segm 验证。
上游最后权重自动恢复训练，重新实验应更换 `--output-dir`。

## 训练后权重加载

官方原始检查点包含 `detector.*` / `tracker.*`，可以传给官方 builder 的
`checkpoint_path`。官方训练器输出 `checkpoint['model']` 则是不带 `detector.` 前缀
的图像模型状态字典。后者应先建立 `load_from_HF=False` 的相同原版架构，再用
`model.load_state_dict(checkpoint['model'], strict=True)`；不能直接交给 builder 的
原始检查点加载器，否则可能没有加载到训练参数。

CPU 可执行的配置检查：

```bash
python -m unittest discover -s chapter3_segmentation/tests -p test_train_config.py -v
```

这些检查验证配置和失败路径；实际前向、反向、保存及权重重载仍须在满足要求的
训练环境中实测。原版训练命令和模式说明见[官方训练文档](https://github.com/facebookresearch/sam3/blob/eef6839bcfc3458dd52ea80535dc6bd62aea86b1/README_TRAIN.md)。
