# 第3章 器官分割与视觉表型

当前代码覆盖原版SAM3基线的数据、训练、评估与特征导出。GPU训练及真实预测尚待在支持的环境中执行；没有生成模型性能数值。

服务器部署请先看[SERVER.md](SERVER.md)：默认单机5张RTX4090（每卡24GiB），包含独立小样本GPU试跑和完整训练入口。GitHub仅发布代码，不包含数据或权重。

## 文件与数据

| 文件 | 功能 |
| --- | --- |
| prepare_data.py | LabelMe转COCO、EXIF校正、分层划分、几何及来源留存 |
| train.py | 生成完整Hydra配置、环境预检、启动官方训练器 |
| evaluate.py | 自动文字提示预测、COCO mask/box AP、各类别累计IoU/Dice |
| phenotypes.py | 预测掩膜面积、像素栅格周长、PCA轴长宽、质心及可选人工主脉参照 |
| extract_features.py | 冻结ImageNet ResNet50的掩膜引导多尺度器官特征 |
| run_server.sh | Linux预检、独立GPU试跑、正式训练、验证/测试的统一入口 |
| make_smoke_data.py | 单独构建10训练/5验证图像的小样本试跑集 |
| export_data.py | 将规范数据和固定划分打包为可校验的Linux传输归档 |

`data/manifest.csv`是各章节共用的划分及来源索引；`annotations/{train,val,test}.json`是COCO实例标签；`metadata/*.json`保存人工主脉、标尺和原标签信息；`split_stats.json`记录分层及类别分布。原图只读，规范图像写入`data/images`。同名源文件通过完整相对路径生成稳定ID，避免覆盖。

## 数据准备

在项目根目录使用Python3.10+，安装`requirements-data.txt`。本次实际转换使用Codex附带Python3.12和Pillow；现有torch_env也可执行不依赖SAM3的数据与评估工具。

```powershell
python -m pip install -r chapter3_segmentation/requirements-data.txt
python chapter3_segmentation/prepare_data.py --data-root "F:\我的数据集\硕士论文数据\data" --output chapter3_segmentation/data --seed 42
```

默认每个`设备×产地`层按8:1:1分配并平衡整数余数，总数为799/100/100。仅四种polygon进入实例训练；linestrip不变成伪造mask。相同解码像素图像和明确指定的生物样本组不会跨集合。

有真实新梢/采样单元身份时，创建UTF-8 CSV：

```csv
relative_path,group_id
相机/产地目录/RGB/Q1/IMG_0001.JPG,shoot_001
相机/产地目录/RGB/Q1/IMG_0002.JPG,shoot_001
```

使用真实存在的相对路径并加`--groups-csv groups.csv`重新转换。不要凭Q编号猜测样本身份。组无法拆开时，总数可能偏离799/100/100，统计文件报告实际比例。更换划分后必须重训相关模型，不混用旧权重、旧特征和新测试集。

## 训练与预测

先按[TRAINING.md](TRAINING.md)准备独立Linux/WSL2环境。CLI的`--dry-run`仅导出配置；`--check`不满足要求时返回2。正式训练默认50轮、5GPU、每卡batch1、1008px、种子42。`--num-gpus 1`可以显式改为单卡；实际显存容量需要服务器试跑确认。

```bash
python chapter3_segmentation/train.py --sam3-root external/sam3 --check
python chapter3_segmentation/train.py --sam3-root external/sam3 --output-dir chapter3_segmentation/runs/sam3

# 先在验证集评估；可以另设输出目录，用原始weights/sam3.pt作未微调参照
python chapter3_segmentation/evaluate.py --sam3-root external/sam3 --checkpoint chapter3_segmentation/runs/sam3/checkpoints/checkpoint.pt --split val --output-dir chapter3_segmentation/runs/val

# 所有提示、阈值和训练方案冻结后，才运行最终测试
python chapter3_segmentation/evaluate.py --sam3-root external/sam3 --checkpoint chapter3_segmentation/runs/sam3/checkpoints/checkpoint.pt --split test --output-dir chapter3_segmentation/runs/test
```

提示词与训练类别保持一致：`tea bud`、`first tea leaf`、`second tea leaf`、`tea stem`。评估输入不使用人工真值框或点；每张图运行四类自动文字提示。`--prediction-threshold`默认0.05，用于保留AP排序候选；`--metric-threshold`默认0.5，用于累计类别IoU/Dice，应在验证集确定。COCO mask AP与bbox AP分别评估；IoU/Dice是**逐类跨图累计的语义并集指标**，不是实例平均Dice。

输出`predictions.json`、`metrics.json`、`evaluation.json`。已有COCO预测时，可以用`--predictions path.json`替代`--checkpoint`，在CPU重算指标。多个器官/图像重复测量不是独立统计重复；显著性与区间分析需在确认样本组后开展。

## 表型与深度特征

```bash
python chapter3_segmentation/phenotypes.py --split test --predictions chapter3_segmentation/runs/test/predictions.json --output chapter3_segmentation/runs/test/phenotypes.csv

# 明确使用人工标尺，且把人工主脉列为参考量；不是全自动主脉提取
python chapter3_segmentation/phenotypes.py --split test --predictions chapter3_segmentation/runs/test/predictions.json --manual-geometry --output chapter3_segmentation/runs/test/phenotypes_assisted.csv

# 人工轮廓参考测量，用于对照预测表型，不作为正式自动结果
python chapter3_segmentation/phenotypes.py --split test --reference-labels --manual-geometry --output chapter3_segmentation/runs/reference/test.csv

python chapter3_segmentation/extract_features.py --split test --predictions chapter3_segmentation/runs/test/predictions.json --output-dir chapter3_segmentation/runs/test/features
```

面积基于mask像素；`perimeter_grid_px`是栅格边长和，有方向依赖，不是平滑轮廓的欧氏周长；轴长宽是像素中心PCA方向上的投影范围，不能称为主叶脉弧长。主脉人工折线输出弧长、端点弦长与弧弦比。使用唯一有效1cm线时换算cm/cm²，否则物理量留空。当前未实现自动主脉检测、叶齿量化和人工重复测量的一致性统计。

深度特征是冻结ResNet50基线，不是SAM3-Tea创新模块。每个预测器官按1.0/1.5倍尺度裁剪、掩膜外填中性灰、方形补边、缩放224；保存`features.npz`（N×2×2048）、`index.csv`和配置说明。未提供`--weights`时，首次执行需下载torchvision ImageNet权重；也可提供相同ResNet50结构的本地state_dict。缺失/低置信度器官没有记录，后续分类需保留缺失标志并在训练集拟合填补方案。

## 论文实验矩阵

| 实验 | 当前支持 | 待完成条件 |
| --- | --- | --- |
| 预训练SAM3 vs 原版微调 | 同架构训练/评估入口 | 支持的GPU环境实际运行 |
| 四器官分项性能及难例 | AP、IoU/Dice、逐实例输出 | 真实预测；补充边界和耗时分析 |
| 表型提取与人工参照 | 掩膜几何、手工标尺、主脉参照 | 独立复核、重复测量、MAE/RMSE与一致性图 |
| 掩膜引导多尺度特征 | 冻结ResNet50导出 | 实际权重、预测掩膜与下游验证 |
| SAM3-Tea消融、响应与显著性 | 仅保留后续实验设计 | 真实改进模块、多种子与独立组协议 |
| YOLO-seg/Mask R-CNN等比较 | 仅保留后续实验设计 | 同划分、同调参预算、实际训练 |

源码与环境约束见[官方训练文档](https://github.com/facebookresearch/sam3/blob/eef6839bcfc3458dd52ea80535dc6bd62aea86b1/README_TRAIN.md)。
