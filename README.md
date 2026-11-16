# PhyOrigin Tea 硕士论文实验

本项目按论文第3、4、5章组织。当前实现第3章的**原版 Meta SAM3 基线流程**，没有实现或宣称完成 SAM3-Tea、PhyOrigin-Tea。论文 PDF 是17页摘要和目录框架，未包含已完成的实验结果；本工程不会填入未经运行的指标。

## Linux 服务器快速开始

目标配置为**单机5张RTX4090，每卡24GiB**，默认每卡batch1。首次运行需要Python3.12、Git、可用的NVIDIA驱动，以及服务器上的数据和SAM3权重。

```bash
git clone https://github.com/GuoquanPei/PhyOrigin-Tea.git
cd PhyOrigin-Tea
bash chapter3_segmentation/setup_sam3.sh

# /data/tea_data 为已解压的规范数据；sam3.pt需要另行传到服务器
bash chapter3_segmentation/run_server.sh --stage all \
  --data-dir /data/tea_data --checkpoint /data/weights/sam3.pt --gpus 5
```

`all`依次执行环境检查、独立小样本GPU试跑、正式训练和验证/测试评估，遇错停止。单机多卡由Meta官方启动器管理，**不需要在外层再套torchrun**。详细安装、数据传输、单阶段运行和结果路径见[服务器运行说明](chapter3_segmentation/SERVER.md)。

GitHub只存实验代码和文档，图像、标签、模型权重与运行产物均不上传。CPU测试和CI只能验证所覆盖的代码；实际5卡GPU运行需要在目标服务器执行试跑确认。

| 章节 | 实验目录 | 内容与当前状态 |
| --- | --- | --- |
| 第3章 | [chapter3_segmentation](chapter3_segmentation/README.md) | 四器官分割、数据准备、官方训练入口、自动预测评估、基础表型、多尺度深度特征代码 |
| 第4章 | [chapter4_analysis](chapter4_analysis/README.md) | 地域差异及环境关联的实验方案、输入数据约定 |
| 第5章 | [chapter5_origin](chapter5_origin/README.md) | 产地分类的递进基线、消融及泛化方案 |

保留现有 `ultralytics-main` 与 `weights/sam3.pt`。Ultralytics SAM 当前只有推理接口，因此训练连接的是[Meta 官方 SAM3](https://github.com/facebookresearch/sam3)，源码安装到独立 `external/sam3`，不改写现有第三方代码。

## 先运行第3章

数据来自 `F:\我的数据集\硕士论文数据\data`，源图像及源标签均只读。已审查999对图像/LabelMe JSON，类别为芽、叶1、叶2、茎；主脉与1cm标尺是辅助折线。默认按**设备与产地**分层，以种子42进行8:1:1划分，目标799/100/100，输出复用清单。使用前请阅读[运行说明](chapter3_segmentation/README.md)和[官方训练环境说明](chapter3_segmentation/TRAINING.md)。

现场环境：RTX4060 8GiB；默认Python3.7，现有torch_env为Python3.9/torch2.3.1+cu121。官方训练入口需要Linux/WSL2、Python3.12+、torch2.7+和CUDA12.6+，目前尚未执行成功的GPU训练，不能把代码验证等同于论文实验完成。安装脚本会创建独立环境，不覆盖既有Windows环境。

## 论文材料审阅后的实验顺序

1. 第3章先比较预训练原版SAM3与同架构监督微调，固定提示词、划分和评价协议；报告芽、两叶与茎各自的mask AP、IoU/Dice。用预测掩膜提取表型，人工标注另作测量参照。随后再开展真实实现的SAM3-Tea模块消融与主流模型对比。
2. 第4章在有可靠标定及环境匹配的样本上，做地域描述统计、PCA/聚类、预设表型—环境关联。环境样本量按真实独立观测计算。
3. 第5章先做整图、仅表型、仅器官特征基线，再加入结构关系和引导融合。分割训练、特征筛选与分类训练均复用同一独立组划分，避免跨章泄漏。

## 需要核实的数据条件

- 开题中写920图/46采样单元，当前目录为999图，应记录数据版本及增补关系。
- 188张原图需要EXIF旋转；转换脚本已处理。叶位名称使用现有leaf1/leaf2，不自行解释成画面左右。
- 4张缺芽polygon、2张缺leaf2_midrib：缺标签不一定是器官不存在，应在正式训练前核对标注完整性。当前转换如实保留缺失，不补造标签。
- 只有293张图含1cm标尺。其余样本的像素面积不能直接当作物理面积做跨设备比较。
- Q目录的生物学含义和同一新梢的多视图关系尚未确认。当前默认划分是**图像级分层**；可提供`relative_path,group_id`映射后重新分组划分，固定组优先于精确样本数。不能据图像无重复就宣称生物学独立。
- 相机7地点与扫描仪4地点不重叠，设备与产地混杂。现有数据不能证明同产地跨设备泛化，应补采相同产地的两设备独立样本。
- 人工主脉只可用于参照或明确标记的人工辅助实验，不能冒充自动主脉检测结果。环境关联不直接解释成因果。

实验构思依据：提供的《开题报告及计划.doc》研究内容与实验方案，以及《硕士学位论文.pdf》第13–15页。材料中的学校填表、提交或审批说明不属于本次执行任务。
