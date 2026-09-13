# OVS44Reason 三模型使用说明

本文档说明如何在两张 24 GB 显卡上使用以下三个项目处理 OVS44Reason：

- [DGSeg](https://github.com/iguodongsun/DGSeg)
- [SegEarth-R1](https://github.com/iguodongsun/SegEarth-R1)
- [Think2Seg-RS](https://github.com/iguodongsun/Think2Seg-RS)

三个项目均只读取 `rgb`、`D2mask` 和 `text`。当前阶段不使用 `sar`。

## 1. 数据目录

数据集应采用以下结构：

~~~text
OVS44Reason/
├── train/
│   ├── rgb/
│   ├── sar/
│   ├── text/
│   └── D2mask/
└── test/
    ├── rgb/
    ├── sar/
    ├── text/
    └── D2mask/
~~~

`rgb` 与 `sar` 文件同名。`text` 与 `D2mask` 文件同名，并在 RGB 文件名后增加类别名。文本 JSON 中的 `questions[i]` 与 `answer[i]` 一一对应。掩码中大于 0 的像素作为目标。

每个项目的 `tools/prepare_ovs44.py` 会检查文件对应关系、图像尺寸、二值掩码和问答配对，并生成训练与测试清单。清单是运行时生成文件，不需要提交到 Git。

## 2. DGSeg

DGSeg 使用 Qwen2.5-VL 产生语义描述和空间框，再由 SAM3 的两个分支及动态门控模块生成最终掩码。

### 安装

~~~bash
git clone https://github.com/iguodongsun/DGSeg.git
cd DGSeg
conda create -n dgseg python=3.10 -y
conda activate dgseg
pip install -r requirements.txt
pip install -e src/open-r1-multimodal
pip install -e sam3
~~~

准备 Qwen2.5-VL-3B-Instruct 权重和 `sam3.pt`。模型权重不会保存在 GitHub 仓库中。

### 两卡小样本运行

~~~bash
cd DGSeg
PYTHON=/path/to/python \
DATA_ROOT=/path/to/OVS44Reason \
MODEL_PATH=/path/to/Qwen2.5-VL-3B-Instruct \
SAM3_CHECKPOINT=/path/to/sam3.pt \
MAX_SAMPLES=2 \
ACCUM_STEPS=8 \
bash run_scripts/run_ovs44_2x4090.sh
~~~

该脚本依次生成训练与测试预测、训练融合门控模块并评测。正式运行时去掉 `MAX_SAMPLES`。默认训练采用每卡 1 个样本和 8 次梯度累计，等效 batch size 为 16。

可选的 GRPO 阶段：

~~~bash
MODEL_PATH=/path/to/Qwen2.5-VL-3B-Instruct \
SAM3_CHECKPOINT=/path/to/sam3.pt \
ACCUM_STEPS=8 \
bash run_scripts/run_grpo_ovs44_2x4090.sh
~~~

## 3. SegEarth-R1

SegEarth-R1 将图像、问题和答案特征直接输入语言引导的分割解码器。OVS44Reason 已接入官方 `train_dataset.py`，数据集名称为 `OVS44Reason`。

### 安装

按照仓库的 `docs/Installation.md` 创建环境。首次使用还需编译 CUDA 算子：

~~~bash
cd segearth_r1/model/mask_decoder/Mask2Former_Simplify/modeling/pixel_decoder/ops
python setup.py build install
~~~

准备 `SegEarth-R1-EarthReason` 权重后，可执行双卡推理：

~~~bash
cd SegEarth-R1
DATA_ROOT=/path/to/OVS44Reason \
MODEL_PATH=/path/to/SegEarth-R1-EarthReason \
MAX_SAMPLES=2 \
bash run_ovs44_smoke_2gpu.sh
~~~

执行一次真实的双卡训练检查：

~~~bash
DATA_ROOT=/path/to/OVS44Reason \
MODEL_PATH=/path/to/SegEarth-R1-EarthReason \
GRAD_ACCUM=2 \
bash run_ovs44_train_smoke_2gpu.sh
~~~

训练检查会更新约 2476 万个分割相关参数。每卡 batch size 为 1，梯度累计 2 次，等效 batch size 为 4。服务器实测峰值显存约 8.3 GB/卡。

## 4. Think2Seg-RS

Think2Seg-RS 使用 Qwen2.5-VL 输出边界框和两个正点，再由 SAM2.1 生成掩码。训练入口使用官方 GRPO 奖励链路，并通过 LoRA 控制显存。

### 安装

~~~bash
git clone https://github.com/iguodongsun/Think2Seg-RS.git
cd Think2Seg-RS
bash setup.sh
git clone https://github.com/facebookresearch/sam2.git /path/to/sam2
pip install -e /path/to/sam2
~~~

准备 `Think2Seg-RS-3B` 权重及 SAM2.1 small 检查点。

双卡推理：

~~~bash
DATA_ROOT=/path/to/OVS44Reason \
MODEL_PATH=/path/to/Think2Seg-RS-3B \
SAM_ROOT=/path/to/sam2 \
MAX_SAMPLES=2 \
bash run_ovs44_smoke_2gpu.sh
~~~

双卡 GRPO 训练检查：

~~~bash
DATA_ROOT=/path/to/OVS44Reason \
MODEL_PATH=/path/to/Think2Seg-RS-3B \
SAM_ROOT=/path/to/sam2 \
GRAD_ACCUM=2 \
bash run_ovs44_train_smoke_2gpu.sh
~~~

当前配置冻结视觉编码器，使用 LoRA rank 8、每卡 batch size 2、每个问题 4 个候选和 2 次梯度累计。服务器实测峰值显存约 9.1 GB/卡。训练结果和 LoRA adapter 保存在 `OUTPUT_DIR`。

## 5. 已验证结果

| 项目 | 双卡推理 | 反向传播 | 参数更新 | 峰值显存/卡 |
|---|---:|---:|---:|---:|
| DGSeg | 通过 | 融合模块通过 | 通过 | 24 GB 配置可运行 |
| SegEarth-R1 | 通过 | 通过 | 通过 | 约 8.3 GB |
| Think2Seg-RS | 通过 | 通过 | LoRA 更新通过 | 约 8.9–9.1 GB |

冒烟测试用于确认环境、数据、双卡通信、损失、梯度和保存流程。正式精度需要完整训练并在全部测试集上评估。
