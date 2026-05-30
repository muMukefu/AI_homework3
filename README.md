---
domain: nlp
tags:
  - llm
  - chat
  - comparison
models:
  - qwen/Qwen-7B-Chat
  - deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
  - ZhipuAI/chatglm3-6b
license: Apache License 2.0
---

# 人工智能导论第三次作业

## 简介

1. 登录并使用魔搭平台，注册时需要关联阿里云账号来获得免费的CPU云计算资源；
2. 通过Jupyter Notebook或者相应环境镜像进入相应的项目部署环境，根据相应模型的部署文档完成模型的部署；
3. 针对3个不同的模型（这里采用了通义千问、DeepSeek和智谱三个大模型）进行问答测试，并开展不同模型之间的横向对比；

## 测试模型介绍

| 模型 | 模型介绍页 |
|------|------------|
| 通义千问 Qwen-7B-Chat | https://modelscope.cn/models/qwen/Qwen-7B-Chat/summary |
| DeepSeek-R1-Distill-Qwen-1.5B | https://modelscope.cn/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B/summary |
| 智谱 ChatGLM3-6B | https://modelscope.cn/models/ZhipuAI/chatglm3-6b/summary |

## 搭建步骤

### 1. 登录魔搭平台
关联阿里云账号获得免费的CPU云计算资源，启动CPU，启动Notebook准备；

### 2. 环境搭建（以conda环境为例）

#### 2.1 手动下载conda环境：
```bash
cd /opt/conda/envs
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda
echo 'export PATH="/opt/conda/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
conda --version
```

#### 2.2 激活conda环境：
```bash
conda create -n qwen_env python=3.10 -y
source /opt/conda/etc/profile.d/conda.sh
conda activate qwen_env
```

### 3. 基础依赖下载

#### 3.1 基础环境：
```bash
pip install \
    torch==2.3.0 \
    torchvision==0.18.0 \
    --index-url https://download.pytorch.org/whl/cpu
```

#### 3.2 基础依赖：
```bash
pip install -U pip setuptools wheel

pip install \
    "intel-extension-for-transformers==1.4.2" \
    "neural-compressor==2.5" \
    "transformers==4.33.3" \
    "modelscope==1.9.5" \
    "pydantic==1.10.13" \
    "sentencepiece" \
    "tiktoken" \
    "einops" \
    "transformers_stream_generator" \
    "uvicorn" \
    "fastapi" \
    "yacs" \
    "setuptools_scm"

pip install fschat --use-pep517
```

### 4. 下载大模型到本地（每次建议选一个）
```bash
cd /mnt/data
git clone https://www.modelscope.cn/ZhipuAI/chatglm3-6b.git
git clone https://www.modelscope.cn/qwen/Qwen-7B-Chat.git
git clone https://www.modelscope.cn/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B.git
```

### 5. 编写脚本进行测试

## 测试问题

1. 请说出以下两句话区别在哪里？ 1、冬天：能穿多少穿多少 2、夏天：能穿多少穿多少
2. 单身狗产生的原因有两个，一是谁都看不上，二是谁都看不上。这两句话区别在哪里？
3. 明明明明明白白白喜欢他，可她就是不说。这句话里，明明和白白谁喜欢谁？
4. 他知道我知道你知道他不知道吗？这句话里，到底谁不知道？
5. 领导：你这是什么意思？
   小明：没什么意思。意思意思。
   领导：你这就不够意思了。
   小明：小意思，小意思。
   领导：你这人真有意思。
   小明：其实也没有别的意思。
   领导：那我就不好意思了。
   小明：是我不好意思。
   请问：以上所有"意思"分别是什么意思？

## 横向对比分析

> **测试说明**：本次测试输出被限制为100 tokens。根据实际测试体验，各模型表现如下：

### 1. 响应速度对比（实际测试感受）

| 排名 | 模型 | 响应速度 | 说明 |
|------|-----|---------|------|
| **1** | DeepSeek-1.5B | **最快** | 响应速度最快，但受token限制影响较大 |
| **2** | Qwen-7B-Chat | 较快 | 回答风格简洁，输出较快 |
| **3** | ChatGLM3-6B | 较慢 | 回答较为详细，输出时间稍长 |

### 2. 回答质量对比（实际测试感受）

| 排名 | 模型 | 回答质量 | 说明 |
|------|-----|---------|------|
| **1** | ChatGLM3-6B | **最佳** | 大多数问题回答更详细准确 |
| **2** | Qwen-7B-Chat | 良好 | 回答风格简洁，准确度良好 |
| **3** | DeepSeek-1.5B | 受限制 | 受token限制影响，部分题目输出不完整 |

### 3. 各模型特点分析

**DeepSeek-R1-Distill-Qwen-1.5B：**
-  **响应速度最快**：在三个模型中响应速度最快
-  **受token限制影响大**：部分题目可以输出完整，部分题目被截断
-  **潜力待发掘**：在更大token限制下可能有更好表现

**智谱 ChatGLM3-6B：**
-  **回答质量最高**：大多数问题比千问回答更详细准确
-  **分析深入**：对问题的分析较为全面
-  **输出较慢**：因回答详细，输出时间比千问稍长

**通义千问 Qwen-7B-Chat：**
-  **风格简洁**：回答风格简洁明了
-  **输出较快**：因回答简洁，输出速度比智谱快一点
-  **详细度一般**：相比智谱，回答不够详细

### 4. Token限制影响分析

| 模型 | 受100 token限制影响 | 具体表现 |
|------|-------------------|---------|
| DeepSeek-1.5B | **较大** | 部分题目可输出完整，部分被截断 |
| ChatGLM3-6B | 较小 | 大多数问题能在限制内完成 |
| Qwen-7B-Chat | 较小 | 回答简洁，基本不受影响 |

### 5. 应用场景建议

| 场景 | 推荐模型 | 理由 |
|------|---------|------|
| **追求响应速度** | DeepSeek-1.5B | 响应最快，但需注意token限制 |
| **追求回答质量** | ChatGLM3-6B | 回答最详细准确 |
| **追求简洁高效** | Qwen-7B-Chat | 回答简洁，速度较快，稳定性好 |
| **复杂分析任务** | ChatGLM3-6B | 分析深入，回答全面 |

### 6. 本项目的可改进之处

1. **增加token限制**：建议将输出token限制增加到512以上，以充分评估DeepSeek模型的完整性能
2. **多维度测试**：建议增加更多类型的测试问题，全面评估模型能力
3. **量化指标**：可增加响应时间、token使用量等量化指标的记录

## 模型下载及测试结果截图（详见魔搭平台"空间文件"栏）
