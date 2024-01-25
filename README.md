![](img/logo-long-chatchat-trans-v2.png)

[Langchain-Chatchat + 阿里通义千问Qwen 保姆级教程  次世代知识管理解决方案 - 知乎](https://zhuanlan.zhihu.com/p/651189680)

[Home · chatchat-spaceLangchain-Chatchat Wiki](https://github.com/chatchat-space/Langchain-Chatchat/wiki/)

最轻量部署方案   

```bash
# 安装依赖,最好是python 3.10
conda create -n py310-chat python=3.10

conda install jq
pip install -r requirements_lite.txt
pip install -r requirements_api.txt
pip install -r requirements_webui.txt  

# 启动
python init_database.py --recreate-vs
python startup.py -a --lite
# python startup.py --webui
```

http://127.0.0.1:8501 



```bash
git lfs install
git clone https://huggingface.co/Qwen/Qwen-1_8B-Chat-Int4
```





趋动云安装镜像命令


```bash
RUN pip install torch~=2.1.2
RUN pip install torchvision~=0.16.2
RUN pip install torchaudio~=2.1.2
RUN pip install xformers==0.0.23.post1
RUN pip install transformers==4.36.2
RUN pip install sentence_transformers==2.2.2
RUN pip install langchain==0.0.352
RUN pip install langchain-experimental==0.0.47
RUN pip install pydantic==1.10.13
RUN pip install fschat==0.2.34
RUN pip install openai~=1.6.0
RUN pip install fastapi>=0.105
RUN pip install sse_starlette
RUN pip install nltk>=3.8.1
RUN pip install uvicorn>=0.24.0.post1
RUN pip install starlette~=0.27.0
RUN pip install unstructured[all-docs]==0.11.0
RUN pip install SQLAlchemy==2.0.19
RUN pip install faiss-cpu~=1.7.4
RUN pip install accelerate==0.24.1
RUN pip install spacy~=3.7.2
RUN pip install PyMuPDF~=1.23.8
RUN pip install rapidocr_onnxruntime==1.3.8
RUN pip install requests>=2.31.0
RUN pip install pathlib>=1.0.1
RUN pip install pytest>=7.4.3
RUN pip install numexpr>=2.8.6
RUN pip install strsimpy>=0.2.1
RUN pip install markdownify>=0.11.6
RUN pip install tiktoken~=0.5.2
RUN pip install tqdm>=4.66.1
RUN pip install websockets>=12.0
RUN pip install numpy~=1.24.4
RUN pip install pandas~=2.0.3
RUN pip install einops>=0.7.0
RUN pip install transformers_stream_generator==0.0.4
RUN pip install vllm==0.2.6; sys_platform == "linux"
RUN pip install httpx[brotli,http2,socks]==0.25.2
RUN pip install llama-index
RUN pip install jq>=1.6.0
RUN pip install beautifulsoup4~=4.12.2 # for .mhtml files
RUN pip install pysrt~=1.1.2
RUN pip install zhipuai>=1.0.7, <=2.0.0 # zhipu
RUN pip install dashscope>=1.13.6 # qwen
RUN pip install arxiv>=2.0.0
RUN pip install youtube-search>=2.1.2
RUN pip install duckduckgo-search>=3.9.9
RUN pip install metaphor-python>=0.1.23
RUN pip install streamlit~=1.29.0
RUN pip install streamlit-option-menu>=0.3.6
RUN pip install streamlit-chatbox==1.1.11
RUN pip install streamlit-modal>=0.1.0
RUN pip install streamlit-aggrid>=0.3.4.post3
RUN pip install watchdog>=3.0.0

RUN pip install scipy
RUN pip install peft
RUN pip install deepspeed
RUN pip install auto-gptq
RUN pip install optimum

```








🌍 [READ THIS IN ENGLISH](README_en.md)

📃 **LangChain-Chatchat** (原 Langchain-ChatGLM)

基于 ChatGLM 等大语言模型与 Langchain 等应用框架实现，开源、可离线部署的检索增强生成(RAG)大模型知识库项目。

---

## 目录

* [介绍](README.md#介绍)
* [解决的痛点](README.md#解决的痛点)
* [快速上手](README.md#快速上手)
  * [1. 环境配置](README.md#1-环境配置)
  * [2. 模型下载](README.md#2-模型下载)
  * [3. 初始化知识库和配置文件](README.md#3-初始化知识库和配置文件)
  * [4. 一键启动](README.md#4-一键启动)
  * [5. 启动界面示例](README.md#5-启动界面示例)
* [联系我们](README.md#联系我们)


## 介绍

🤖️ 一种利用 [langchain](https://github.com/hwchase17/langchain) 思想实现的基于本地知识库的问答应用，目标期望建立一套对中文场景与开源模型支持友好、可离线运行的知识库问答解决方案。

💡 受 [GanymedeNil](https://github.com/GanymedeNil) 的项目 [document.ai](https://github.com/GanymedeNil/document.ai) 和 [AlexZhangji](https://github.com/AlexZhangji) 创建的 [ChatGLM-6B Pull Request](https://github.com/THUDM/ChatGLM-6B/pull/216) 启发，建立了全流程可使用开源模型实现的本地知识库问答应用。本项目的最新版本中通过使用 [FastChat](https://github.com/lm-sys/FastChat) 接入 Vicuna, Alpaca, LLaMA, Koala, RWKV 等模型，依托于 [langchain](https://github.com/langchain-ai/langchain) 框架支持通过基于 [FastAPI](https://github.com/tiangolo/fastapi) 提供的 API 调用服务，或使用基于 [Streamlit](https://github.com/streamlit/streamlit) 的 WebUI 进行操作。

✅ 依托于本项目支持的开源 LLM 与 Embedding 模型，本项目可实现全部使用**开源**模型**离线私有部署**。与此同时，本项目也支持 OpenAI GPT API 的调用，并将在后续持续扩充对各类模型及模型 API 的接入。

⛓️ 本项目实现原理如下图所示，过程包括加载文件 -> 读取文本 -> 文本分割 -> 文本向量化 -> 问句向量化 -> 在文本向量中匹配出与问句向量最相似的 `top k`个 -> 匹配出的文本作为上下文和问题一起添加到 `prompt`中 -> 提交给 `LLM`生成回答。

📺 [原理介绍视频](https://www.bilibili.com/video/BV13M4y1e7cN/?share_source=copy_web&vd_source=e6c5aafe684f30fbe41925d61ca6d514)

![实现原理图](img/langchain+chatglm.png)

从文档处理角度来看，实现流程如下：

![实现原理图2](img/langchain+chatglm2.png)

🚩 本项目未涉及微调、训练过程，但可利用微调或训练对本项目效果进行优化。

🌐 [AutoDL 镜像](https://www.codewithgpu.com/i/chatchat-space/Langchain-Chatchat/Langchain-Chatchat) 中 `v11` 版本所使用代码已更新至本项目 `v0.2.7` 版本。

🐳 [Docker 镜像](registry.cn-beijing.aliyuncs.com/chatchat/chatchat:0.2.6) 已经更新到 ```0.2.7``` 版本。

🌲 一行命令运行 Docker ：

```shell
docker run -d --gpus all -p 80:8501 registry.cn-beijing.aliyuncs.com/chatchat/chatchat:0.2.7
```

🧩 本项目有一个非常完整的[Wiki](https://github.com/chatchat-space/Langchain-Chatchat/wiki/) ， README只是一个简单的介绍，__仅仅是入门教程，能够基础运行__。 如果你想要更深入的了解本项目，或者想对本项目做出贡献。请移步 [Wiki](https://github.com/chatchat-space/Langchain-Chatchat/wiki/)  界面

## 解决的痛点

该项目是一个可以实现 __完全本地化__推理的知识库增强方案, 重点解决数据安全保护，私域化部署的企业痛点。
本开源方案采用```Apache License```，可以免费商用，无需付费。

我们支持市面上主流的本地大语言模型和Embedding模型，支持开源的本地向量数据库。
支持列表详见[Wiki](https://github.com/chatchat-space/Langchain-Chatchat/wiki/)


## 快速上手

### 1. 环境配置

+ 首先，确保你的机器安装了 Python 3.8 - 3.10
```
$ python --version
Python 3.10.12
```
接着，创建一个虚拟环境，并在虚拟环境内安装项目的依赖
```shell

# 拉取仓库
$ git clone https://github.com/chatchat-space/Langchain-Chatchat.git

# 进入目录
$ cd Langchain-Chatchat

# 安装全部依赖
$ pip install -r requirements.txt 
$ pip install -r requirements_api.txt
$ pip install -r requirements_webui.txt  

# 默认依赖包括基本运行环境（FAISS向量库）。如果要使用 milvus/pg_vector 等向量库，请将 requirements.txt 中相应依赖取消注释再安装。
```
### 2， 模型下载

如需在本地或离线环境下运行本项目，需要首先将项目所需的模型下载至本地，通常开源 LLM 与 Embedding 模型可以从 [HuggingFace](https://huggingface.co/models) 下载。

以本项目中默认使用的 LLM 模型 [THUDM/ChatGLM3-6B](https://huggingface.co/THUDM/chatglm3-6b) 与 Embedding 模型 [BAAI/bge-large-zh](https://huggingface.co/BAAI/bge-large-zh) 为例：

下载模型需要先[安装 Git LFS](https://docs.github.com/zh/repositories/working-with-files/managing-large-files/installing-git-large-file-storage)，然后运行

```Shell
$ git lfs install
$ git clone https://huggingface.co/THUDM/chatglm3-6b
$ git clone https://huggingface.co/BAAI/bge-large-zh
```
### 3. 初始化知识库和配置文件

按照下列方式初始化自己的知识库和简单的复制配置文件
```shell
$ python copy_config_example.py
$ python init_database.py --recreate-vs
 ```
### 4. 一键启动

按照以下命令启动项目
```shell
$ python startup.py -a
```
### 5. 启动界面示例

如果正常启动，你将能看到以下界面

1. FastAPI Docs 界面

![](img/fastapi_docs_026.png)

2. Web UI 启动界面示例：

- Web UI 对话界面：

![img](img/LLM_success.png)

- Web UI 知识库管理页面：

![](img/init_knowledge_base.jpg)


### 注意

以上方式只是为了快速上手，如果需要更多的功能和自定义启动方式 ，请参考[Wiki](https://github.com/chatchat-space/Langchain-Chatchat/wiki/)


---
## 项目里程碑


---
## 联系我们
### Telegram
[![Telegram](https://img.shields.io/badge/Telegram-2CA5E0?style=for-the-badge&logo=telegram&logoColor=white "langchain-chatglm")](https://t.me/+RjliQ3jnJ1YyN2E9)

### 项目交流群
<img src="img/qr_code_83.jpg" alt="二维码" width="300" />

🎉 Langchain-Chatchat 项目微信交流群，如果你也对本项目感兴趣，欢迎加入群聊参与讨论交流。

### 公众号

<img src="img/official_wechat_mp_account.png" alt="二维码" width="300" />

🎉 Langchain-Chatchat 项目官方公众号，欢迎扫码关注。
