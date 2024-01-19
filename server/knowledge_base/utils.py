import os
from configs import (
    KB_ROOT_PATH,
    CHUNK_SIZE,
    OVERLAP_SIZE,
    ZH_TITLE_ENHANCE,
    logger,
    log_verbose,
    text_splitter_dict,
    LLM_MODELS,
    TEXT_SPLITTER_NAME,
)
import importlib
from text_splitter import zh_title_enhance as func_zh_title_enhance
import langchain.document_loaders
from langchain.docstore.document import Document
from langchain.text_splitter import TextSplitter
from pathlib import Path
from server.utils import run_in_thread_pool, get_model_worker_config
import json
from typing import List, Union, Dict, Tuple, Generator
import chardet


def validate_kb_name(knowledge_base_id: str) -> bool:
    # 检查是否包含预期外的字符或路径攻击关键字
    if "../" in knowledge_base_id:
        return False
    return True


def get_kb_path(knowledge_base_name: str):
    return os.path.join(KB_ROOT_PATH, knowledge_base_name)


def get_doc_path(knowledge_base_name: str):
    return os.path.join(get_kb_path(knowledge_base_name), "content")


def get_vs_path(knowledge_base_name: str, vector_name: str):
    return os.path.join(get_kb_path(knowledge_base_name), "vector_store", vector_name)


def get_file_path(knowledge_base_name: str, doc_name: str):
    return os.path.join(get_doc_path(knowledge_base_name), doc_name)


# 列出KB_ROOT_PATH下的所有子目录
def list_kbs_from_folder():
    return [
        f
        for f in os.listdir(KB_ROOT_PATH)
        if os.path.isdir(os.path.join(KB_ROOT_PATH, f))
    ]


def list_files_from_folder(kb_name: str):
    doc_path = get_doc_path(kb_name)
    result = []

    def is_skiped_path(path: str):
        tail = os.path.basename(path).lower()
        for x in ["temp", "tmp", ".", "~$"]:
            if tail.startswith(x):
                return True
        return False

    def process_entry(entry):
        if is_skiped_path(entry.path):
            return

        if entry.is_symlink():
            target_path = os.path.realpath(entry.path)
            with os.scandir(target_path) as target_it:
                for target_entry in target_it:
                    process_entry(target_entry)
        elif entry.is_file():
            file_path = Path(
                os.path.relpath(entry.path, doc_path)
            ).as_posix()  # 路径统一为 posix 格式
            result.append(file_path)
        elif entry.is_dir():
            with os.scandir(entry.path) as it:
                for sub_entry in it:
                    process_entry(sub_entry)

    with os.scandir(doc_path) as it:
        for entry in it:
            process_entry(entry)

    return result


LOADER_DICT = {
    "UnstructuredHTMLLoader": [".html"],
    "MHTMLLoader": [".mhtml"],
    "UnstructuredMarkdownLoader": [".md"],
    # "JSONLoader": [".json"],   # PPP ### jq模块安装不了，暂时不支持
    # "JSONLinesLoader": [".jsonl"],
    "CSVLoader": [".csv"],
    # "FilteredCSVLoader": [".csv"], # 需要自己指定，目前还没有支持
    "RapidOCRPDFLoader": [".pdf"],
    # "RapidOCRLoader": [".png", ".jpg", ".jpeg", ".bmp"],   # PPP ### jpg需要GPU来OCR，暂时不支持
    "UnstructuredEmailLoader": [".eml", ".msg"],
    "UnstructuredEPubLoader": [".epub"],
    "UnstructuredExcelLoader": [".xlsx", ".xls", ".xlsd"],
    "NotebookLoader": [".ipynb"],
    "UnstructuredODTLoader": [".odt"],
    "PythonLoader": [".py"],
    "UnstructuredRSTLoader": [".rst"],
    "UnstructuredRTFLoader": [".rtf"],
    "SRTLoader": [".srt"],
    "TomlLoader": [".toml"],
    "UnstructuredTSVLoader": [".tsv"],
    "UnstructuredWordDocumentLoader": [".docx", ".doc"],
    "UnstructuredXMLLoader": [".xml"],
    "UnstructuredPowerPointLoader": [".ppt", ".pptx"],
    "EverNoteLoader": [".enex"],
    "UnstructuredFileLoader": [".txt"],
}
SUPPORTED_EXTS = [ext for sublist in LOADER_DICT.values() for ext in sublist]


# patch json.dumps to disable ensure_ascii
def _new_json_dumps(obj, **kwargs):
    kwargs["ensure_ascii"] = False
    return _origin_json_dumps(obj, **kwargs)


if json.dumps is not _new_json_dumps:
    _origin_json_dumps = json.dumps
    json.dumps = _new_json_dumps


class JSONLinesLoader(langchain.document_loaders.JSONLoader):
    """
    行式 Json 加载器，要求文件扩展名为 .jsonl
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._json_lines = True


langchain.document_loaders.JSONLinesLoader = JSONLinesLoader


def get_LoaderClass(file_extension):
    for LoaderClass, extensions in LOADER_DICT.items():
        if file_extension in extensions:
            return LoaderClass


# 把一些向量化共用逻辑从KnowledgeFile抽取出来，等langchain支持内存文件的时候，可以将非磁盘文件向量化
def get_loader(loader_name: str, file_path: str, loader_kwargs: Dict = None):
    """
    根据loader_name和文件路径或内容返回文档加载器。
    """
    loader_kwargs = loader_kwargs or {}
    try:
        if loader_name in ["RapidOCRPDFLoader", "RapidOCRLoader", "FilteredCSVLoader"]:
            document_loaders_module = importlib.import_module("document_loaders")
        else:
            document_loaders_module = importlib.import_module(
                "langchain.document_loaders"
            )
        DocumentLoader = getattr(document_loaders_module, loader_name)
    except Exception as e:
        msg = f"为文件{file_path}查找加载器{loader_name}时出错：{e}"
        logger.error(
            f"{e.__class__.__name__}: {msg}", exc_info=e if log_verbose else None
        )
        document_loaders_module = importlib.import_module("langchain.document_loaders")
        DocumentLoader = getattr(document_loaders_module, "UnstructuredFileLoader")

    if loader_name == "UnstructuredFileLoader":
        loader_kwargs.setdefault("autodetect_encoding", True)
    elif loader_name == "CSVLoader":
        if not loader_kwargs.get("encoding"):
            # 如果未指定 encoding，自动识别文件编码类型，避免langchain loader 加载文件报编码错误
            with open(file_path, "rb") as struct_file:
                encode_detect = chardet.detect(struct_file.read())
            if encode_detect is None:
                encode_detect = {"encoding": "utf-8"}
            loader_kwargs["encoding"] = encode_detect["encoding"]
        ## TODO：支持更多的自定义CSV读取逻辑

    elif loader_name == "JSONLoader":
        loader_kwargs.setdefault("jq_schema", ".")
        loader_kwargs.setdefault("text_content", False)
    elif loader_name == "JSONLinesLoader":
        loader_kwargs.setdefault("jq_schema", ".")
        loader_kwargs.setdefault("text_content", False)

    loader = DocumentLoader(file_path, **loader_kwargs)
    return loader


# PPP# 根据用户配置文件，创建文章分段器对象,但是Tokenzier并没有卵用，实际上是用chinese_recursive_text_splitter.py直接对中文字进行分段的
def make_text_splitter(
    splitter_name: str = TEXT_SPLITTER_NAME,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = OVERLAP_SIZE,
    llm_model: str = LLM_MODELS[0],
):
    """
    根据参数获取特定的分段器对象,如果不提供就采用默认的SpacyTextSplitter
    """
    splitter_name = splitter_name or "SpacyTextSplitter"
    try:
        if (
            splitter_name == "MarkdownHeaderTextSplitter"
        ):  # MarkdownHeaderTextSplitter特殊判定
            headers_to_split_on = text_splitter_dict[splitter_name][
                "headers_to_split_on"
            ]
            text_splitter = langchain.text_splitter.MarkdownHeaderTextSplitter(
                headers_to_split_on=headers_to_split_on
            )
        else:
            try:  # 优先使用用户自定义的text_splitter(就是从项目根目录的text_splitter目录下的所有定义的类中寻找)
                text_splitter_module = importlib.import_module("text_splitter")
                TextSplitter = getattr(text_splitter_module, splitter_name)
            except:  ## 如果用户没有自定义的text_splitter，就使用langchain的text_splitter
                text_splitter_module = importlib.import_module(
                    "langchain.text_splitter"
                )
                TextSplitter = getattr(text_splitter_module, splitter_name)

            if (
                text_splitter_dict[splitter_name]["source"] == "tiktoken"
            ):  ## 如果分段器是tiktoken提供的(tiktoken是OpenAI开源的Python第三方模块，该模块主要实现了tokenizer的BPE（Byte pair encoding）算法)
                try:
                    text_splitter = TextSplitter.from_tiktoken_encoder(
                        encoding_name=text_splitter_dict[splitter_name][
                            "tokenizer_name_or_path"
                        ],
                        pipeline="zh_core_web_sm",
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )
                except:  # 如果没有安装tiktoken，就使用langchain的text_splitter
                    text_splitter = TextSplitter.from_tiktoken_encoder(
                        encoding_name=text_splitter_dict[splitter_name][
                            "tokenizer_name_or_path"
                        ],
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )
            elif (
                text_splitter_dict[splitter_name]["source"] == "huggingface"
            ):  ## 从huggingface加载 text splitter 实际上就是tokenizer
                if (
                    text_splitter_dict[splitter_name]["tokenizer_name_or_path"] == ""
                ):  # 如果tokenizer_name_or_path为空，则采用本地llm模型的model_path
                    config = get_model_worker_config(llm_model)
                    text_splitter_dict[splitter_name][
                        "tokenizer_name_or_path"
                    ] = config.get("model_path")

                if (
                    text_splitter_dict[splitter_name]["tokenizer_name_or_path"]
                    == "gpt2"  # 如果是gpt2的tokenizer
                ):
                    from transformers import GPT2TokenizerFast
                    from langchain.text_splitter import CharacterTextSplitter

                    tokenizer = GPT2TokenizerFast.from_pretrained(
                        "gpt2"
                    )  # 从huggingface加载gpt2的tokenizer
                else:  # 如果是其他tokenizer,那么从huggingface加载
                    from transformers import AutoTokenizer

                    tokenizer = AutoTokenizer.from_pretrained(
                        text_splitter_dict[splitter_name]["tokenizer_name_or_path"],
                        trust_remote_code=True,
                    )
                text_splitter = TextSplitter.from_huggingface_tokenizer(  # 给出tokenizer对象,创建text_splitter对象
                    tokenizer=tokenizer,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
            else:
                try:
                    text_splitter = TextSplitter(
                        pipeline="zh_core_web_sm",
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )
                except:
                    text_splitter = TextSplitter(
                        chunk_size=chunk_size, chunk_overlap=chunk_overlap
                    )
    except Exception as e:
        print(e)
        text_splitter_module = importlib.import_module("langchain.text_splitter")
        TextSplitter = getattr(text_splitter_module, "RecursiveCharacterTextSplitter")
        text_splitter = TextSplitter(chunk_size=250, chunk_overlap=50)
    return text_splitter


# PPP# 对磁盘的某个文件进行封装，用于加载、读取、转换为文字、切分
class KnowledgeFile:
    def __init__(
        self,
        filename: str,
        knowledge_base_name: str,
        loader_kwargs: Dict = {},
    ):
        """
        对应知识库目录中的文件，必须是磁盘上存在的才能进行向量化等操作。
        """
        self.kb_name = knowledge_base_name
        self.filename = str(Path(filename).as_posix())
        self.ext = os.path.splitext(filename)[-1].lower()
        if self.ext not in SUPPORTED_EXTS:
            raise ValueError(f"暂未支持的文件格式 {self.filename}")
        self.loader_kwargs = loader_kwargs
        self.filepath = get_file_path(knowledge_base_name, filename)
        self.docs = None
        self.splited_docs = None
        # 根据文件后缀找到对应的文件加载器
        self.document_loader_name = get_LoaderClass(self.ext)
        self.text_splitter_name = TEXT_SPLITTER_NAME

    # PPP### 加载文件
    def file2docs(self, refresh: bool = False):
        if self.docs is None or refresh:
            logger.info(f"{self.document_loader_name} used for {self.filepath}")
            loader = get_loader(
                loader_name=self.document_loader_name,
                file_path=self.filepath,
                loader_kwargs=self.loader_kwargs,
            )
            self.docs = loader.load()  # [Document]
        return self.docs

    # PPP## 把文档进行切分，返回切分后的多个文档（注意这个时候还没有tokenizer）
    def docs2texts(
        self,
        docs: List[Document] = None,
        zh_title_enhance: bool = ZH_TITLE_ENHANCE,
        refresh: bool = False,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = OVERLAP_SIZE,
        text_splitter: TextSplitter = None,
    ):
        """docs是输入，只包含一个Document,把这个Document切分成多个段落，形成多个切分后的Document，返回splited_docs数组"""
        docs = docs or self.file2docs(refresh=refresh)
        if not docs:
            return []
        if self.ext not in [".csv"]:
            if text_splitter is None:
                text_splitter = make_text_splitter(
                    splitter_name=self.text_splitter_name,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
            if self.text_splitter_name == "MarkdownHeaderTextSplitter":
                docs = text_splitter.split_text(docs[0].page_content)
            else:
                docs = text_splitter.split_documents(docs)

        if not docs:
            return []

        print(f"文档切分示例：{docs[0]}")
        if zh_title_enhance:
            docs = func_zh_title_enhance(docs)
        self.splited_docs = docs
        return self.splited_docs  # [Document]

    # PPP### 对文件加载、读取、转换为文字、切分
    def file2text(
        self,
        zh_title_enhance: bool = ZH_TITLE_ENHANCE,
        refresh: bool = False,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = OVERLAP_SIZE,
        text_splitter: TextSplitter = None,
    ):
        if self.splited_docs is None or refresh:
            docs = self.file2docs()
            self.splited_docs = self.docs2texts(
                docs=docs,
                zh_title_enhance=zh_title_enhance,
                refresh=refresh,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                text_splitter=text_splitter,
            )
        return self.splited_docs

    def file_exist(self):
        return os.path.isfile(self.filepath)

    def get_mtime(self):
        return os.path.getmtime(self.filepath)

    def get_size(self):
        return os.path.getsize(self.filepath)


def files2docs_in_thread(
    files: List[Union[KnowledgeFile, Tuple[str, str], Dict]],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = OVERLAP_SIZE,
    zh_title_enhance: bool = ZH_TITLE_ENHANCE,
) -> Generator:
    """
    # PPP# 利用多线程批量将磁盘文件转化成langchain Document.
    如果传入参数是Tuple，形式为(filename, kb_name)
    生成器返回值为 status, (kb_name, file_name, docs | error)
    """

    def file2docs(
        *, file: KnowledgeFile, **kwargs
    ) -> Tuple[bool, Tuple[str, str, List[Document]]]:
        try:
            return True, (file.kb_name, file.filename, file.file2text(**kwargs))
        except Exception as e:
            msg = f"从文件 {file.kb_name}/{file.filename} 加载文档时出错：{e}"
            logger.error(
                f"{e.__class__.__name__}: {msg}", exc_info=e if log_verbose else None
            )
            return False, (file.kb_name, file.filename, msg)

    kwargs_list = []
    for i, file in enumerate(files):
        kwargs = {}
        try:
            if isinstance(file, tuple) and len(file) >= 2:
                filename = file[0]
                kb_name = file[1]
                file = KnowledgeFile(filename=filename, knowledge_base_name=kb_name)
            elif isinstance(file, dict):
                filename = file.pop("filename")
                kb_name = file.pop("kb_name")
                kwargs.update(file)
                file = KnowledgeFile(filename=filename, knowledge_base_name=kb_name)
            kwargs["file"] = file
            kwargs["chunk_size"] = chunk_size
            kwargs["chunk_overlap"] = chunk_overlap
            kwargs["zh_title_enhance"] = zh_title_enhance
            kwargs_list.append(kwargs)
        except Exception as e:
            yield False, (kb_name, filename, str(e))

    for result in run_in_thread_pool(func=file2docs, params=kwargs_list):
        yield result


if __name__ == "__main__":
    from pprint import pprint

    kb_file = KnowledgeFile(
        filename="/home/congyin/Code/Project_Langchain_0814/Langchain-Chatchat/knowledge_base/csv1/content/gm.csv",
        knowledge_base_name="samples",
    )
    # kb_file.text_splitter_name = "RecursiveCharacterTextSplitter"
    docs = kb_file.file2docs()
    # pprint(docs[-1])
