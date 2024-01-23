import asyncio
import multiprocessing as mp
import os
import subprocess
import sys
from multiprocessing import Process
from datetime import datetime
from pprint import pprint


# 设置numexpr最大线程数，默认为CPU核心数
try:
    import numexpr

    n_cores = numexpr.utils.detect_number_of_cores()
    os.environ["NUMEXPR_MAX_THREADS"] = str(n_cores)
except:
    pass

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs import (
    LOG_PATH,
    log_verbose,
    logger,
    LLM_MODELS,
    EMBEDDING_MODEL,
    TEXT_SPLITTER_NAME,
    FSCHAT_CONTROLLER,
    FSCHAT_OPENAI_API,
    FSCHAT_MODEL_WORKERS,
    API_SERVER,
    WEBUI_SERVER,
    HTTPX_DEFAULT_TIMEOUT,
)
from server.utils import (
    fschat_controller_address,
    fschat_model_worker_address,
    fschat_openai_api_address,
    set_httpx_config,
    get_httpx_client,
    get_model_worker_config,
    get_all_model_worker_configs,
    MakeFastAPIOffline,
    FastAPI,
    llm_device,
    embedding_device,
)
from server.knowledge_base.migrate import create_tables
import argparse
from typing import Tuple, List, Dict
from configs import VERSION


def create_controller_app(
    dispatch_method: str,
    log_level: str = "INFO",
) -> FastAPI:
    import fastchat.constants

    fastchat.constants.LOGDIR = LOG_PATH
    from fastchat.serve.controller import app, Controller, logger

    logger.setLevel(log_level)

    controller = Controller(dispatch_method)
    sys.modules["fastchat.serve.controller"].controller = controller

    MakeFastAPIOffline(app)
    app.title = "FastChat Controller"
    app._controller = controller
    return app


# PPP### 启动本地LLM模型
def create_model_worker_app(log_level: str = "INFO", **kwargs) -> FastAPI:
    """
    kwargs包含的字段如下：
    host:
    port:
    model_names:[`model_name`]
    controller_address:
    worker_address:

    对于Langchain支持的模型：
        langchain_model:True  --- 实际上配置中根本没有
        不会使用fastchat

    对于online_api:   --- 在线大模型使用 online_api
        online_api:True
        worker_class: `provider`

    对于本地模型：
        model_path: `model_name_or_path`,huggingface的repo-id或本地路径
        device:`LLM_DEVICE`
    """
    import fastchat.constants

    fastchat.constants.LOGDIR = LOG_PATH
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args([])

    for k, v in kwargs.items():
        setattr(args, k, v)
    if worker_class := kwargs.get("langchain_model"):  # Langchian支持的模型不用做操作
        from fastchat.serve.base_model_worker import app

        worker = ""
    # 在线模型API
    elif worker_class := kwargs.get("worker_class"):
        from fastchat.serve.base_model_worker import app

        worker = worker_class(
            model_names=args.model_names,
            controller_addr=args.controller_address,
            worker_addr=args.worker_address,
        )
        # sys.modules["fastchat.serve.base_model_worker"].worker = worker
        sys.modules["fastchat.serve.base_model_worker"].logger.setLevel(log_level)
    # 本地模型
    else:
        from configs.model_config import VLLM_MODEL_DICT

        # 如果使用vllm推理加速框架
        if kwargs["model_names"][0] in VLLM_MODEL_DICT and args.infer_turbo == "vllm":
            import fastchat.serve.vllm_worker
            from fastchat.serve.vllm_worker import VLLMWorker, app, worker_id
            from vllm import AsyncLLMEngine
            from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs

            args.tokenizer = (
                args.model_path
            )  # 如果tokenizer与model_path不一致在此处添加
            args.tokenizer_mode = "auto"
            args.trust_remote_code = True
            args.download_dir = None
            args.load_format = "auto"
            args.dtype = "auto"
            args.seed = 0
            args.worker_use_ray = False
            args.pipeline_parallel_size = 1
            args.tensor_parallel_size = 1
            args.block_size = 16
            args.swap_space = 4  # GiB
            args.gpu_memory_utilization = 0.90
            args.max_num_batched_tokens = None  # 一个批次中的最大令牌（tokens）数量，这个取决于你的显卡和大模型设置，设置太大显存会不够
            args.max_num_seqs = 256
            args.disable_log_stats = False
            args.conv_template = None
            args.limit_worker_concurrency = 5
            args.no_register = False
            args.num_gpus = 1  # vllm worker的切分是tensor并行，这里填写显卡的数量
            args.engine_use_ray = False
            args.disable_log_requests = False

            # 0.2.1 vllm后要加的参数, 但是这里不需要
            args.max_model_len = None
            args.revision = None
            args.quantization = None
            args.max_log_len = None
            args.tokenizer_revision = None

            # 0.2.2 vllm需要新加的参数
            args.max_paddings = 256

            if args.model_path:
                args.model = args.model_path
            if args.num_gpus > 1:
                args.tensor_parallel_size = args.num_gpus

            for k, v in kwargs.items():
                setattr(args, k, v)

            engine_args = AsyncEngineArgs.from_cli_args(args)
            engine = AsyncLLMEngine.from_engine_args(engine_args)

            worker = VLLMWorker(
                controller_addr=args.controller_address,
                worker_addr=args.worker_address,
                worker_id=worker_id,
                model_path=args.model_path,
                model_names=args.model_names,
                limit_worker_concurrency=args.limit_worker_concurrency,
                no_register=args.no_register,
                llm_engine=engine,
                conv_template=args.conv_template,
            )
            sys.modules["fastchat.serve.vllm_worker"].engine = engine
            sys.modules["fastchat.serve.vllm_worker"].worker = worker
            sys.modules["fastchat.serve.vllm_worker"].logger.setLevel(log_level)
        # 如果不使用VLLM
        else:
            from fastchat.serve.model_worker import (
                app,
                GptqConfig,
                AWQConfig,
                ModelWorker,
                worker_id,
            )

            args.gpus = "0"  # GPU的编号,如果有多个GPU，可以设置为"0,1,2,3"
            args.max_gpu_memory = "22GiB"
            args.num_gpus = 1  # model worker的切分是model并行，这里填写显卡的数量

            args.load_8bit = False
            args.cpu_offloading = None
            args.gptq_ckpt = None
            args.gptq_wbits = 16
            args.gptq_groupsize = -1
            args.gptq_act_order = False
            args.awq_ckpt = None
            args.awq_wbits = 16
            args.awq_groupsize = -1
            args.model_names = [""]
            args.conv_template = None
            args.limit_worker_concurrency = 5
            args.stream_interval = 2
            args.no_register = False
            args.embed_in_truncate = False
            for k, v in kwargs.items():
                setattr(args, k, v)
            if args.gpus:
                if args.num_gpus is None:
                    args.num_gpus = len(args.gpus.split(","))
                if len(args.gpus.split(",")) < args.num_gpus:
                    raise ValueError(
                        f"Larger --num-gpus ({args.num_gpus}) than --gpus {args.gpus}!"
                    )
                os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
            gptq_config = GptqConfig(
                ckpt=args.gptq_ckpt or args.model_path,
                wbits=args.gptq_wbits,
                groupsize=args.gptq_groupsize,
                act_order=args.gptq_act_order,
            )
            awq_config = AWQConfig(
                ckpt=args.awq_ckpt or args.model_path,
                wbits=args.awq_wbits,
                groupsize=args.awq_groupsize,
            )

            worker = ModelWorker(
                controller_addr=args.controller_address,
                worker_addr=args.worker_address,
                worker_id=worker_id,
                model_path=args.model_path,
                model_names=args.model_names,
                limit_worker_concurrency=args.limit_worker_concurrency,
                no_register=args.no_register,
                device=args.device,
                num_gpus=args.num_gpus,
                max_gpu_memory=args.max_gpu_memory,
                load_8bit=args.load_8bit,
                cpu_offloading=args.cpu_offloading,
                gptq_config=gptq_config,
                awq_config=awq_config,
                stream_interval=args.stream_interval,
                conv_template=args.conv_template,
                embed_in_truncate=args.embed_in_truncate,
            )
            sys.modules["fastchat.serve.model_worker"].args = args
            sys.modules["fastchat.serve.model_worker"].gptq_config = gptq_config
            # sys.modules["fastchat.serve.model_worker"].worker = worker
            sys.modules["fastchat.serve.model_worker"].logger.setLevel(log_level)

    MakeFastAPIOffline(app)
    app.title = f"FastChat LLM Server ({args.model_names[0]})"
    app._worker = worker
    return app


def create_openai_api_app(
    controller_address: str,
    api_keys: List = [],
    log_level: str = "INFO",
) -> FastAPI:
    import fastchat.constants

    fastchat.constants.LOGDIR = LOG_PATH
    from fastchat.serve.openai_api_server import app, CORSMiddleware, app_settings
    from fastchat.utils import build_logger

    logger = build_logger("openai_api", "openai_api.log")
    logger.setLevel(log_level)

    app.add_middleware(
        CORSMiddleware,
        allow_credentials=True,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    sys.modules["fastchat.serve.openai_api_server"].logger = logger
    app_settings.controller_address = controller_address
    app_settings.api_keys = api_keys

    MakeFastAPIOffline(app)
    app.title = "FastChat OpeanAI API Server"
    return app


def _set_app_event(app: FastAPI, started_event: mp.Event = None):
    @app.on_event("startup")
    async def on_startup():
        if started_event is not None:
            started_event.set()


def run_controller(log_level: str = "INFO", started_event: mp.Event = None):
    import uvicorn
    import httpx
    from fastapi import Body
    import time
    import sys
    from server.utils import set_httpx_config

    set_httpx_config()

    app = create_controller_app(
        dispatch_method=FSCHAT_CONTROLLER.get("dispatch_method"),
        log_level=log_level,
    )
    _set_app_event(app, started_event)

    # add interface to release and load model worker
    @app.post("/release_worker")
    def release_worker(
        model_name: str = Body(
            ..., description="要释放模型的名称", samples=["chatglm-6b"]
        ),
        # worker_address: str = Body(None, description="要释放模型的地址，与名称二选一", samples=[FSCHAT_CONTROLLER_address()]),
        new_model_name: str = Body(None, description="释放后加载该模型"),
        keep_origin: bool = Body(False, description="不释放原模型，加载新模型"),
    ) -> Dict:
        available_models = app._controller.list_models()
        if new_model_name in available_models:
            msg = f"要切换的LLM模型 {new_model_name} 已经存在"
            logger.info(msg)
            return {"code": 500, "msg": msg}

        if new_model_name:
            logger.info(f"开始切换LLM模型：从 {model_name} 到 {new_model_name}")
        else:
            logger.info(f"即将停止LLM模型： {model_name}")

        if model_name not in available_models:
            msg = f"the model {model_name} is not available"
            logger.error(msg)
            return {"code": 500, "msg": msg}

        worker_address = app._controller.get_worker_address(model_name)
        if not worker_address:
            msg = f"can not find model_worker address for {model_name}"
            logger.error(msg)
            return {"code": 500, "msg": msg}

        with get_httpx_client() as client:
            r = client.post(
                worker_address + "/release",
                json={"new_model_name": new_model_name, "keep_origin": keep_origin},
            )
            if r.status_code != 200:
                msg = f"failed to release model: {model_name}"
                logger.error(msg)
                return {"code": 500, "msg": msg}

        if new_model_name:
            timer = HTTPX_DEFAULT_TIMEOUT  # wait for new model_worker register
            while timer > 0:
                models = app._controller.list_models()
                if new_model_name in models:
                    break
                time.sleep(1)
                timer -= 1
            if timer > 0:
                msg = f"sucess change model from {model_name} to {new_model_name}"
                logger.info(msg)
                return {"code": 200, "msg": msg}
            else:
                msg = f"failed change model from {model_name} to {new_model_name}"
                logger.error(msg)
                return {"code": 500, "msg": msg}
        else:
            msg = f"sucess to release model: {model_name}"
            logger.info(msg)
            return {"code": 200, "msg": msg}

    host = FSCHAT_CONTROLLER["host"]
    port = FSCHAT_CONTROLLER["port"]

    if log_level == "ERROR":
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    uvicorn.run(app, host=host, port=port, log_level=log_level.lower())


# PPP## 启动本地LLM模型
def run_model_worker(
    model_name: str = LLM_MODELS[0],
    controller_address: str = "",
    log_level: str = "INFO",
    q: mp.Queue = None,
    started_event: mp.Event = None,
):
    import uvicorn
    from fastapi import Body
    import sys
    from server.utils import set_httpx_config

    set_httpx_config()

    kwargs = get_model_worker_config(model_name)
    host = kwargs.pop("host")
    port = kwargs.pop("port")
    kwargs["model_names"] = [model_name]
    kwargs["controller_address"] = controller_address or fschat_controller_address()
    kwargs["worker_address"] = fschat_model_worker_address(model_name)
    model_path = kwargs.get("model_path", "")
    kwargs["model_path"] = model_path

    app = create_model_worker_app(log_level=log_level, **kwargs)
    _set_app_event(app, started_event)
    if log_level == "ERROR":
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__

    # add interface to release and load model
    @app.post("/release")
    def release_model(
        new_model_name: str = Body(None, description="释放后加载该模型"),
        keep_origin: bool = Body(False, description="不释放原模型，加载新模型"),
    ) -> Dict:
        if keep_origin:
            if new_model_name:
                q.put([model_name, "start", new_model_name])
        else:
            if new_model_name:
                q.put([model_name, "replace", new_model_name])
            else:
                q.put([model_name, "stop", None])
        return {"code": 200, "msg": "done"}

    uvicorn.run(app, host=host, port=port, log_level=log_level.lower())


def run_openai_api(log_level: str = "INFO", started_event: mp.Event = None):
    import uvicorn
    import sys
    from server.utils import set_httpx_config

    set_httpx_config()

    controller_addr = fschat_controller_address()
    app = create_openai_api_app(
        controller_addr, log_level=log_level
    )  # TODO: not support keys yet.
    _set_app_event(app, started_event)

    host = FSCHAT_OPENAI_API["host"]
    port = FSCHAT_OPENAI_API["port"]
    if log_level == "ERROR":
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
    uvicorn.run(app, host=host, port=port)


def run_api_server(started_event: mp.Event = None, run_mode: str = None):
    from server.api import create_app
    import uvicorn
    from server.utils import set_httpx_config

    set_httpx_config()

    app = create_app(run_mode=run_mode)
    _set_app_event(app, started_event)

    host = API_SERVER["host"]
    port = API_SERVER["port"]

    """ 
    uvicorn.run是一个用于运行Python Web应用程序的命令行工具，它使用了ASGI（Asynchronous Server Gateway Interface）接口来实现异步Web服务器和应用程序之间的通信。通过调用uvicorn.run函数，可以将Python Web应用程序作为HTTP服务器运行，并在给定的主机和端口上监听HTTP请求。它通常用于运行基于Python的Web框架（如FastAPI、Starlette等）的生产级Web服务。
    """
    # PPP### 运行api server
    uvicorn.run(app, host=host, port=port)


""" 
# PPP# 运行WebUI的入口函数。它使用subprocess模块启动一个子进程来运行WebUI应用程序。它接受两个参数：started_event和run_mode。started_event是一个可选参数，它是一个multiprocessing模块的Event对象，用于在WebUI应用程序开始运行时设置。run_mode是可选参数，它用于指定运行模式，如果为"lite"，则在命令行参数中添加"--lite"。函数最后会等待子进程的结束。
 """


def run_webui(started_event: mp.Event = None, run_mode: str = None):
    from server.utils import set_httpx_config

    set_httpx_config()

    host = WEBUI_SERVER["host"]
    port = WEBUI_SERVER["port"]

    cmd = [
        "streamlit",
        "run",
        "webui.py",  # webui的主函数
        "--server.address",
        host,
        "--server.port",
        str(port),
        "--theme.base",
        "light",
        "--theme.primaryColor",
        "#165dff",
        "--theme.secondaryBackgroundColor",
        "#f5f5f5",
        "--theme.textColor",
        "#000000",
    ]
    if run_mode == "lite":
        cmd += [
            "--",
            "lite",
        ]

    p = subprocess.Popen(cmd)  # 启动子进程
    started_event.set()
    p.wait()  # 等待子进程的结束


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    # 全套
    parser.add_argument(
        "-a",
        "--all-webui",
        action="store_true",
        help="run fastchat's controller/openai_api/model_worker servers, run api.py and webui.py",
        dest="all_webui",
    )

    # 加载全套后台，不运行webui
    parser.add_argument(
        "--all-api",
        action="store_true",
        help="run fastchat's controller/openai_api/model_worker servers, run api.py",
        dest="all_api",
    )

    # 加载本地模型后台
    parser.add_argument(
        "--llm-api",
        action="store_true",
        help="run fastchat's controller/openai_api/model_worker servers",
        dest="llm_api",
    )

    # fastchat后台，不加载本地模型，
    parser.add_argument(
        "-o",
        "--openai-api",
        action="store_true",
        help="run fastchat's controller/openai_api servers",
        dest="openai_api",
    )

    # 加载--model-name指定名字的模型
    parser.add_argument(
        "-m",
        "--model-worker",
        action="store_true",
        help="run fastchat's model_worker server with specified model name. "
        "specify --model-name if not using default LLM_MODELS",
        dest="model_worker",
    )

    # 指定加载的模型名字
    parser.add_argument(
        "-n",
        "--model-name",
        type=str,
        nargs="+",
        default=LLM_MODELS,
        help="specify model name for model worker. "
        "add addition names with space seperated to start multiple model workers.",
        dest="model_name",
    )

    # 指定fastchat controller的监听地址端口
    parser.add_argument(
        "-c",
        "--controller",
        type=str,
        help="specify controller address the worker is registered to. default is FSCHAT_CONTROLLER",
        dest="controller_address",
    )

    # 运行在线llm的API服务监听
    parser.add_argument(
        "--api",
        action="store_true",
        help="run api.py server",
        dest="api",
    )

    # 运行在线llm的api服务监听-可以指定名字
    parser.add_argument(
        "-p",
        "--api-worker",
        action="store_true",
        help="run online model api such as zhipuai",
        dest="api_worker",
    )

    # 运行网站UI
    parser.add_argument(
        "-w",
        "--webui",
        action="store_true",
        help="run webui.py server",
        dest="webui",
    )

    # 减少fastchat服务log信息
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="减少fastchat服务log信息",
        dest="quiet",
    )

    # 以Lite模式运行：仅支持在线API的LLM对话、搜索引擎对话
    parser.add_argument(
        "-i",
        "--lite",
        action="store_true",
        help="以Lite模式运行：仅支持在线API的LLM对话、搜索引擎对话",
        dest="lite",
    )
    args = parser.parse_args()
    return args, parser


def dump_server_info(after_start=False, args=None):
    import platform
    import langchain
    import fastchat
    from server.utils import api_address, webui_address

    print("\n")
    print("=" * 30 + "Langchain-Chatchat Configuration" + "=" * 30)
    print(f"操作系统：{platform.platform()}.")
    print(f"python版本：{sys.version}")
    print(f"项目版本：{VERSION}")
    print(
        f"langchain版本：{langchain.__version__}. fastchat版本：{fastchat.__version__}"
    )
    print("\n")

    models = LLM_MODELS
    if args and args.model_name:
        models = args.model_name

    print(f"当前使用的分段器：{TEXT_SPLITTER_NAME}")
    print(f"当前启动的LLM模型：{models} @ {llm_device()}")

    for model in models:
        pprint(get_model_worker_config(model))
    print(f"当前Embbedings模型： {EMBEDDING_MODEL} @ {embedding_device()}")

    if after_start:
        print("\n")
        print(f"服务端运行信息：")
        if args.openai_api:
            print(f"    OpenAI API Server: {fschat_openai_api_address()}")
        if args.api:
            print(f"    Chatchat  API  Server: {api_address()}")
        if args.webui:
            print(f"    Chatchat WEBUI Server: {webui_address()}")
    print("=" * 30 + "Langchain-Chatchat Configuration" + "=" * 30)
    print("\n")


# PPP# 在新开的线程中启动服务端和webui
async def start_main_server():
    import time
    import signal

    def handler(signalname):
        """
        Python 3.9 has `signal.strsignal(signalnum)` so this closure would not be needed.
        Also, 3.8 includes `signal.valid_signals()` that can be used to create a mapping for the same purpose.
        """

        def f(signal_received, frame):
            raise KeyboardInterrupt(f"{signalname} received")

        return f

    # This will be inherited by the child process if it is forked (not spawned)
    signal.signal(signal.SIGINT, handler("SIGINT"))
    signal.signal(signal.SIGTERM, handler("SIGTERM"))

    mp.set_start_method("spawn")
    manager = mp.Manager()
    run_mode = None

    queue = manager.Queue()
    args, parser = parse_args()

    if args.all_webui:
        args.openai_api = True
        args.model_worker = True
        args.api = True
        args.api_worker = True
        args.webui = True

    # 启动后台服务
    elif args.all_api:
        args.openai_api = True
        args.model_worker = True
        args.api = True
        args.api_worker = True
        args.webui = False

    elif args.llm_api:
        args.openai_api = True
        args.model_worker = True
        args.api_worker = True
        args.api = False
        args.webui = False

    # 只使用在线的大模型
    if args.lite:
        args.model_worker = False
        run_mode = "lite"

    dump_server_info(args=args)

    if len(sys.argv) > 1:
        logger.info(f"正在启动服务：")
        logger.info(f"如需查看 llm_api 日志，请前往 {LOG_PATH}")

    processes = {"online_api": {}, "model_worker": {}}

    def process_count():
        return (
            len(processes)
            + len(processes["online_api"])
            + len(processes["model_worker"])
            - 2
        )

    if args.quiet or not log_verbose:
        log_level = "ERROR"
    else:
        log_level = "INFO"

    controller_started = manager.Event()
    if args.openai_api:
        process = Process(
            target=run_controller,
            name=f"controller",
            kwargs=dict(log_level=log_level, started_event=controller_started),
            daemon=True,
        )
        processes["controller"] = process

        process = Process(
            target=run_openai_api,
            name=f"openai_api",
            daemon=True,
        )
        processes["openai_api"] = process

    # 加载本地llm模型
    model_worker_started = []  # 用于存储已启动的模型工作器的事件对象
    if args.model_worker:  # 判断是否需要启动模型工作器
        for model_name in args.model_name:  # 遍历模型名称列表
            config = get_model_worker_config(model_name)  # 获取模型工作器的配置信息
            if not config.get("online_api"):  # 判断配置信息中是否包含"online_api"字段
                e = manager.Event()  # 创建一个事件对象
                model_worker_started.append(
                    e
                )  # 将事件对象添加到已启动的模型工作器列表中
                process = Process(  # 创建一个新的进程
                    target=run_model_worker,  # 设置进程的目标函数为run_model_worker
                    name=f"model_worker - {model_name}",  # 设置进程的名称
                    kwargs=dict(  # 设置进程的参数字典
                        model_name=model_name,  # 模型名称
                        controller_address=args.controller_address,  # 控制器地址
                        log_level=log_level,  # 日志级别
                        q=queue,  # 队列对象
                        started_event=e,  # 已启动的模型工作器的事件对象
                    ),
                    daemon=True,  # 设置为守护线程，后台运行
                )
                processes["model_worker"][
                    model_name
                ] = process  # 将进程对象添加到进程字典中

    if args.api_worker:
        for model_name in args.model_name:  # PPP## 子进程方式启动本地大模型
            config = get_model_worker_config(model_name)
            if (
                config.get("online_api")
                and config.get("worker_class")
                and model_name in FSCHAT_MODEL_WORKERS
            ):
                e = manager.Event()
                model_worker_started.append(e)
                process = Process(
                    target=run_model_worker,
                    name=f"api_worker - {model_name}",
                    kwargs=dict(
                        model_name=model_name,
                        controller_address=args.controller_address,
                        log_level=log_level,
                        q=queue,
                        started_event=e,
                    ),
                    daemon=True,
                )
                processes["online_api"][model_name] = process

    api_started = manager.Event()
    if not args.api:  # PPP## （临时关闭api服务用于调试）子进程方式启动服务端的api
        process = Process(
            target=run_api_server,
            name=f"API Server",
            kwargs=dict(started_event=api_started, run_mode=run_mode),
            daemon=True,
        )
        processes["api"] = process

    webui_started = manager.Event()
    if args.webui:  # PPP## 子进程方式启动webui
        process = Process(
            target=run_webui,
            name=f"WEBUI Server",
            kwargs=dict(started_event=webui_started, run_mode=run_mode),
            daemon=True,
        )
        processes["webui"] = process

    if process_count() == 0:
        parser.print_help()
    else:
        try:
            # 保证任务收到SIGINT后，能够正常退出
            if p := processes.get("controller"):
                p.start()
                p.name = f"{p.name} ({p.pid})"
                controller_started.wait()  # 等待controller启动完成

            if p := processes.get("openai_api"):
                p.start()
                p.name = f"{p.name} ({p.pid})"

            for n, p in processes.get("model_worker", {}).items():
                p.start()
                p.name = f"{p.name} ({p.pid})"

            for n, p in processes.get("online_api", []).items():
                p.start()
                p.name = f"{p.name} ({p.pid})"

            # 等待所有model_worker启动完成
            for e in model_worker_started:
                e.wait()

            if p := processes.get("api"):
                p.start()
                p.name = f"{p.name} ({p.pid})"
                api_started.wait()  # 等待api.py启动完成

            if p := processes.get("webui"):
                p.start()
                p.name = f"{p.name} ({p.pid})"
                webui_started.wait()  # 等待webui.py启动完成

            dump_server_info(after_start=True, args=args)

            while True:  # 等待前面的服务加载完成后，执行到这里
                cmd = queue.get()  # 收到切换模型的消息
                e = manager.Event()
                if isinstance(cmd, list):
                    model_name, cmd, new_model_name = cmd
                    if cmd == "start":  # 运行新模型
                        logger.info(f"准备启动新模型进程：{new_model_name}")
                        process = Process(
                            target=run_model_worker,
                            name=f"model_worker - {new_model_name}",
                            kwargs=dict(
                                model_name=new_model_name,
                                controller_address=args.controller_address,
                                log_level=log_level,
                                q=queue,
                                started_event=e,
                            ),
                            daemon=True,
                        )
                        process.start()
                        process.name = f"{process.name} ({process.pid})"
                        processes["model_worker"][new_model_name] = process
                        e.wait()
                        logger.info(f"成功启动新模型进程：{new_model_name}")
                    elif cmd == "stop":
                        if process := processes["model_worker"].get(model_name):
                            time.sleep(1)
                            process.terminate()
                            process.join()
                            logger.info(f"停止模型进程：{model_name}")
                        else:
                            logger.error(f"未找到模型进程：{model_name}")
                    elif cmd == "replace":
                        if process := processes["model_worker"].pop(model_name, None):
                            logger.info(f"停止模型进程：{model_name}")
                            start_time = datetime.now()
                            time.sleep(1)
                            process.terminate()
                            process.join()
                            process = Process(
                                target=run_model_worker,
                                name=f"model_worker - {new_model_name}",
                                kwargs=dict(
                                    model_name=new_model_name,
                                    controller_address=args.controller_address,
                                    log_level=log_level,
                                    q=queue,
                                    started_event=e,
                                ),
                                daemon=True,
                            )
                            process.start()
                            process.name = f"{process.name} ({process.pid})"
                            processes["model_worker"][new_model_name] = process
                            e.wait()
                            timing = datetime.now() - start_time
                            logger.info(
                                f"成功启动新模型进程：{new_model_name}。用时：{timing}。"
                            )
                        else:
                            logger.error(f"未找到模型进程：{model_name}")

            # for process in processes.get("model_worker", {}).values():
            #     process.join()
            # for process in processes.get("online_api", {}).values():
            #     process.join()

            # for name, process in processes.items():
            #     if name not in ["model_worker", "online_api"]:
            #         if isinstance(p, dict):
            #             for work_process in p.values():
            #                 work_process.join()
            #         else:
            #             process.join()
        except Exception as e:
            logger.error(e)
            logger.warning("Caught KeyboardInterrupt! Setting stop event...")
        finally:
            # Send SIGINT if process doesn't exit quickly enough, and kill it as last resort
            # .is_alive() also implicitly joins the process (good practice in linux)
            # while alive_procs := [p for p in processes.values() if p.is_alive()]:

            for p in processes.values():
                logger.warning("Sending SIGKILL to %s", p)
                # Queues and other inter-process communication primitives can break when
                # process is killed, but we don't care here

                if isinstance(p, dict):
                    for process in p.values():
                        process.kill()
                else:
                    p.kill()

            for p in processes.values():
                logger.info("Process status: %s", p)


if __name__ == "__main__":
    # 确保数据库表被创建
    create_tables()

    if sys.version_info < (3, 10):
        loop = asyncio.get_event_loop()
    else:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()

        asyncio.set_event_loop(loop)
    # 同步调用协程代码
    loop.run_until_complete(start_main_server())


# 服务启动后接口调用示例：
# import openai
# openai.api_key = "EMPTY" # Not support yet
# openai.api_base = "http://localhost:8888/v1"

# model = "chatglm2-6b"

# # create a chat completion
# completion = openai.ChatCompletion.create(
#   model=model,
#   messages=[{"role": "user", "content": "Hello! What is your name?"}]
# )
# # print the completion
# print(completion.choices[0].message.content)
