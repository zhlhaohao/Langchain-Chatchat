import re
from typing import List, Optional, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging

logger = logging.getLogger(__name__)


def _split_text_with_regex_from_end(
    text: str, separator: str, keep_separator: bool
) -> List[str]:
    # 现在我们有了分隔符，将文本进行拆分
    if separator:
        if keep_separator:
            # 正则表达式中的括号使得结果中保留了分隔符
            _splits = re.split(f"({separator})", text)
            # 将相邻的两个拆分结果进行拼接
            splits = ["".join(i) for i in zip(_splits[0::2], _splits[1::2])]

            # 如果拆分结果的数量为奇数，将最后一个分隔符单独取出来
            if len(_splits) % 2 == 1:
                splits += _splits[-1:]
            # splits = [_splits[0]] + splits
        else:
            # 使用分隔符进行拆分
            splits = re.split(separator, text)
    else:
        # 如果没有分隔符，将文本转换为列表
        splits = list(text)
    # 过滤掉结果中的空字符串
    return [s for s in splits if s != ""]


# PPP# 中文文本递归分段器
class ChineseRecursiveTextSplitter(RecursiveCharacterTextSplitter):
    def __init__(
        self,
        separators: Optional[List[str]] = None,
        keep_separator: bool = True,
        is_separator_regex: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        RecursiveCharacterTextSplitter是一个文本分割器，主要用于对文本进行拆分。它在LangChain中是一种常用的分割器，特别适用于处理一般文档，如纯文本或文本和代码的混合等。
        这个分割器的工作原理是按不同的字符进行递归分割，其优先级为["\n\n", "\n", " ", ""]，尽量将所有和语义相关的内容尽可能长时间地保留在同一位置。这种分割方式有助于将文本分成更小的、具有语义连贯性的片段。
        """

        """Create a new TextSplitter."""
        super().__init__(keep_separator=keep_separator, **kwargs)

        # 定义分割词，在这里是中英文的标点符号以及回车换行符
        self._separators = separators or [
            "\n\n",
            "\n",
            "。|！|？",
            "\.\s|\!\s|\?\s",
            "；|;\s",
            "，|,\s",
        ]
        self._is_separator_regex = is_separator_regex

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """
        将传入的文本分割成块并返回。

        参数：
        - text (str): 要处理的文本
        - separators (List[str]): 分隔符列表

        过程：
        1. 从分隔符列表中从前向后选择分隔符，使用从尾部开始的正则表达式对文本进行分割
        4. 遍历分割后的文本片段，将长度小于chunk_size的片段添加到_good_splits中，将_good_splits中的短片段拼接成一个稍稍大于chunk_size的长文本块，然后添加到final_chunks中
        5. 对于长度大于chunk_size的片段，使用新的分隔符列表递归地分割长片段直到小于chunk_size,这段长文本中间确实没有分隔符，那只好不分割了，直接加入到final_chunks中
        7. 返回去除多余换行并过滤掉空白内容的final_chunks中的所有文本片段
        """

        final_chunks = []  # 存储最终分割出的文本块

        # 获取合适的分隔符，优先选择最后一个在text中出现过的分隔符
        separator = separators[-1]
        new_separators = []
        for i, _s in enumerate(separators):
            escaped_s = _s if self._is_separator_regex else re.escape(_s)
            if _s == "":
                separator = _s
                break
            if re.search(escaped_s, text):
                separator = _s
                new_separators = separators[i + 1 :]
                break

        # 如果分隔符不是正则表达式，则对其进行转义
        _separator = separator if self._is_separator_regex else re.escape(separator)

        # 从文本尾部开始，根据分隔符进行正则分割
        splits = _split_text_with_regex_from_end(text, _separator, self._keep_separator)

        # 合并过短的文本片段
        _good_splits = []
        effective_separator = "" if self._keep_separator else separator
        for s in splits:
            if (
                self._length_function(s) < self._chunk_size
            ):  # 如果长度小于chunk_size，那么将其添加到_good_splits中
                _good_splits.append(s)
            else:
                # 若_good_splits已有内容，那么先合并并添加到final_chunks，然后清空
                if _good_splits:
                    merged_text = self._merge_splits(_good_splits, effective_separator)
                    final_chunks.extend(merged_text)
                    _good_splits = []

                if not new_separators:
                    final_chunks.append(s)
                else:
                    # 如果当前片段大于chunk_size，且还有分隔符可以尝试，则对当前长片段进行递归分割
                    other_info = self._split_text(s, new_separators)
                    final_chunks.extend(other_info)

        # 处理最后剩下的_good_splits
        if _good_splits:
            merged_text = self._merge_splits(_good_splits, effective_separator)
            final_chunks.extend(merged_text)

        # 清理并返回结果：去除连续多余的换行并将空白内容过滤掉
        return [
            re.sub(r"\n{2,}", "\n", chunk.strip())
            for chunk in final_chunks
            if chunk.strip() != ""
        ]


if __name__ == "__main__":
    text_splitter = ChineseRecursiveTextSplitter(
        keep_separator=True, is_separator_regex=True, chunk_size=50, chunk_overlap=0
    )
    ls = [
        """中国对外贸易形势报告（75页）。前 10 个月，一般贸易进出口 19.5 万亿元，增长 25.1%， 比整体进出口增速高出 2.9 个百分点，占进出口总额的 61.7%，较去年同期提升 1.6 个百分点。其中，一般贸易出口 10.6 万亿元，增长 25.3%，占出口总额的 60.9%，提升 1.5 个百分点；进口8.9万亿元，增长24.9%，占进口总额的62.7%， 提升 1.8 个百分点。加工贸易进出口 6.8 万亿元，增长 11.8%， 占进出口总额的 21.5%，减少 2.0 个百分点。其中，出口增 长 10.4%，占出口总额的 24.3%，减少 2.6 个百分点；进口增 长 14.2%，占进口总额的 18.0%，减少 1.2 个百分点。此外， 以保税物流方式进出口 3.96 万亿元，增长 27.9%。其中，出 口 1.47 万亿元，增长 38.9%；进口 2.49 万亿元，增长 22.2%。前三季度，中国服务贸易继续保持快速增长态势。服务 进出口总额 37834.3 亿元，增长 11.6%；其中服务出口 17820.9 亿元，增长 27.3%；进口 20013.4 亿元，增长 0.5%，进口增 速实现了疫情以来的首次转正。服务出口增幅大于进口 26.8 个百分点，带动服务贸易逆差下降 62.9%至 2192.5 亿元。服 务贸易结构持续优化，知识密集型服务进出口 16917.7 亿元， 增长 13.3%，占服务进出口总额的比重达到 44.7%，提升 0.7 个百分点。 二、中国对外贸易发展环境分析和展望 全球疫情起伏反复，经济复苏分化加剧，大宗商品价格 上涨、能源紧缺、运力紧张及发达经济体政策调整外溢等风 险交织叠加。同时也要看到，我国经济长期向好的趋势没有 改变，外贸企业韧性和活力不断增强，新业态新模式加快发 展，创新转型步伐提速。产业链供应链面临挑战。美欧等加快出台制造业回迁计 划，加速产业链供应链本土布局，跨国公司调整产业链供应 链，全球双链面临新一轮重构，区域化、近岸化、本土化、 短链化趋势凸显。疫苗供应不足，制造业“缺芯”、物流受限、 运价高企，全球产业链供应链面临压力。 全球通胀持续高位运行。能源价格上涨加大主要经济体 的通胀压力，增加全球经济复苏的不确定性。世界银行今年 10 月发布《大宗商品市场展望》指出，能源价格在 2021 年 大涨逾 80%，并且仍将在 2022 年小幅上涨。IMF 指出，全 球通胀上行风险加剧，通胀前景存在巨大不确定性。""",
    ]
    # text = """"""
    for inum, text in enumerate(ls):
        print(inum)
        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            print(chunk)
