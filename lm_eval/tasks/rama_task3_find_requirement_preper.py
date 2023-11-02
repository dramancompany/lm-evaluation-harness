"""

Task 7. 공고의 요구사항 찾기
관련문서: https://dramancompany.atlassian.net/wiki/spaces/BDCAI/pages/28246245377/RAMA+PeT+Benchmark+Task+3

"""
from lm_eval.base import MultipleChoiceTask
import json


class FindReqPrep(MultipleChoiceTask):
    QUERY = """
instruction:
공고가 주어집니다.
이 공고를 분석하고 target에서 지정한 필수 또는 우대조건에 해당하는 내용을 주어진 보기중에 모두 선택하세요.


target:
{target}

공고:
{jd}


정답: """

    VERSION = 1.0
    DATASET_PATH = "/raid/ailab-workspace/gyholee/project/rama_pet_benchmark/llm_pet/benchmark/data/FRP_benchmark.json"
    DATASET_NAME = None

    def __init__(self):
        self.dataset = json.load(open(self.DATASET_PATH))

        self._training_docs = None
        self._fewshot_docs = None

    def has_training_docs(self):
        return False

    def has_validation_docs(self):
        return False

    def has_test_docs(self):
        return True

    def training_docs(self):
        if self.has_training_docs():
            if self._training_docs is None:
                self._training_docs = list(map(self._process_doc, self.dataset["train"]))
            return self._training_docs

    def validation_docs(self):
        if self.has_validation_docs():
            return map(self._process_doc, self.dataset["validation"])

    def test_docs(self):
        if self.has_test_docs():
            return map(self._process_doc, self.dataset["test"])

    def _process_doc(self, doc):
        query = self.QUERY.format_map({"target": doc["target"], "jd": doc["jd"]})

        return {
            "query": query,
            "choices": doc["candidates"],
            "gold": doc["answer"],
        }

    def doc_to_text(self, doc):
        return doc["query"]
