"""

Task 7. 유사회사 찾기
관련문서: https://dramancompany.atlassian.net/wiki/spaces/BDCAI/pages/28245557261/RAMA+PeT+Benchmark+Task+7

"""
from lm_eval.base import MultipleChoiceTask
import json


class FindSimilarCompany(MultipleChoiceTask):
    QUERY = """
instruction:
입력된 유사회사 정의에 따라, target으로 주어진 회사와 유사한 회사를 선택하세요.

유사회사 정의:
{definition}

target:
{target}


정답: """

    VERSION = 1.0
    DATASET_PATH = "/raid/ailab-workspace/gyholee/project/rama_pet_benchmark/llm_pet/benchmark/data/benchmark_find_similar_company.json"
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
        query = self.QUERY.format_map({"target": doc["target"], "definition": doc["definition"]})

        return {
            "query": query,
            "choices": doc["candidates"],
            "gold": doc["answer"],
        }

    def doc_to_text(self, doc):
        return doc["query"]
