"""

공고/프로필의 항목 분류
관련문서: https://dramancompany.atlassian.net/wiki/spaces/BDCAI/pages/28241461371/RAMA+PET+Benchmark+Task+4

"""
from lm_eval.base import MultipleChoiceTask
import json
from lm_eval.tasks.rama_common import RAMAUtilsMixin


class SelectProperCandidates(MultipleChoiceTask, RAMAUtilsMixin):
    QUERY = """
instruction:
주어진 채용공고와 회사정보를 고려할때, 더 적절한 후보자를 선택하세요.
 
공고:
{jd}

후보: 
"""

    VERSION = 1.0
    DATASET_PATH = "rama_project/rama_benchmark/llm_benchmark/v_{VERSION}/SPC_benchmark.json"
    DATASET_NAME = None

    def __init__(self):
        self.dataset = self.load_benchmark_dataset(self.DATASET_PATH.format_map({"VERSION": self.VERSION}))

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
        query = self.QUERY.format_map({"jd": doc["jd"]})

        return {
            "type": doc["type"],
            "query": query,
            "choices": doc["candidates"],
            "gold": doc["answer"],
        }

    def doc_to_text(self, doc):
        return doc["query"]
