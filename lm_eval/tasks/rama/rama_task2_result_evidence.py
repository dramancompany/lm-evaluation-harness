"""

적합 부적합 이유 예측
관련문서: https://dramancompany.atlassian.net/wiki/spaces/BDCAI/pages/28243296308/RAMA+PeT+Benchmark+Task+5

"""
from lm_eval.api.registry import register_task
from lm_eval.tasks.rama.rama_common import RAMAUtilsMixin
from lm_eval.api.task import MultipleChoiceTask


@register_task("prp")
class ReasonPrediction(MultipleChoiceTask, RAMAUtilsMixin):
    QUERY = """
아래 채용공고에 지원한 프로필이 {type}한 이유 대해서 가장 적합한 설명을 고르시오.

{input}

정답:    
"""

    VERSION = 1.0
    DATASET_PATH = "rama_project/rama_benchmark/llm_benchmark/v_{VERSION}/PRP_benchmark.json"
    DATASET_NAME = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def download(self, data_dir=None, cache_dir=None, download_mode=None) -> None:
        self.dataset = self.load_benchmark_dataset(self.DATASET_PATH.format_map({"VERSION": self.VERSION}))

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
        query = self.QUERY.format_map({"input": doc["input"], "type": doc["type"]})

        return {
            "query": query,
            "choices": doc["candidates"],
            "gold": doc["answer"],
        }

    def doc_to_text(self, doc):
        return doc["query"]
