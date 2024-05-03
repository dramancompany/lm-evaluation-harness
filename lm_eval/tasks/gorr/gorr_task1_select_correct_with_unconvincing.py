from lm_eval.tasks.gorr.gorr_common import GORRUtilsMixin
from lm_eval.api.registry import register_task
from lm_eval.api.task import MultipleChoiceTask


@register_task("unconvincing")
class SelectCorrectOfferMessageUnconvincing(MultipleChoiceTask, GORRUtilsMixin):
    QUERY = """
instruction:
주어진 <공고>와 <프로필>을 고려할 때, 가장 적절한 제안 메시지를 선택하세요.

<공고>:
{jd}

<프로필>:
{profile}

<제안 메시지>:
"""

    VERSION = 1.0
    DATASET_PATH = f"llm_evaluation/benchmark/personalized_offer_message/multiple_choice/v_{VERSION}/benchmark_unconvincing.json"
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
        query = self.QUERY.format_map({"jd": doc["jd"], "profile": doc["profile"]})

        return {
            "type": doc["type"],
            "query": query,
            "choices": doc["offer_messages"],
            "gold": doc["answer"],
        }

    def doc_to_text(self, doc):
        return doc["query"]
