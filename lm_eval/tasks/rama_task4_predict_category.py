"""

공고/프로필의 항목 분류
관련문서: https://dramancompany.atlassian.net/wiki/spaces/BDCAI/pages/28241461371/RAMA+PET+Benchmark+Task+4

"""
from lm_eval.base import MultipleChoiceTask
import json


class PredictCategory(MultipleChoiceTask):
    QUERY = """
instruction:
주어진 공고/이력서의 내용 중, 주어진 target 항목의 내용을 담고 있는 적절한 지문을 선택하세요.
 
target:
{target}

공고/프로필:
{body}

정답:    
"""

    VERSION = 0.1
    DATASET_PATH = "/raid/ailab-workspace/gyholee/project/rama_pet_benchmark/llm_pet/benchmark/data/benchmark_predict_category.json"
    DATASET_NAME = None

    def __init__(self):
        self.dataset = json.load(open(self.DATASET_PATH))

        # train = pd.DataFrame(dataset["train"])
        # test = pd.DataFrame(dataset["test"])
        #
        # self.dataset = concatenate_datasets(
        #     [Dataset.from_pandas(train, split="train"), Dataset.from_pandas(test, split="test")]
        # )

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
        query = self.QUERY.format_map({"target": doc["target"], "body": doc["body"]})

        return {
            "query": query,
            "choices": doc["candidates"],
            "gold": doc["answer"],
        }

    def doc_to_text(self, doc):
        return doc["query"]


if __name__ == "__main__":
    pc = PredictCategory(
        "/raid/ailab-workspace/gyholee/project/rama_pet_benchmark/llm_pet/benchmark/data/benchmark_predict_category.json"
    )
