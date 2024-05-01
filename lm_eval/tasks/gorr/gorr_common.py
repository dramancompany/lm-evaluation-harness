from bdc.db.s3_helper import S3Helper
import json
import pandas as pd


class GORRUtilsMixin:
    @staticmethod
    def load_benchmark_dataset(path) -> pd.DataFrame:
        content_object = S3Helper().get_objects(path)[0]
        file_content = content_object.get()["Body"].read().decode("utf-8")
        json_content = json.loads(file_content)

        return json_content


# %%
