import abc
from typing import Iterable
import numpy as np
import random
import re
import os
import json
import hashlib
import datasets
from datasets import Dataset, concatenate_datasets
import pandas as pd
from sqlitedict import SqliteDict
from tqdm import tqdm
import torch
import torch.nn.functional as F
from accelerate import find_executable_batch_size

from lm_eval.metrics import mean, weighted_perplexity, weighted_mean, bits_per_byte
from lm_eval import utils
from abc import abstractmethod
import lm_eval.base as base


class MultipleChoiceTask(base.MultipleChoiceTask):
    """기존의 기본 Task 클래스의 확장
    local에 있는 우리 benchmark 파일을 load하여 사용할 수있도록 수정

    """
