import os

import numpy as np
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer


class TritonPythonModel:
    def initialize(self, args):
        model = os.environ.get("E5_MODEL", "intfloat/multilingual-e5-large")
        self.tok = AutoTokenizer.from_pretrained(model)
        self.maxlen = 192

    def execute(self, requests):
        responses = []
        for request in requests:
            arr = pb_utils.get_input_tensor_by_name(request, "TEXT").as_numpy()
            texts = ["query: " + (t[0].decode("utf-8") if isinstance(t[0], bytes) else str(t[0])) for t in arr]
            enc = self.tok(texts, padding="max_length", truncation=True, max_length=self.maxlen, return_tensors="np")
            ids = pb_utils.Tensor("INPUT_IDS", enc["input_ids"].astype(np.int64))
            mask = pb_utils.Tensor("ATTENTION_MASK", enc["attention_mask"].astype(np.int64))
            responses.append(pb_utils.InferenceResponse([ids, mask]))
        return responses
