import json
import os

import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        path = os.path.join(os.path.dirname(__file__), "classes.json")
        with open(path, encoding="utf-8") as fp:
            self.classes = np.array(json.load(fp), dtype=np.int64)

    def execute(self, requests):
        responses = []
        for request in requests:
            probs = pb_utils.get_input_tensor_by_name(request, "PROBS").as_numpy()
            idx = probs.argmax(axis=1)
            cat = self.classes[idx].astype(np.int64).reshape(-1, 1)
            responses.append(pb_utils.InferenceResponse([pb_utils.Tensor("CATEGORY_IND", cat)]))
        return responses
