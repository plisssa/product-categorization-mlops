"""Конвертация моделей в ONNX и подготовка репозитория Triton (паттерн hw7).

Делает две вещи:
1. ``export_text_encoder_onnx`` — выгружает E5 c mean-pooling+L2-norm в ONNX
   (вход: INPUT_IDS, ATTENTION_MASK -> выход: EMBEDDING).
2. ``convert_head_to_onnx`` — конвертирует обученную sklearn-голову в ONNX
   (вход: EMBEDDING -> выход: PROBS) и пишет classes.json для postprocess.

Запуск (на машине с extra embeddings):
  python triton/prepare_models.py \
      --head artifacts/e5_logreg/models/embedding_head.joblib \
      --text-model intfloat/multilingual-e5-large --dim 1024
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).parent / "model_repository"


def _rename_io(model, in_map: dict[str, str], out_map: dict[str, str]):
    """Переименовать входы/выходы ONNX-графа и все ссылки на них в узлах."""
    rename = {**in_map, **out_map}
    for vi in list(model.graph.input) + list(model.graph.output):
        if vi.name in rename:
            vi.name = rename[vi.name]
    for node in model.graph.node:
        node.input[:] = [rename.get(x, x) for x in node.input]
        node.output[:] = [rename.get(x, x) for x in node.output]
    return model


def convert_head_to_onnx(head_path: Path, dim: int, out_path: Path, classes_out: Path) -> None:
    import joblib
    import onnx
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    bundle = joblib.load(head_path)
    head = bundle["head"] if isinstance(bundle, dict) else bundle
    classes = list(bundle["classes"]) if isinstance(bundle, dict) else list(head.classes_)

    onx = convert_sklearn(
        head,
        initial_types=[("input", FloatTensorType([None, dim]))],
        options={id(head): {"zipmap": False}},
        target_opset=17,
    )
    # выходы skl2onnx: 'label', 'probabilities' -> приводим к контракту Triton
    out_names = [o.name for o in onx.graph.output]
    prob_name = next((n for n in out_names if "prob" in n.lower()), out_names[-1])
    onx = _rename_io(onx, {"input": "EMBEDDING"}, {prob_name: "PROBS"})
    # оставляем только PROBS на выходе
    keep = [o for o in onx.graph.output if o.name == "PROBS"]
    del onx.graph.output[:]
    onx.graph.output.extend(keep)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(onx, str(out_path))
    classes_out.parent.mkdir(parents=True, exist_ok=True)
    classes_out.write_text(json.dumps([int(c) for c in classes]), encoding="utf-8")
    print(f"Head ONNX -> {out_path}; classes -> {classes_out}")


def export_text_encoder_onnx(model_name: str, out_path: Path, max_len: int = 192) -> None:
    import torch
    from torch import nn
    from transformers import AutoModel

    class E5Pooled(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base

        def forward(self, input_ids, attention_mask):
            out = self.base(input_ids=input_ids, attention_mask=attention_mask)
            mask = attention_mask.unsqueeze(-1).float()
            emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            return torch.nn.functional.normalize(emb, dim=-1)

    model = E5Pooled(AutoModel.from_pretrained(model_name)).eval()
    dummy_ids = torch.ones(1, max_len, dtype=torch.long)
    dummy_mask = torch.ones(1, max_len, dtype=torch.long)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (dummy_ids, dummy_mask),
        str(out_path),
        input_names=["INPUT_IDS", "ATTENTION_MASK"],
        output_names=["EMBEDDING"],
        dynamic_axes={"INPUT_IDS": {0: "batch"}, "ATTENTION_MASK": {0: "batch"}, "EMBEDDING": {0: "batch"}},
        opset_version=17,
    )
    print(f"Text encoder ONNX -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--head", type=Path, required=True, help="embedding_head.joblib")
    parser.add_argument("--dim", type=int, default=1024, help="embedding dim (E5-large=1024)")
    parser.add_argument("--text-model", default="intfloat/multilingual-e5-large")
    parser.add_argument("--skip-encoder", action="store_true")
    args = parser.parse_args()

    convert_head_to_onnx(
        args.head,
        args.dim,
        REPO / "category_head" / "1" / "model.onnx",
        REPO / "postprocess" / "1" / "classes.json",
    )
    if not args.skip_encoder:
        export_text_encoder_onnx(args.text_model, REPO / "text_encoder" / "1" / "model.onnx")


if __name__ == "__main__":
    main()
