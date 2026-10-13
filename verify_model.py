"""Verify the completed Hub snapshot against its pinned file manifest."""
import os
import json
from pathlib import Path
from huggingface_hub import HfApi

ROOT = Path(os.environ.get("MACQWEN_ROOT", str(Path.home() / "llm" / "macqwen")))
REPO = "TensorFold/Qwen3.8-Flash-Next-MLX-4bit-MTP"
REVISION = "2b170fa6309d5d1ee380b35636075fac7945f286"


def main():
    info = HfApi().model_info(REPO, revision=REVISION, files_metadata=True)
    model = ROOT / "models/qwen38-flash-next-4bit"
    records = []
    for sibling in info.siblings:
        if not sibling.rfilename.endswith(".safetensors"):
            continue
        file = model / sibling.rfilename
        if not file.is_file() or file.stat().st_size != sibling.size:
            raise SystemExit(f"incomplete model file: {sibling.rfilename}")
        records.append({"file":sibling.rfilename, "bytes":sibling.size,
                        "hub_sha256":getattr(sibling.lfs, "sha256", None)})
    if len(records) != 22 or sum(row["bytes"] for row in records) != 113209682735:
        raise SystemExit("pinned model manifest differs from the reviewed snapshot")
    index = json.loads((model / "model.safetensors.index.json").read_text())
    if set(index["weight_map"].values()) - {row["file"] for row in records}:
        raise SystemExit("the model index references unavailable shards")
    result = {"repo":REPO, "revision":info.sha, "verification":"Hub download plus file-size and index checks",
              "weights_bytes":sum(row["bytes"] for row in records),
              "tensors":len(index["weight_map"]), "shards":records}
    (ROOT / "config/model-manifest.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"model_revision":info.sha, "shards":len(records),
                      "weights_bytes":result["weights_bytes"], "tensors":result["tensors"]}))


if __name__ == "__main__":
    main()
