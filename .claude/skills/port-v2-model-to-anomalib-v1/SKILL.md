---
name: port-v2-model-to-anomalib-v1
description: Port an anomaly detection model from anomalib's current main branch (v2.x `AnomalibModule` API) to this anomalib<2 (v1.x, `AnomalyModule` API) branch. Use when asked to "add", "port", or "backport" a model (e.g. Dinomaly, UniNet, GLASS, SuperSimpleNet) to anomalib<2, or when working on a `feature/*-v1` branch that mirrors a v2 model.
---

# Porting a v2 anomalib model to anomalib<2 (v1.x)

This repo's `main` branch is anomalib v2.x. Models added there use a newer Lightning
API (`AnomalibModule`, dataclass batches, `PreProcessor`/`PostProcessor`/`Evaluator`/
`Visualizer`). This branch (and other `feature/*-v1` branches like it) targets
anomalib v1.x (tagged around `v1.2.0`), which predates that redesign and uses the
older `AnomalyModule` API. New models land on `main` first; this skill is for
bringing one of those models back to the v1.x API on a branch like this one.

It was written while porting **Dinomaly** (`src/anomalib/models/image/dinomaly/`) —
use that port as the reference example throughout.

## 0. Find or create the target branch

The v1.x porting branches are named `feature/<model>-v1` and are typically created
from a v1.x tag (e.g. `git branch feature/foo-v1 v1.2.0`, check with
`git reflog show feature/<model>-v1` — it should say `branch: Created from v1.2.0`
or similar). If a `main` checkout and the target branch both need to be inspected at
once, use a git worktree instead of switching branches back and forth:

```bash
git worktree add /path/to/scratch/anomalib-v1 feature/<model>-v1
git worktree prune  # if you get "already used by worktree" for a path that no longer exists
```

Keep `main` (source of truth for the v2 implementation) and the worktree (v1.x
target) open side by side.

## 1. Read the v2 implementation on `main` first

Read every file under `src/anomalib/models/image/<model>/` on `main`:
`__init__.py`, `lightning_model.py`, `torch_model.py`, `README.md`, and everything
under `components/`. For each file, classify it:

- **Framework-agnostic (pure PyTorch)** — no `anomalib.*` imports beyond generic
  utilities. These copy over **verbatim**, no changes needed. In Dinomaly this was
  `components/layers.py`, `loss.py`, `optimizer.py`, `vision_transformer.py`.
- **Anomalib-dependent but stable API** — imports things like
  `anomalib.models.components.GaussianBlur2d` or
  `anomalib.data.utils.DownloadInfo` / `DownloadProgressBar`. Check these exist in
  the v1.x tree (`grep` for the class in `src/anomalib/models/components/__init__.py`
  or `src/anomalib/data/utils/__init__.py`) — they usually do, since these are old,
  stable utilities. Copy verbatim once confirmed. In Dinomaly this was
  `components/dinov2_loader.py`.
- **Lightning-module glue** — `lightning_model.py`, and any `torch_model.py` code
  that imports `anomalib.data.InferenceBatch` or similar v2-only dataclasses. These
  need adapting; see §2.

## 2. API differences to adapt (v2 `AnomalibModule` → v1 `AnomalyModule`)

| v2 (`main`) | v1.x (this branch) |
|---|---|
| `from anomalib.models.components import AnomalibModule` | `from anomalib.models.components import AnomalyModule` |
| `__init__(self, ..., pre_processor=True, post_processor=True, evaluator=True, visualizer=True)`, forwarded to `super().__init__(...)` | `__init__(self, ...)` → `super().__init__()` takes **no args**. Drop the pre/post-processor/evaluator/visualizer params entirely — v1 has no equivalent constructor concept. |
| `batch: Batch` dataclass, fields via `batch.image`, updated via `batch.update(pred_score=..., anomaly_map=...)` | `batch: dict[str, str \| torch.Tensor]`, fields via `batch["image"]`, updated via `batch["pred_scores"] = ...` / `batch["anomaly_maps"] = ...`. **Keys are plural** (`pred_scores`, `anomaly_maps`), unlike the v2 dataclass field names. |
| Torch model returns `anomalib.data.InferenceBatch(pred_score=..., anomaly_map=...)` at inference | Torch model returns a plain `tuple[pred_score, anomaly_map]` (or whatever shape matches how the lightning module unpacks it — grep sibling models like `dfm`, `winclip` for the `batch["pred_scores"], batch["anomaly_maps"] = self.model(...)` pattern). |
| `@classmethod configure_pre_processor(cls, image_size=None) -> PreProcessor` returning a `PreProcessor(transform=Compose(...))` | Override `configure_transforms(self, image_size=None) -> Transform` (defined on the `AnomalyModule` base in `models/components/base/anomaly_module.py`) returning the `Compose(...)` directly. **If the method body doesn't use `self`, make it `@staticmethod`** — that's the convention every other v1 model follows (`grep -rn "def configure_transforms" src/anomalib/models/`), and ruff (`PLR6301`) will flag it otherwise. |
| `PostProcessor`, `Evaluator`, `Visualizer` classes | Don't exist in v1. Metrics, thresholding and score-from-anomaly-map derivation happen automatically via Engine callbacks (`src/anomalib/callbacks/post_processor.py`, `thresholding.py`, `metrics.py`) — you don't wire anything up in the lightning module. **But**: the default derivation of `pred_scores` from `anomaly_maps` (when you don't set it yourself) is a plain per-image max — see `_PostProcessorCallback._post_process` in `callbacks/post_processor.py`. If the v2 model uses a smarter aggregation (e.g. Dinomaly's top-1%-mean), compute `pred_scores` explicitly in `validation_step` instead of relying on the default. |
| Model can be built directly in `__init__`, or deferred via `_setup()` if it needs `self.input_size` (resolved from the datamodule/transform at trainer setup time) | Same `_setup()` hook exists on the v1 `AnomalyModule` base — use it for backbones whose channel/size config depends on the datamodule. Not needed if the v2 version also builds eagerly in `__init__` (true for Dinomaly). |
| `self.log(...)`, `self.global_step`, `self.trainer.max_epochs/max_steps`, `configure_optimizers()` returning `[optimizer], [scheduler]` | Unchanged — this is plain Lightning `LightningModule` API, identical in both versions. Copy verbatim. |
| `learning_type` / `trainer_arguments` properties | Same shape, same abstract properties on both base classes. Copy verbatim. |

## 3. Registration (this is what makes the model discoverable)

Two files, both alphabetically ordered by class name — **double check alphabetical
order**, it's easy to insert in the wrong spot (`Dfkde` < `Dfm` < `Dinomaly` <
`Draem`, not `Dfkde` < `Dinomaly` < `Dfm`):

1. `src/anomalib/models/image/__init__.py` — add `from .<model> import <Model>` and
   add `"<Model>"` to `__all__`.
2. `src/anomalib/models/__init__.py` — same: add to the `from .image import (...)`
   block and to `__all__`.

That's it — there's no separate model registry. `get_available_models()` and
`get_model("<model>")` in `src/anomalib/models/__init__.py` both work by walking
`AnomalyModule.__subclasses__()` / matching class names, so importing the class is
the entire registration mechanism. It also means the model automatically gets
picked up by `tests/integration/model/test_models.py`, which parametrizes over
`get_available_models()` and runs full fit/test/predict/export cycles — that test
will need network access to download any pretrained weights the model uses.

## 4. Config, docs, README

- `configs/model/<model>.yaml` — jsonargparse CLI config. Copy the shape from a
  similar existing model (e.g. `configs/model/reverse_distillation.yaml` or
  `efficient_ad.yaml` for step-based training):
  ```yaml
  model:
    class_path: anomalib.models.<Model>
    init_args:
      <constructor kwargs>
  metrics:
    pixel:
      - AUROC
  trainer:
    <trainer overrides, e.g. max_steps, gradient_clip_val>
  ```
- `docs/source/markdown/guides/reference/models/image/<model>.md` — an
  `automodule` stub pointing at `lightning_model` and `torch_model` (copy an
  existing one and rename). Register it in
  `docs/source/markdown/guides/reference/models/image/index.md`: one
  `:::{grid-item-card}` entry (alphabetically among the cards) **and** one line in
  the `{toctree}` list at the bottom.
- `README.md` inside the model folder — copy from `main`, but fix the "Usage" CLI
  example: v1.x dataset class names can differ from v2 (e.g. `MVTec`, not
  `MVTecAD`), and if the model trains by `max_steps` rather than epochs, show
  `--trainer.max_steps N` explicitly since v1.x configs don't default to it.

## 5. Dependency version traps

v1.x's `pyproject.toml` often leaves fast-moving deps like `timm` **unconstrained**
(just `"timm"`), because `main` just tracks whatever's latest at CI time. A model
ported from `main` may use APIs that only exist in a *recent* version of that
dependency (e.g. Dinomaly's decoder needs `timm.layers.*` and
`Attention(proj_bias=...)`, which don't exist before `timm==1.0.13` — anything
from the `timm.models.layers` era, like `timm==0.5.4` or `1.0.3`, fails on import
or raises `TypeError: unexpected keyword argument`).

Don't guess — verify empirically and pin a floor with a comment explaining why:

```bash
for v in 1.0.7 1.0.8 1.0.13; do
  pip install -q "timm==$v"
  python -c "
import inspect
from timm.models.vision_transformer import Attention
print('$v', 'proj_bias' in inspect.signature(Attention.__init__).parameters)
"
done
```

```toml
"timm>=1.0.13",  # <model>'s decoder needs timm's `Attention(proj_bias=...)`, added in 1.0.13
```

## 6. Testing without wrecking the sandbox's Python env

The sandbox's default/system `anomalib` install is very likely a *different major
version* (v2.x) from the branch you're porting to — don't try to test by importing
plain `anomalib`, it'll resolve the wrong package. Two options, in order of
preference:

1. **Look for an existing venv/conda env that already has a working, matching
   anomalib install** before creating a new one — e.g. `conda env list` /
   `ls ~/miniconda3/envs`. If one exists, prepend the worktree's `src/` to
   `PYTHONPATH` so your locally-edited package shadows the installed one:
   ```bash
   PYTHONPATH=/path/to/worktree/src <env>/bin/python your_script.py
   ```
   This lets you iterate on the code without any install step.
2. If none exists, create an isolated venv with `uv venv --system-site-packages`
   and `uv pip install --no-deps <missing-package>` for just the one or two
   packages that are actually missing/outdated (e.g. `timm`). **Avoid unconstrained
   `uv pip install <pkg>`** (without `--no-deps`) for anything torch-adjacent — it
   will happily resolve and install a *different* torch/torchvision/CUDA build than
   the one already on the box, which can break unrelated binary extensions that
   were compiled against the original CUDA runtime (this exact thing happened
   during the Dinomaly port: reinstalling `timm` with deps pulled a newer
   torch+cu13 stack, which broke a prebuilt `flash_attn` `.so` that
   `anomalib.metrics` transitively imports via `kornia` — a completely unrelated
   import chain, several layers away from anything touched by the port).

For the actual test, avoid needing real pretrained-weight downloads for a basic
architecture/forward-pass check by monkeypatching the weight loader:

```python
from unittest.mock import patch
with patch.object(SomeWeightLoader, "_load_weights", lambda *a, **k: None):
    from anomalib.models.image.<model> import <Model>
    ...
```

Then separately do **one real** `get_model("<model>")` call (no mocking) to confirm
the actual download URL and `state_dict` loading path work end to end — cheap
insurance that's easy to skip but catches real bugs.

For the Lightning module itself, don't hand-roll a fake `Trainer` — `self.log(...)`
inside `training_step` needs real `Trainer` internals (`trainer.barebones`, the
logger connector, etc.) that a `SimpleNamespace` can't fake convincingly. Instead
spin up a real tiny `lightning.pytorch.Trainer(max_steps=2, limit_val_batches=0,
enable_checkpointing=False, logger=False)` against a two-line dummy `Dataset`
yielding `{"image": torch.randn(...)}` batches, and call `trainer.fit(...)`. This
exercises `configure_optimizers`, `training_step`, `self.log`, and gradient
updates for real, on GPU if available (`torch.cuda.is_available()`), for the cost
of a couple of seconds.

Also sanity-check registration:
```python
from anomalib.models import get_available_models, get_model
assert "<model>" in get_available_models()
```

## 7. Lint

Check the pinned ruff version in `.pre-commit-config.yaml` (`rev: "vX.Y.Z"` under
the `ruff-pre-commit` repo) and run **that exact version**, not whatever's latest —
`uvx ruff@X.Y.Z check <paths>`. Running a newer ruff than the repo pins will report
rules the project doesn't actually enforce (this happened during the Dinomaly
port — `uvx ruff` with no pin flagged ~30 issues from preview rules that don't
apply; pinning to the repo's actual `v0.6.2` cut that to a single real one,
`PLR6301`, fixed by the `@staticmethod` change in §2).

## Sandbox gotcha: git-lfs missing

If `git status`/`checkout`/`worktree add` fails with something like
`git-lfs filter-process: 1: git-lfs: not found` and there's no sudo to install it,
don't get stuck — relax the LFS filter locally (safe as long as you're not editing
LFS-tracked binary assets, which model source code never is):
```bash
git config --global --unset filter.lfs.process
git config --global filter.lfs.required false
git config --global filter.lfs.smudge cat
git config --global filter.lfs.clean cat
```

## Checklist for the next model

- [ ] Worktree/branch for the v1.x target set up
- [ ] Every file under the v2 model's directory read and classified (verbatim /
      needs-adapting)
- [ ] `torch_model.py` + `components/` ported (verbatim where possible)
- [ ] `lightning_model.py` ported per the table in §2
- [ ] `__init__.py` for the model package
- [ ] Registered in both `models/image/__init__.py` and `models/__init__.py`
      (alphabetical!)
- [ ] `configs/model/<model>.yaml` added
- [ ] Docs page added + `index.md` card and toctree entry
- [ ] `README.md` usage line fixed for v1.x dataset/CLI conventions
- [ ] Dependency floors verified empirically, not guessed
- [ ] Smoke test: mocked-weights forward pass + real `get_model()` weight download
      + real 2-step `Trainer.fit()` + `validation_step`
- [ ] `ruff check` / `ruff format --check` with the repo's pinned version
