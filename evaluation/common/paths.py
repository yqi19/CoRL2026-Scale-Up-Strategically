"""Put the ManiSkill fork and the data-collection code on ``sys.path``.

The evaluation scripts import three things that live outside ``evaluation/``:

* ``mani_skill``  (the fork with ``VerbObjectColor-v1``; ``pip install -e maniskill``)
* ``collection_strategy.lib.{training_vocab,pairwise_task_language}``
* ``scripts.run_verb_color_shape_motion_planning.get_env_id_and_color``

The last two are top-level folders of a source checkout, not installed
packages, so their parent directory must be on ``sys.path``. Override the
defaults with ``MANISKILL_ROOT`` / ``DATA_COLLECTION_ROOT``.
"""
from __future__ import annotations

import os
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
EVAL_ROOT = REPO_ROOT / "evaluation"


def setup_paths() -> None:
    roots = [
        os.environ.get("DATA_COLLECTION_ROOT", str(REPO_ROOT / "data_collection")),
        os.environ.get("MANISKILL_ROOT", str(REPO_ROOT / "maniskill")),
    ]
    # Inserted in reverse so MANISKILL_ROOT ends up first on sys.path.
    for root in roots:
        if root and pathlib.Path(root).is_dir():
            if root in sys.path:
                sys.path.remove(root)
            sys.path.insert(0, root)
