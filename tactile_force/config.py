"""YAML config loading with dotted command-line overrides (e.g. train.epochs=5)."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    """dict with attribute access; nested dicts become Config."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def _wrap(obj: Any) -> Any:
    if isinstance(obj, dict):
        return Config({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj


def _apply_override(cfg: dict, dotted: str) -> None:
    if "=" not in dotted:
        raise ValueError(f"Override must look like key.sub=value, got {dotted!r}")
    key, raw = dotted.split("=", 1)
    value = yaml.safe_load(raw)  # parses numbers, bools, lists
    node = cfg
    parts = key.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _read(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    base = raw.pop("base", None)  # `base: default.yaml` -> inherit, then override
    return _deep_merge(_read(path.parent / base), raw) if base else raw


def load_config(path: str | Path | list, overrides: list[str] | None = None) -> Config:
    """Load one config, or several merged left to right (later files win)."""
    paths = path if isinstance(path, (list, tuple)) else [path]
    raw: dict = {}
    for p in paths:
        raw = _deep_merge(raw, _read(Path(p)))
    for item in overrides or []:
        _apply_override(raw, item)
    if isinstance(raw.get("work_dir"), str):  # e.g. outputs/unseen_shapes/{arch}
        raw["work_dir"] = raw["work_dir"].format(arch=raw.get("model", {}).get("arch", "model"))
    return _wrap(raw)


def save_config(cfg: dict, path: str | Path) -> None:
    def plain(o):
        if isinstance(o, dict):
            return {k: plain(v) for k, v in o.items()}
        if isinstance(o, list):
            return [plain(v) for v in o]
        return o

    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(plain(cfg), fh, sort_keys=False)
