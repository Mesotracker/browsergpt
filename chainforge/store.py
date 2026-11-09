"""Model registry: runtime models plus the bundled starter model."""

from __future__ import annotations

import re
import shutil
import time
from pathlib import Path

from . import cgpt


def safe_name(name: str) -> str:
    base = name.strip()
    if base.lower().endswith(".cgpt"):
        base = base[:-5]
    base = re.sub(r"[^A-Za-z0-9._-]+", "-", base).strip(".-_")
    return (base or "model")[:64]


class ModelStore:
    def __init__(self, data_dir: Path, bundled_dir: Path | None = None) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.bundled_dir = Path(bundled_dir) if bundled_dir else None

    def _dirs(self) -> list[tuple[Path, bool]]:
        dirs = [(self.data_dir, False)]
        if self.bundled_dir is not None:
            dirs.append((self.bundled_dir, True))
        return dirs

    def list_models(self) -> list[dict]:
        items: list[dict] = []
        names: set[str] = set()
        for directory, bundled in self._dirs():
            if not directory.is_dir():
                continue
            for path in sorted(directory.glob("*.cgpt")):
                name = path.stem
                if name in names:
                    continue
                try:
                    manifest = cgpt.read_manifest(path)
                except Exception:
                    continue
                names.add(name)
                training = {}
                try:
                    training = cgpt.inspect(path).get("training", {})
                except Exception:
                    training = {}
                items.append(
                    {
                        "name": name,
                        "id": manifest.get("id"),
                        "params": manifest.get("params", 0),
                        "arch": manifest.get("arch", {}),
                        "created": manifest.get("created"),
                        "lineage": manifest.get("lineage", {}),
                        "steps": training.get("steps"),
                        "final_loss": training.get("final_loss"),
                        "size": path.stat().st_size,
                        "bundled": bundled,
                    }
                )
        items.sort(key=lambda m: (str(m.get("created") or ""), m["name"]), reverse=True)
        return items

    def resolve(self, name: str) -> Path:
        clean = safe_name(name)
        for directory, _bundled in self._dirs():
            candidate = directory / f"{clean}.cgpt"
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"model {name!r} was not found")

    def exists(self, name: str) -> bool:
        try:
            self.resolve(name)
            return True
        except FileNotFoundError:
            return False

    def _unique_name(self, name: str) -> str:
        candidate = safe_name(name)
        if not (self.data_dir / f"{candidate}.cgpt").exists():
            return candidate
        stamp = time.strftime("%Y%m%d-%H%M%S")
        candidate = f"{candidate}-{stamp}"
        if not (self.data_dir / f"{candidate}.cgpt").exists():
            return candidate
        for index in range(2, 1000):
            numbered = f"{candidate}-{index}"
            if not (self.data_dir / f"{numbered}.cgpt").exists():
                return numbered
        raise ValueError("could not find a free model name")

    def add_file(self, source: Path, name: str | None = None) -> dict:
        source = Path(source)
        manifest = cgpt.read_manifest(source)
        chosen = self._unique_name(name or str(manifest.get("name") or source.stem))
        destination = self.data_dir / f"{chosen}.cgpt"
        if source.resolve() == destination.resolve():
            return {**manifest, "name": chosen}
        shutil.copyfile(source, destination)
        return {**manifest, "name": chosen}

    def delete(self, name: str) -> None:
        clean = safe_name(name)
        path = self.data_dir / f"{clean}.cgpt"
        if not path.is_file():
            raise FileNotFoundError(f"model {name!r} is not deletable or was not found")
        path.unlink()
