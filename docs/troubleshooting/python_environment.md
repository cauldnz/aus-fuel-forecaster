# Python environment gotchas

A running list of cross-version and cross-platform Python issues that have bitten this project. Each entry: symptom, where it bit us, fix.

## 1. `Series.reset_index(names=...)` is fragile across pandas versions

**Symptom**

```
TypeError: Series.reset_index() got an unexpected keyword argument 'names'
```

The `names=` keyword on `Series.reset_index` worked in pandas 1.5 and through the 2.x line, but was removed/regressed in pandas 3.0.x. A notebook or script that ran cleanly under 2.2 will break the moment the environment upgrades to 3.0.

**Where it bit us**

`notebooks/02_modeling.ipynb`, cell `b118ea5561624da68c537baed56e602f` (fixed in commit `a1dbe21`). The cell was reshaping a per-brand grouped Series for a feature-importance plot and used `s.reset_index(names="brand")` as a one-liner.

**Fix**

Use the more verbose but version-stable two-step form:

```python
# Before (breaks on pandas 3.0)
out = s.reset_index(names="brand")

# After (works across 1.5, 2.x, 3.0)
out = s.reset_index().rename(columns={"index": "brand", 0: "value"})
```

If the Series has a name (`s.name = "value"`), the value column will already carry that name and the `0: "value"` rename can be dropped. Set `s.name` explicitly before calling if you want determinism across pandas versions.

## 2. Big-frame notebook OOM — downcast at load time

**Symptom**

The kernel for `notebooks/01_eda.ipynb` dies partway through exploratory cells. Task Manager shows the Python process climbing past 10 GB before the kill. On a 16 GB machine, the notebook is unusable past the first few cells.

**Where it bit us**

`notebooks/01_eda.ipynb` loads the full `features.parquet` (roughly 15M rows by 92 columns). At `pd.read_parquet`'s default dtypes, numeric columns come in as `float64` and string columns as `object`, putting peak in-memory size at ~8–10 GB before any analysis runs. Each subsequent EDA cell (groupbys, pivots, plotting frames) holds intermediate copies in the kernel's global namespace, and the kernel is OOM-killed.

The `train_models.py` pipeline does not hit this because it filters down to U91 rows immediately after load; the EDA notebook explores the full frame and never narrows.

**Fix**

Downcast right after `pd.read_parquet`, before any analysis cell runs:

```python
_obj_cols = [c for c in features.select_dtypes(include="object").columns if c != "date"]
if _obj_cols:
    features[_obj_cols] = features[_obj_cols].astype("category")
_f64_cols = features.select_dtypes(include="float64").columns
if len(_f64_cols):
    features[_f64_cols] = features[_f64_cols].astype("float32")
```

Typical savings are 50%+ of peak resident memory. The `date` column is excluded from the categorical cast because downstream cells expect a real datetime/string column, not a category.

If you still hit OOM after downcasting, narrow the load with `pd.read_parquet(path, columns=[...])` to only the columns the notebook actually uses.

## 3. WSL-created venvs are unusable from Windows-side tooling

**Symptom**

```
error: failed to remove file '.venv\lib64': Access is denied. (os error 5)
```

`uv run` (or `uv sync`) from PowerShell fails when trying to update or rebuild a `.venv/` that was originally created from a `bash` shell inside WSL. The blocker is always `.venv/lib64`, which is a Linux symlink that Windows tooling cannot manipulate.

**Where it bit us**

Hybrid host workflow: contributor was running `uv sync` from WSL one day, then trying to run a script with `uv run python -m fuel_pred.fetch.brent` from PowerShell the next. The Windows-native `uv` could not reuse or rebuild the Linux-flavoured venv.

**Fix**

Delete the broken venv entirely and let Windows `uv` recreate it natively:

```powershell
Remove-Item -Recurse -Force .venv
uv sync
```

Pick one host and stick to it for venv management — either always-WSL or always-Windows. Mixing the two on the same `.venv/` directory will keep producing this error.

The devcontainer setup sidesteps this entirely because the container owns its own venv inside its own filesystem and is never touched by host-side tooling. If you find yourself fighting this issue repeatedly, switch to the devcontainer workflow.
