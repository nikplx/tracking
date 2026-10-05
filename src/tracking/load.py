import json
from pathlib import Path
from typing import Optional

import pandas as pd

from tracking import normalize_doc


def load_mongo_data(collection, query: Optional[dict] = None):
    """Fetch a Mongo collection and return a normalized ``pandas.DataFrame``."""
    import pandas as pd

    docs = list(collection.find(query or {}))
    if not docs:
        return pd.DataFrame()
    return pd.DataFrame([normalize_doc(d) for d in docs])

def load_json_data(path: Path) -> pd.DataFrame:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # tolerate a half-written last line from a live run
    if not rows:
        raise SystemExit(f"{path} has no complete JSON lines yet.")

    df = pd.json_normalize(rows, sep="_")

    # pygwalker needs scalar cells; stringify anything that came through as
    # a list or dict (e.g. a non-empty `checkpoints` array).
    for col in df.columns:
        if df[col].map(lambda v: isinstance(v, (list, dict))).any():
            df[col] = df[col].map(lambda v: json.dumps(v, default=str) if isinstance(v, (list, dict)) else v)

    # Best-effort: parse anything that looks like a timestamp column.
    for col in df.columns:
        if "timestamp" in col.lower():
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().mean() > 0.9:
                df[col] = parsed

    return df

