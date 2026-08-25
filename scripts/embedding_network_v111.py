"""Entity-embedding network for control-success probability.

Every model in the ensemble so far is a scikit-learn tree or a linear model, and
each learns about a pitcher only through scalar summaries: target encodings, as-of
rates, and the in-season reconstruction. This network instead gives every pitcher,
batter and team its own learned vector, so pitcher identity can interact with the
count, the runners and the inning directly rather than through a single rate.

V48 tried a scikit-learn MLP and it was rejected, but that model had no embeddings,
no in-season reconstruction, and ran before the blend was reweighted. The function
class here is genuinely different from anything in the ensemble, which is the
property that made the logistic (V17, +3.1) and the joint-feature models
(V25, +5.9) worth their weight.

Deliberate choices
------------------
Inference is pinned to CPU. `docs/05` warns that MPS and CUDA differ in
floating-point detail, and the evaluation server has an NVIDIA L4 while development
happens on Apple silicon; scoring on CPU removes that discrepancy entirely and is
fast enough for 245,789 rows through a network this small.

Only the weights are saved, never a pickled module, and the architecture is
reconstructed from a plain dict. That keeps the artifact independent of both torch
internals and numpy's pickle format.

Unseen identifiers map to a dedicated index rather than a random row, so a 2025
debutant gets the learned "unknown" vector instead of another player's.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

SEED = 42
# Every non-numeric column in the Form feature set has to appear here, otherwise it
# would be fed to the numeric branch and fail to cast. `assert_numeric` enforces it.
EMBEDDING_SPECS = [
    ("pitcher_id", 24),
    ("batter_id", 24),
    ("pitcher_team_id", 4),
    ("batter_team_id", 4),
    ("count_state", 6),
    ("base_state", 4),
    ("hand_matchup", 4),
    ("pitcher_hand", 2),
    ("batter_hand", 2),
    ("top_bottom", 2),
    ("game_type", 3),
]


class ControlNet(nn.Module):
    def __init__(self, cardinalities, numeric_width, hidden=(256, 128), dropout=0.15):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(cardinality, width) for cardinality, width in cardinalities
        ])
        embedded_width = sum(width for _, width in cardinalities)
        self.numeric_norm = nn.BatchNorm1d(numeric_width)
        layers, previous = [], embedded_width + numeric_width
        for size in hidden:
            layers += [nn.Linear(previous, size), nn.BatchNorm1d(size),
                       nn.ReLU(), nn.Dropout(dropout)]
            previous = size
        layers.append(nn.Linear(previous, 1))
        self.body = nn.Sequential(*layers)

    def forward(self, categorical, numeric):
        parts = [embedding(categorical[:, index])
                 for index, embedding in enumerate(self.embeddings)]
        parts.append(self.numeric_norm(numeric))
        return self.body(torch.cat(parts, dim=1)).squeeze(1)


def build_vocabularies(frame, specs=EMBEDDING_SPECS):
    """Map each identifier to a contiguous index, reserving 0 for unseen values."""
    vocabularies = {}
    for column, _ in specs:
        values = pd.Index(frame[column].astype(str).unique()).sort_values()
        vocabularies[column] = {value: index + 1 for index, value in enumerate(values)}
    return vocabularies


def encode_categorical(frame, vocabularies, specs=EMBEDDING_SPECS):
    columns = []
    for column, _ in specs:
        mapping = vocabularies[column]
        codes = frame[column].astype(str).map(mapping).fillna(0).to_numpy(dtype=np.int64)
        columns.append(codes)
    return np.column_stack(columns)


def cardinalities(vocabularies, specs=EMBEDDING_SPECS):
    return [(len(vocabularies[column]) + 1, width)
            for column, width in specs]


def assert_numeric(frame, columns):
    """Fail loudly rather than let a string column reach the numeric branch."""
    offenders = [c for c in columns
                 if not pd.api.types.is_numeric_dtype(frame[c])]
    if offenders:
        raise ValueError(
            f"non-numeric columns routed to the numeric branch: {offenders}. "
            "Add them to EMBEDDING_SPECS."
        )


def numeric_statistics(matrix):
    median = np.nanmedian(matrix, axis=0)
    filled = np.where(np.isnan(matrix), median, matrix)
    scale = filled.std(axis=0)
    scale[scale < 1e-8] = 1.0
    return {"median": median, "mean": filled.mean(axis=0), "scale": scale}


def encode_numeric(matrix, statistics):
    filled = np.where(np.isnan(matrix), statistics["median"], matrix)
    return ((filled - statistics["mean"]) / statistics["scale"]).astype(np.float32)


def train(categorical, numeric, target, cardinality_spec, *, epochs=6,
          batch_size=8192, learning_rate=2e-3, weight_decay=1e-5, seed=SEED,
          hidden=(256, 128), dropout=0.15, verbose=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = ControlNet(cardinality_spec, numeric.shape[1],
                       hidden=hidden, dropout=dropout)
    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                                  weight_decay=weight_decay)
    loss_function = nn.BCEWithLogitsLoss()
    categorical_tensor = torch.from_numpy(categorical)
    numeric_tensor = torch.from_numpy(numeric)
    target_tensor = torch.from_numpy(target.astype(np.float32))
    count = len(target)
    steps = int(np.ceil(count / batch_size)) * epochs
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=learning_rate, total_steps=steps)
    generator = np.random.default_rng(seed)
    model.train()
    for epoch in range(epochs):
        order = generator.permutation(count)
        total = 0.0
        for start in range(0, count, batch_size):
            index = order[start:start + batch_size]
            if len(index) < 2:
                continue
            optimiser.zero_grad()
            logits = model(categorical_tensor[index], numeric_tensor[index])
            loss = loss_function(logits, target_tensor[index])
            loss.backward()
            optimiser.step()
            schedule.step()
            total += float(loss) * len(index)
        if verbose:
            print(f"    epoch {epoch + 1}/{epochs} loss {total / count:.6f}", flush=True)
    model.eval()
    return model


def predict(model, categorical, numeric, batch_size=65536):
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(categorical), batch_size):
            stop = start + batch_size
            logits = model(torch.from_numpy(categorical[start:stop]),
                           torch.from_numpy(numeric[start:stop]))
            outputs.append(torch.sigmoid(logits).numpy())
    return np.concatenate(outputs).astype(np.float64)


def state_bundle(model, vocabularies, statistics, numeric_columns, cardinality_spec,
                 hidden=(256, 128), dropout=0.15):
    """Plain-dict artifact: weights plus everything needed to rebuild the module."""
    return {
        "state_dict": {key: value.cpu().numpy()
                       for key, value in model.state_dict().items()},
        "vocabularies": vocabularies,
        "numeric_statistics": {key: np.asarray(value)
                               for key, value in statistics.items()},
        "numeric_columns": list(numeric_columns),
        "cardinalities": [list(item) for item in cardinality_spec],
        "embedding_specs": [list(item) for item in EMBEDDING_SPECS],
        "hidden": list(hidden),
        "dropout": dropout,
        "seed": SEED,
    }


def restore(bundle):
    model = ControlNet(
        [tuple(item) for item in bundle["cardinalities"]],
        len(bundle["numeric_columns"]),
        hidden=tuple(bundle.get("hidden", (256, 128))),
        dropout=bundle.get("dropout", 0.15),
    )
    model.load_state_dict(
        {key: torch.from_numpy(np.asarray(value))
         for key, value in bundle["state_dict"].items()})
    model.eval()
    return model
