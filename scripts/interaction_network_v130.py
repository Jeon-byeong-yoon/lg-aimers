"""Neural factorization machine: multiplicative field interactions, stated explicitly.

The V111 network concatenates its embeddings and hands them to a dense stack. That
learns interactions only implicitly, and a ReLU network approximates products badly --
representing x*y needs many units to fake a curve that one multiplication gives for
free. Every other component in the ensemble is worse still: trees split on one feature
at a time and reach an interaction only by stacking depth.

Control success is close to a product by nature. It is pitcher command against
situation difficulty, and the same pitcher is differently affected by 3-0 than by 0-2.
So this model computes the pairwise products directly. Each field -- pitcher, batter,
teams, count, bases, handedness, the numeric block as a whole -- gets a vector in one
shared latent space, and the second-order term is their elementwise pairwise sum,

    0.5 * ((sum_i e_i)^2 - sum_i e_i^2)

which is every pair <e_i, e_j> for i < j accumulated in linear rather than quadratic
time. That vector, not the raw embeddings, is what the dense stack sees.

The embeddings are not concatenated forward -- that is what makes this a different
function class from V111 rather than V111 with an extra term -- but the *numerics* are.
A first attempt withheld them too, on the reasoning that anything shared with V111
costs diversity. That was wrong, and badly: the ninety-three numeric columns carry most
of the signal in this problem (as-of rates, the in-season reconstruction, the target
encodings), and routing them through a single rank-k field vector left the dense stack
with no way to read them. Standalone skill collapsed to -638 on 2024 against V111's
+616. Diversity is worth nothing from a component that cannot predict, so the
multiplicative prior is an addition to full numeric access, not a replacement for it.

The first-order part is kept as an explicit additive term per field, so the interaction
stack does not spend capacity relearning simple main effects.

Same operational choices as V111, for the same reasons: CPU inference to avoid the
MPS/CUDA discrepancy `docs/05` warns about, plain-dict artifacts with no pickled
module, and index 0 reserved for unseen identifiers so a 2025 debutant gets the learned
unknown vector.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from embedding_network_v111 import (
    EMBEDDING_SPECS, assert_numeric, build_vocabularies, encode_categorical,
    encode_numeric, numeric_statistics,
)

SEED = 42
LATENT = 24
FIELDS = [column for column, _ in EMBEDDING_SPECS]
# Every field must share one latent dimension for the pairwise term to be defined,
# unlike V111 where each column carried its own width.
FIELD_SPECS = [(column, LATENT) for column in FIELDS]


def field_cardinalities(vocabularies):
    return [(len(vocabularies[column]) + 1, LATENT) for column in FIELDS]


class InteractionNet(nn.Module):
    def __init__(self, cardinalities, numeric_width, hidden=(256, 128), dropout=0.15):
        super().__init__()
        # Read the latent width from the spec, never from the module global. The
        # evaluation sweep mutated that global to compare latent sizes, so a restored
        # model would otherwise be built at whatever width was set last rather than the
        # width its own weights were trained at.
        latent = cardinalities[0][1]
        if any(width != latent for _, width in cardinalities):
            raise ValueError(
                "every field must share one latent width for the pairwise term to be "
                f"defined; got {sorted({w for _, w in cardinalities})}")
        self.factors = nn.ModuleList([
            nn.Embedding(cardinality, width) for cardinality, width in cardinalities
        ])
        self.biases = nn.ModuleList([
            nn.Embedding(cardinality, 1) for cardinality, _ in cardinalities
        ])
        for embedding in self.factors:
            nn.init.normal_(embedding.weight, std=0.05)
        for embedding in self.biases:
            nn.init.zeros_(embedding.weight)
        self.numeric_norm = nn.BatchNorm1d(numeric_width)
        # The numeric block enters as one additional field, so its own values also take
        # part in the pairwise term instead of only in the dense stack.
        self.numeric_factor = nn.Linear(numeric_width, latent)
        self.numeric_linear = nn.Linear(numeric_width, 1)
        self.interaction_norm = nn.BatchNorm1d(latent)
        # Interaction vector plus the normalised numerics. Withholding the numerics
        # here starved the model; see the module docstring.
        layers, previous = [], latent + numeric_width
        for size in hidden:
            layers += [nn.Linear(previous, size), nn.BatchNorm1d(size),
                       nn.ReLU(), nn.Dropout(dropout)]
            previous = size
        layers.append(nn.Linear(previous, 1))
        self.body = nn.Sequential(*layers)
        self.intercept = nn.Parameter(torch.zeros(1))

    def forward(self, categorical, numeric):
        normalised = self.numeric_norm(numeric)
        vectors = [embedding(categorical[:, index])
                   for index, embedding in enumerate(self.factors)]
        vectors.append(self.numeric_factor(normalised))
        stacked = torch.stack(vectors, dim=1)
        total = stacked.sum(dim=1)
        interaction = 0.5 * (total * total - (stacked * stacked).sum(dim=1))
        first_order = self.intercept + self.numeric_linear(normalised).squeeze(1)
        for index, embedding in enumerate(self.biases):
            first_order = first_order + embedding(categorical[:, index]).squeeze(1)
        deep = self.body(torch.cat(
            [self.interaction_norm(interaction), normalised], dim=1)).squeeze(1)
        return first_order + deep


def train(categorical, numeric, target, cardinality_spec, *, epochs=8,
          batch_size=8192, learning_rate=2e-3, weight_decay=1e-5, seed=SEED,
          hidden=(128, 64), dropout=0.15, verbose=True):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = InteractionNet(cardinality_spec, numeric.shape[1],
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
        running = 0.0
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
            running += float(loss) * len(index)
        if verbose:
            print(f"    epoch {epoch + 1}/{epochs} loss {running / count:.6f}",
                  flush=True)
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
                 dropout=0.15):
    """Record the hidden widths read off the fitted model, never from a default.

    Passing them in as a defaulted argument let the bundle disagree with the weights it
    was describing: a build recorded (256, 128) for a model trained at (128, 64), and
    `restore` then failed on a size mismatch at load time. Reading them from the module
    makes that class of bug impossible.
    """
    hidden = tuple(layer.out_features for layer in model.body
                   if isinstance(layer, nn.Linear))[:-1]
    return {
        "state_dict": {key: value.cpu().numpy()
                       for key, value in model.state_dict().items()},
        "vocabularies": vocabularies,
        "numeric_statistics": {key: np.asarray(value)
                               for key, value in statistics.items()},
        "numeric_columns": list(numeric_columns),
        "cardinalities": [list(item) for item in cardinality_spec],
        "field_specs": [list(item) for item in FIELD_SPECS],
        "latent": LATENT,
        "hidden": list(hidden),
        "dropout": dropout,
        "seed": SEED,
    }


def restore(bundle):
    model = InteractionNet(
        [tuple(item) for item in bundle["cardinalities"]],
        len(bundle["numeric_columns"]),
        hidden=tuple(bundle["hidden"]),
        dropout=bundle.get("dropout", 0.15),
    )
    model.load_state_dict({key: torch.from_numpy(np.asarray(value))
                           for key, value in bundle["state_dict"].items()})
    model.eval()
    return model


__all__ = [
    "FIELDS", "FIELD_SPECS", "LATENT", "InteractionNet", "assert_numeric",
    "build_vocabularies", "encode_categorical", "encode_numeric",
    "field_cardinalities", "numeric_statistics", "predict", "restore",
    "state_bundle", "train",
]
