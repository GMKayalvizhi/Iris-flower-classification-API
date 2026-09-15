import math
import numpy as np
from app.state import ml_models
from app.models.schemas import IrisInput

SPECIES_MAP = {
    0: "setosa",
    1: "versicolor",
    2: "virginica"
}

def run_inference(model, features: np.ndarray):
    """
    Run inference on a batch of feature rows in a single vectorized call.

    Task 11 investigation -- is it better to call model.predict() once on
    the whole batch, or in a loop?

    Always once on the whole batch. scikit-learn estimators (this
    RandomForestClassifier included) are built on NumPy and are
    vectorized internally: a single call on an (n_rows, 4) array runs
    every row's tree traversal within one pass of compiled code. Calling
    .predict() n times in a Python loop instead pays Python-level
    function-call overhead AND repeats internal setup work n times --
    and that cost scales with batch size, so it matters more, not less,
    as batches get bigger. This same function is used by both /predict
    (a 1-row array) and /predict-batch (an n-row array) for exactly this
    reason -- there should only be one place in the code that calls
    .predict(), so both endpoints always benefit from this automatically.

    Task 14 addition: this now always computes and returns the FULL
    probability breakdown across all classes, not just the winning
    class's confidence. v1 routes only use "species"/"confidence" and
    ignore "probabilities" -- v2 uses all three. This means v2 never
    has to call model.predict_proba() a second time; the work is
    already done once, here, shared by both API versions. Widening
    what this helper returns is safe for v1 -- v1's PredictionOutput
    schema simply doesn't include the extra data, so nothing about
    v1's actual HTTP response changes.

    Task 18 addition: also computes Shannon entropy (bits) per row,
    shared by both versions for the same reason as everything else
    here -- one place that touches model.predict_proba() output.

    Returns a list of dicts, one per input row, in the same order as
    the input array:
        {"species": str, "confidence": float, "probabilities": {species: float, ...}, "entropy_bits": float}
    """
    predictions = model.predict(features)
    probabilities_matrix = model.predict_proba(features)
 
    results = []
    for pred, probs in zip(predictions, probabilities_matrix):
        species_name = SPECIES_MAP[int(pred)]
        confidence = float(probs[pred])
        probability_breakdown = {
            SPECIES_MAP[i]: float(p) for i, p in enumerate(probs)
        }
        # Shannon entropy in bits. Skipping p == 0 is exact, not an
        # approximation -- log2(0) is undefined, but a zero-probability
        # class's true contribution to entropy is 0 in the limit anyway.
        entropy_bits = -sum(p * math.log2(p) for p in probs if p > 0)

        results.append({
            "species": species_name,
            "confidence": confidence,
            "probabilities": probability_breakdown,
            "entropy_bits": entropy_bits,
        })
    return results
 
 
def to_feature_array(inputs: list[IrisInput]) -> np.ndarray:
    return np.array([
        [item.sepal_length, item.sepal_width, item.petal_length, item.petal_width]
        for item in inputs
    ])


def get_model_version() -> str:
    """
    Single place that reads the currently-loaded model's version.

    Both /predict and /predict-batch need this. Routing both endpoints
    through one function means there's only one place to fix if the
    lookup logic ever needs to change.

    Raises a clear, explicit error if model_info was never loaded,
    rather than letting a bare KeyError surface with no context.
    """

    if "model_info" not in ml_models:
        raise RuntimeError("model_info was not loaded at startup")
    return ml_models["model_info"]["model_version"]