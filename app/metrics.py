"""
Custom Prometheus metrics for the Iris API.

prometheus-fastapi-instrumentator wires up the generic HTTP metrics
(request count, latency, in-progress requests) automatically -- see
Instrumentator().instrument(app).expose(app) in main.py. Those metrics
know nothing about what this API actually does.

This module holds the one metric that's specific to iris classification:
the Shannon entropy of each prediction's probability distribution. A
class-count metric would catch distribution drift (predicting one
species far more than usual); entropy catches something a class count
can't -- individual predictions becoming less decisive even while the
overall class mix looks perfectly normal. That's a meaningfully earlier
signal of distribution shift for a model like this one, where
versicolor/virginica genuinely overlap and setosa is easily separable.

Defined in its own module (not inline in routers/v1.py or v2.py) so
both routers can import and record against the same histogram.
"""

from prometheus_client import Histogram

# Max entropy for 3 classes is log2(3) ~= 1.585 bits (total uncertainty,
# probabilities evenly split three ways). 0 bits means the model is
# completely decisive. Buckets are chosen to resolve that whole range,
# with extra resolution near the top where "the model is confused"
# actually lives.
PREDICTION_ENTROPY_BITS = Histogram(
    "iris_prediction_entropy_bits",
    "Shannon entropy (bits) of the predicted probability distribution -- "
    "near 0 means the model is decisive, approaching log2(n_classes) "
    "means the input sits close to a decision boundary",
    ["api_version"],
    buckets=(0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.585),
)