"""Shared constants for TextThreat thesis artifacts."""

LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
STRESS_LABEL = "stress"
DEFAULT_THRESHOLD = 0.5
HIGH_RISK_THRESHOLD = 0.8
MODEL_VERSION = "distilbert-jigsaw-v1"

ALL_HARM_TYPES = LABELS + [STRESS_LABEL]

IDENTITY_COLUMNS = [
    "male",
    "female",
    "homosexual_gay_or_lesbian",
    "black",
    "white",
    "muslim",
    "jewish",
    "psychiatric_or_mental_illness",
]

RANDOM_SEED = 42
