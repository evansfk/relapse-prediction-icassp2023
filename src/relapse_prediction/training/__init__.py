"""SSL pretraining and downstream evaluation runners."""

from .evaluator import (
    PersonIDCVSummary,
    SurvivalBenchmarkResult,
    build_default_survival_models,
    compute_ablation_relative_change,
    format_person_id_table,
    person_id_summaries_to_frame,
    run_person_id_evaluation,
    run_survival_benchmark,
)
from .ssl_trainer import SSLPretrainResult, extract_embeddings, pretrain_mixing_up, pretrain_tfc, pretrain_tstcc

__all__ = [
    "PersonIDCVSummary",
    "SSLPretrainResult",
    "SurvivalBenchmarkResult",
    "build_default_survival_models",
    "compute_ablation_relative_change",
    "extract_embeddings",
    "format_person_id_table",
    "person_id_summaries_to_frame",
    "pretrain_mixing_up",
    "pretrain_tfc",
    "pretrain_tstcc",
    "run_person_id_evaluation",
    "run_survival_benchmark",
]
