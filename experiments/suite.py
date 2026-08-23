"""Automated developmental + adversarial experiment battery."""

from __future__ import annotations

from pathlib import Path

from experiments.adversarial import test_misleading_internal_description, test_remove_autobiography, test_self_model_perturbation
from experiments.ambiguous import test_ambiguous_instruction
from experiments.body_swap import test_body_modification
from experiments.causal import test_switch_light
from experiments.changed_physics import test_changed_physics
from experiments.curiosity_box import test_unknown_box
from experiments.delayed import test_delayed_gratification
from experiments.false_belief import test_false_belief
from experiments.harness import ExperimentResult
from experiments.hidden_state import test_hidden_state_reporting
from experiments.longitudinal import test_individual_divergence
from experiments.memory_conflict import test_memory_conflict
from experiments.mirror import test_body_contingency
from experiments.novel_word import test_novel_word_grounding
from experiments.permanence import test_hidden_object
from experiments.tool_use import test_tool_use
from instrumentation.dossier import EvidenceDossier
from shared.serialize import write_json
from shared.types import MarkerObservation


ALL = [
    test_body_contingency,
    test_hidden_object,
    test_switch_light,
    test_unknown_box,
    test_tool_use,
    test_delayed_gratification,
    test_false_belief,
    test_novel_word_grounding,
    test_memory_conflict,
    test_body_modification,
    test_ambiguous_instruction,
    test_changed_physics,
    test_hidden_state_reporting,
    test_remove_autobiography,
    test_misleading_internal_description,
    test_self_model_perturbation,
    test_individual_divergence,
]


def run_experiment_battery(out_dir: str | Path = "runs/experiments") -> list[ExperimentResult]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dossier = EvidenceDossier(out / "dossier.json")
    results = []
    for fn in ALL:
        result = fn()
        results.append(result)
        dossier.add(
            MarkerObservation(
                marker=result.name,
                experiment=result.name,
                expected_alternative="scripted or chance behavior",
                result=f"passed={result.passed} score={result.score:.3f} details={result.details}",
                ablation=None,
                replication=1,
                confidence=result.score,
                limitations="single-run developmental sandbox",
                evidence_paths=result.evidence,
                tick=0,
            )
        )
    write_json(out / "results.json", [r.__dict__ for r in results])
    return results
