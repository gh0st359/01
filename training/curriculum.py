"""Developmental curriculum. Progress is capability-gated, not timed."""

from __future__ import annotations

from organism.loop import Organism
from simulation.scenarios import ScenarioName
from simulation.world import SimulatedWorld
from training.knowledge import KnowledgeCurriculum
from training.tutor import CaregiverTutor


STAGE_SCENES = {
    "stage_0_sensory_continuity": ScenarioName.NURSERY,
    "stage_1_body": ScenarioName.NURSERY,
    "stage_2_objects": ScenarioName.HIDDEN_OBJECT,
    "stage_3_causality": ScenarioName.SWITCH_LIGHT,
    "stage_4_agents": ScenarioName.SOCIAL,
    "stage_5_communication": ScenarioName.NURSERY,
    "stage_6_abstract": ScenarioName.UNKNOWN_BOX,
    "stage_7_open": ScenarioName.OPEN_WORLD,
}


class DevelopmentalCurriculum:
    def __init__(self, tutor: CaregiverTutor) -> None:
        self.tutor = tutor
        self.knowledge = KnowledgeCurriculum()

    def scene_for(self, stage: str) -> ScenarioName:
        return STAGE_SCENES.get(stage, ScenarioName.NURSERY)

    def apply(self, org: Organism, world: SimulatedWorld) -> str | None:
        stage = org.development.stage.value
        wanted = self.scene_for(stage)
        if world.scenario is not wanted and org.tick % 80 == 0:
            world.reset(wanted)
        speech = None
        talkative = any(k in stage for k in ("object", "causal", "agent", "communication", "abstract", "open"))
        if talkative:
            speech = self.tutor.maybe_label(world, stage)
        if speech is None and ("communication" in stage or "open" in stage):
            speech = self.tutor.maybe_query()
        if speech:
            world.inject_speech("caregiver", speech)
            org.ingest_speech("caregiver", speech)
        if "abstract" in stage or "open" in stage:
            lesson = self.knowledge.next_lesson(org.tick)
            if lesson:
                org.ingest_document(lesson)
        return speech
