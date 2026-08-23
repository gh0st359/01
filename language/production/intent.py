"""Form communicative intent from cognition — meaning exists before words."""

from __future__ import annotations

from shared.types import CommunicativeIntent, Goal, GoalOrigin, SemanticFrame, SpeechAct, Vector, WorkspaceContents
import numpy as np


class IntentFormer:
    def form(
        self,
        workspace: WorkspaceContents,
        goal: Goal | None,
        focus_vec: Vector | None,
        uncertainty: float,
        social_drive: float,
        retrieved_ids: list[str],
        prediction_error: float,
        tick: int,
        addressed: str | None,
    ) -> CommunicativeIntent | None:
        if not workspace.winners and goal is None:
            return None
        urgency = 0.0
        act = SpeechAct.INFORM
        pred = workspace.broadcast.copy()
        roles: dict[str, Vector] = {}
        refs: list[str] = []
        if goal is not None and goal.origin is GoalOrigin.EPISTEMIC:
            act = SpeechAct.QUERY
            urgency = 0.55 + 0.3 * uncertainty
            pred = goal.description_vector
            if goal.target_entity:
                refs.append(goal.target_entity)
        elif goal is not None and goal.origin is GoalOrigin.SOCIAL:
            act = SpeechAct.INFORM
            urgency = 0.4 + 0.4 * social_drive
            pred = goal.description_vector
        elif any(w.kind.value == "language_event" for w in workspace.winners):
            act = SpeechAct.INFORM
            urgency = 0.45 + 0.2 * social_drive
            pred = workspace.broadcast
        elif uncertainty > 0.65 and social_drive > 0.25:
            act = SpeechAct.QUERY
            urgency = 0.5
        elif prediction_error > 0.35:
            act = SpeechAct.ASSERT
            urgency = 0.35
            roles["unexpected"] = workspace.broadcast
        else:
            if social_drive < 0.35 and uncertainty < 0.5:
                return None
            urgency = 0.2 * social_drive
        if focus_vec is not None:
            roles["focus"] = focus_vec
        frame = SemanticFrame(act, pred, roles, refs, min(1.0, urgency + 0.2), tick)
        return CommunicativeIntent(
            frame=frame,
            urgency=urgency,
            social_address=addressed,
            formed_tick=tick,
            workspace_hash=str(abs(hash(workspace.broadcast.tobytes()))),
            preceding_uncertainty=uncertainty,
            preceding_retrieval_ids=retrieved_ids,
            preceding_prediction_error=prediction_error,
        )
