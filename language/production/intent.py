"""Communicative intent is a latent state, not a speech-act lookup table."""

from __future__ import annotations

import numpy as np

from shared.types import CommunicativeIntent, Goal, SemanticFrame, SpeechAct, Vector, WorkspaceContents


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
        pred = workspace.broadcast.copy()
        roles: dict[str, Vector] = {}
        refs: list[str] = []
        urgency = float(np.tanh(uncertainty + social_drive + (0.3 if goal is not None else 0.0)))
        if urgency < 0.15:
            return None
        if focus_vec is not None:
            roles["focus"] = focus_vec
        if goal is not None:
            pred = goal.description_vector
            if goal.target_entity:
                refs.append(goal.target_entity)
        frame = SemanticFrame(SpeechAct.UNKNOWN, pred, roles, refs, min(1.0, urgency + 0.2), tick)
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
