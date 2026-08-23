"""Vectorized developmental worlds. No privileged object IDs reach the organism."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from shared.contracts import (
    AudioFrame,
    BodyState,
    MotorCommand,
    SensorFrame,
    TactileFrame,
    WorldObservation,
    empty_audio,
    empty_tactile,
)


KINDS = ("disk", "block", "switch", "light", "door", "agent", "tool", "container")


@dataclass
class BatchState:
    pos: np.ndarray  # [B, N, 2]
    vel: np.ndarray
    color: np.ndarray  # [B, N, 3]
    radius: np.ndarray  # [B, N]
    kind: np.ndarray  # [B, N] int
    mass: np.ndarray
    friction: np.ndarray
    hidden: np.ndarray  # occlusion flags
    light: np.ndarray
    alive: np.ndarray
    heading: np.ndarray  # [B]
    energy: np.ndarray  # [B]
    held: np.ndarray  # [B] index or -1
    tick: int
    seed: int
    gravity: np.ndarray  # [B, 2]
    walls: np.ndarray  # [B, 4] rooms


class ProceduralBatch:
    """B parallel rooms with movable objects, occlusion, switches, other agents."""

    def __init__(self, batch: int, n_objects: int, vision: int, seed: int, held_out: bool = False) -> None:
        self.batch = batch
        self.n = n_objects
        self.vision = vision
        self.rng = np.random.default_rng(seed + (10_007 if held_out else 0))
        self.held_out = held_out
        self.extent = 8.0
        self.state = self.reset()

    def reset(self) -> BatchState:
        b, n = self.batch, self.n
        pos = self.rng.uniform(0.8, self.extent - 0.8, size=(b, n, 2))
        pos[:, 0] = self.extent * 0.5  # body at center
        vel = np.zeros((b, n, 2), dtype=np.float64)
        color = self.rng.uniform(0.15, 0.95, size=(b, n, 3))
        radius = self.rng.uniform(0.18, 0.42, size=(b, n))
        radius[:, 0] = 0.32
        kind = self.rng.integers(0, len(KINDS), size=(b, n))
        kind[:, 0] = 5  # body as agent
        kind[:, 1] = 3  # at least one light
        if n > 2:
            kind[:, 2] = 2  # switch
        if n > 3:
            kind[:, 3] = 6  # tool
        if n > 4:
            kind[:, 4] = 7  # container
        if n > 5:
            kind[:, 5] = 5  # other agent
        mass = self.rng.uniform(0.4, 2.2, size=(b, n))
        friction = self.rng.uniform(0.04, 0.18, size=(b, n))
        hidden = np.zeros((b, n), dtype=bool)
        light = self.rng.uniform(0.2, 1.0, size=(b, n))
        alive = np.ones((b, n), dtype=bool)
        heading = self.rng.uniform(-np.pi, np.pi, size=(b,))
        energy = np.ones((b,)) * 0.85
        held = np.full((b,), -1, dtype=np.int64)
        gravity = np.zeros((b, 2))
        if self.held_out:
            gravity[:, 1] = self.rng.uniform(-0.01, 0.01, size=(b,))
        walls = np.array([[0.0, 0.0, self.extent, self.extent]] * b)
        return BatchState(pos, vel, color, radius, kind, mass, friction, hidden, light, alive, heading, energy, held, 0, int(self.rng.integers(0, 1_000_000)), gravity, walls)

    def step(self, motors: list[MotorCommand] | np.ndarray) -> None:
        s = self.state
        b, n = self.batch, self.n
        if isinstance(motors, list):
            act = np.zeros((b, 6), dtype=np.float64)
            for i, m in enumerate(motors[:b]):
                act[i] = [m.linear_velocity, m.angular_velocity, m.gripper, m.interact, m.speak, m.look_heading]
        else:
            act = np.asarray(motors, dtype=np.float64)
            if act.ndim == 1:
                act = np.repeat(act.reshape(1, -1), b, axis=0)
        # body control
        s.heading = s.heading + act[:, 1] * 0.25
        dir_xy = np.stack([np.cos(s.heading), np.sin(s.heading)], axis=-1)
        s.vel[:, 0] = dir_xy * act[:, 0][:, None] * 0.35
        # other agent wanders independently
        if n > 5:
            wander = self.rng.normal(0, 0.05, size=(b, 2))
            s.vel[:, 5] += wander
        s.vel += s.gravity[:, None, :]
        s.vel *= (1.0 - s.friction[..., None])
        s.pos = s.pos + s.vel
        s.pos = np.clip(s.pos, 0.3, self.extent - 0.3)
        # collisions
        for i in range(n):
            for j in range(i + 1, n):
                d = s.pos[:, i] - s.pos[:, j]
                dist = np.linalg.norm(d, axis=-1) + 1e-6
                min_d = s.radius[:, i] + s.radius[:, j]
                hit = dist < min_d
                if not hit.any():
                    continue
                nrm = d / dist[:, None]
                push = (min_d - dist)[:, None] * 0.5
                s.pos[hit, i] += nrm[hit] * push[hit]
                s.pos[hit, j] -= nrm[hit] * push[hit]
        # switch → light
        if n > 2:
            dsw = np.linalg.norm(s.pos[:, 0] - s.pos[:, 2], axis=-1)
            tog = (dsw < 0.7) & (act[:, 3] > 0.4)
            s.light[:, 1] = np.where(tog, 1.0 - s.light[:, 1], s.light[:, 1])
        # grasp
        if n > 1:
            near = np.linalg.norm(s.pos[:, 0:1] - s.pos, axis=-1)
            can = (near < (s.radius[:, 0:1] + s.radius + 0.15)) & (np.arange(n)[None, :] != 0)
            want = act[:, 2] > 0.55
            for i in range(b):
                if want[i] and s.held[i] < 0:
                    opts = np.where(can[i])[0]
                    if opts.size:
                        s.held[i] = int(opts[np.argmin(near[i, opts])])
                elif act[i, 2] < 0.35:
                    s.held[i] = -1
                if s.held[i] >= 0:
                    s.pos[i, s.held[i]] = s.pos[i, 0] + dir_xy[i] * 0.45
        # occlusion: hide objects behind containers from egocentric view when overlapping in projection
        s.hidden[:] = False
        if n > 4:
            behind = np.linalg.norm(s.pos - s.pos[:, 4:5], axis=-1) < 0.35
            s.hidden |= behind & (np.arange(n)[None, :] != 4) & (np.arange(n)[None, :] != 0)
        s.energy = np.clip(s.energy - 0.001 + 0.002 * (np.abs(act[:, 0]) < 0.05), 0.05, 1.0)
        s.tick += 1

    def render(self, env: int = 0) -> np.ndarray:
        s = self.state
        h = w = self.vision
        img = np.zeros((h, w, 3), dtype=np.float64)
        img[:, :] = 0.08
        light_scale = 0.45 + 0.55 * float(s.light[env, min(1, self.n - 1)])
        img *= light_scale
        c, sn = np.cos(-s.heading[env]), np.sin(-s.heading[env])
        rot = np.array([[c, -sn], [sn, c]])
        fov = 4.2
        order = np.argsort(-s.pos[env, :, 1])
        for i in order:
            if s.hidden[env, i] and i != 0:
                continue
            rel = rot @ (s.pos[env, i] - s.pos[env, 0])
            if abs(rel[0]) > fov or abs(rel[1]) > fov:
                continue
            px = int((rel[0] / fov + 1.0) * 0.5 * (w - 1))
            py = int((rel[1] / fov + 1.0) * 0.5 * (h - 1))
            rad = max(1, int(s.radius[env, i] / fov * w * 0.8))
            col = np.clip(s.color[env, i] * (0.4 + 0.6 * s.light[env, i]), 0, 1)
            yy, xx = np.ogrid[-rad : rad + 1, -rad : rad + 1]
            mask = xx * xx + yy * yy <= rad * rad
            for dy, dx in zip(*np.where(mask)):
                y, x = py + dy - rad, px + dx - rad
                if 0 <= y < h and 0 <= x < w:
                    img[y, x] = col
        return img

    def observation(self, env: int = 0) -> WorldObservation:
        img = self.render(env)
        s = self.state
        proprio = np.array(
            [
                s.heading[env] / np.pi,
                s.energy[env],
                s.vel[env, 0, 0],
                s.vel[env, 0, 1],
                float(s.held[env] >= 0),
                s.pos[env, 0, 0] / self.extent,
                s.pos[env, 0, 1] / self.extent,
                float(s.tick % 100) / 100.0,
            ],
            dtype=np.float64,
        )
        sensors = SensorFrame(
            vision=img.reshape(-1),
            vision_shape=(self.vision, self.vision, 3),
            audio=empty_audio(s.tick),
            proprioception=proprio,
            tactile=empty_tactile(s.tick),
            body=BodyState(
                position=s.pos[env, 0].copy(),
                heading=float(s.heading[env]),
                velocity=s.vel[env, 0].copy(),
                angular_velocity=0.0,
                joint_angles=np.zeros(4),
                gripper_open=1.0 if s.held[env] < 0 else 0.0,
                energy=float(s.energy[env]),
                damage=0.0,
                contact_mask=np.zeros(8),
                reachable=np.ones(4),
                tick=s.tick,
            ),
            energy=float(s.energy[env]),
            orientation=np.array([np.cos(s.heading[env]), np.sin(s.heading[env])]),
            motion=s.vel[env, 0].copy(),
            internal=np.array([s.energy[env], 0.0]),
            tick=s.tick,
            timestamp=float(s.tick) * 0.05,
        )
        return WorldObservation(
            sensors=sensors,
            nearby_entity_count=int(self.n - 1),
            light_level=float(s.light[env, min(1, self.n - 1)]),
            hidden_from_organism={"truth_pos": s.pos[env].tolist(), "kinds": s.kind[env].tolist()},
        )

    def public_view(self, env: int = 0) -> dict:
        s = self.state
        objs = []
        for i in range(self.n):
            objs.append(
                {
                    "id": f"o{i}",
                    "kind": KINDS[int(s.kind[env, i]) % len(KINDS)],
                    "x": float(s.pos[env, i, 0]),
                    "y": float(s.pos[env, i, 1]),
                    "r": float(s.radius[env, i]),
                    "color": s.color[env, i].tolist(),
                    "hidden": bool(s.hidden[env, i]),
                    "light": float(s.light[env, i]),
                }
            )
        return {"width": self.extent, "height": self.extent, "objects": objs}

    def inject_speech(self, _speaker: str, _text: str) -> None:
        return None

    def body(self, env: int = 0):
        class _B:
            position = self.state.pos[env, 0]
            heading = float(self.state.heading[env])

        return _B()
