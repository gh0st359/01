"""Persistent developmental simulation with physics, devices, and social agents."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from shared.config import OrganismConfig
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
from shared.rng import RNG
from simulation.objects import SimObject
from simulation.physics import clamp_world, integrate, resolve_circles
from simulation.render import render_egocentric
from simulation.scenarios import ScenarioName, build_scenario


@dataclass
class WorldEvent:
    tick: int
    kind: str
    payload: dict = field(default_factory=dict)


class SimulatedWorld:
    def __init__(self, cfg: OrganismConfig, rng: RNG, scenario: ScenarioName = ScenarioName.NURSERY) -> None:
        self.cfg = cfg
        self.rng = rng
        self.width = 12.0
        self.height = 12.0
        self.tick = 0
        self.time = 0.0
        self.events: list[WorldEvent] = []
        self.audio_bus: list[tuple[np.ndarray, float, float]] = []
        self.body_id = "self_body"
        self.held: str | None = None
        self.utterances: list[dict] = []
        self.scenario = scenario
        self.objects: dict[str, SimObject] = {}
        self._init_body()
        build_scenario(self, scenario)

    def _init_body(self) -> None:
        self.objects[self.body_id] = SimObject(
            object_id=self.body_id,
            kind="body",
            position=np.array([6.0, 6.0], dtype=np.float64),
            velocity=np.zeros(2),
            radius=0.35,
            color=np.array([0.2, 0.85, 0.75]),
            graspable=False,
            is_agent=True,
            label="self",
        )

    def body(self) -> SimObject:
        return self.objects[self.body_id]

    def add_object(self, obj: SimObject) -> None:
        self.objects[obj.object_id] = obj

    def reset(self, scenario: ScenarioName | None = None) -> None:
        if scenario is not None:
            self.scenario = scenario
        held = self.held
        self.objects = {}
        self.held = None
        self._init_body()
        build_scenario(self, self.scenario)
        if held and held in self.objects:
            self.held = held

    def observe(self) -> WorldObservation:
        body = self.body()
        vision = render_egocentric(
            objects=list(self.objects.values()),
            body=body,
            h=self.cfg.vision_h,
            w=self.cfg.vision_w,
            fov=5.5,
        )
        audio = self._audio_frame()
        contacts = np.zeros(self.cfg.tactile_dim)
        pressure = np.zeros(self.cfg.tactile_dim)
        i = 0
        for obj in self.objects.values():
            if obj.object_id == self.body_id:
                continue
            d = float(np.linalg.norm(obj.position - body.position))
            if d < body.radius + obj.radius + 0.05 and i < contacts.size:
                contacts[i] = 1.0
                pressure[i] = max(0.0, body.radius + obj.radius + 0.05 - d)
                i += 1
        tactile = TactileFrame(contacts=contacts, pressure=pressure, gripper_force=1.0 if self.held else 0.0, tick=self.tick)
        joint = np.array([body.heading, 1.0 if self.held else 0.0, body.energy], dtype=np.float64)
        body_state = BodyState(
            position=body.position.copy(),
            heading=body.heading,
            velocity=body.velocity.copy(),
            angular_velocity=0.0,
            joint_angles=joint,
            gripper_open=0.0 if self.held else 1.0,
            energy=body.energy,
            damage=0.0,
            contact_mask=contacts.copy(),
            reachable=self._reachability(),
            tick=self.tick,
        )
        proprio = np.array(
            [
                body.position[0] / self.width,
                body.position[1] / self.height,
                np.sin(body.heading),
                np.cos(body.heading),
                body.velocity[0],
                body.velocity[1],
                1.0 if self.held else 0.0,
                body.energy,
            ],
            dtype=np.float64,
        )
        sensors = SensorFrame(
            vision=vision.ravel(),
            vision_shape=(self.cfg.vision_h, self.cfg.vision_w, 3),
            audio=audio,
            proprioception=proprio,
            tactile=tactile,
            body=body_state,
            energy=body.energy,
            orientation=np.array([np.sin(body.heading), np.cos(body.heading)]),
            motion=body.velocity.copy(),
            internal=np.array([body.energy, float(self.tick % 100) / 100.0]),
            tick=self.tick,
            timestamp=self.time,
        )
        light = 0.25
        for obj in self.objects.values():
            if obj.kind == "light":
                light = max(light, 0.25 + 0.75 * obj.light_state)
        hidden = {oid: o.copy_pose() for oid, o in self.objects.items()}
        return WorldObservation(
            sensors=sensors,
            nearby_entity_count=sum(1 for o in self.objects.values() if o.object_id != self.body_id and not o.hidden),
            light_level=light,
            audio_events=[e.kind for e in self.events[-6:]],
            hidden_from_organism=hidden,
        )

    def step(self, command: MotorCommand) -> WorldObservation:
        dt = self.cfg.dt
        body = self.body()
        acc = np.array(
            [np.cos(body.heading) * command.linear_velocity * 6.0, np.sin(body.heading) * command.linear_velocity * 6.0],
            dtype=np.float64,
        )
        body.heading = float(body.heading + command.angular_velocity * dt * 4.0)
        body.position, body.velocity = integrate(body.position, body.velocity, acc, dt)
        body.position = clamp_world(body.position, body.radius, self.width, self.height)
        body.energy = float(np.clip(body.energy - 0.0008 * (abs(command.linear_velocity) + 0.15), 0.05, 1.0))
        if command.look_heading != 0.0:
            body.heading = float(body.heading + 0.35 * command.look_heading)

        self._step_other_agents(dt)
        self._resolve_collisions()
        self._update_devices()
        self._handle_manipulation(command)
        if self.held and self.held in self.objects:
            held = self.objects[self.held]
            held.position = body.position + np.array([np.cos(body.heading), np.sin(body.heading)]) * 0.55
            held.velocity = body.velocity.copy()
            held.held_by = self.body_id
        body.energy = float(np.clip(body.energy + 0.0003, 0.05, 1.0))
        self.tick += 1
        self.time += dt
        return self.observe()

    def inject_speech(self, speaker: str, text: str, bearing: float = 0.0) -> None:
        sig = np.zeros(self.cfg.audio_dim)
        for i, ch in enumerate(text[: sig.size]):
            sig[i % sig.size] += (ord(ch) % 23) / 23.0
        self.audio_bus.append((sig, 1.0, bearing))
        self.events.append(WorldEvent(self.tick, "speech", {"speaker": speaker, "text": text}))
        self.utterances.append({"tick": self.tick, "speaker": speaker, "text": text})

    def set_hidden(self, object_id: str, hidden: bool) -> None:
        if object_id in self.objects:
            self.objects[object_id].hidden = hidden
            self.events.append(WorldEvent(self.tick, "visibility", {"id": object_id, "hidden": hidden}))

    def swap_body_colors(self) -> None:
        body = self.body()
        other = next((o for o in self.objects.values() if o.is_agent and o.object_id != self.body_id), None)
        if other:
            body.color, other.color = other.color.copy(), body.color.copy()
            self.events.append(WorldEvent(self.tick, "body_swap", {"with": other.object_id}))

    def delay_motor(self, steps: int) -> None:
        self.events.append(WorldEvent(self.tick, "agency_delay", {"steps": steps}))

    def snapshot(self) -> dict:
        return {
            "tick": self.tick,
            "time": self.time,
            "held": self.held,
            "scenario": self.scenario.value,
            "objects": [o.copy_pose() | {"heading": o.heading, "energy": o.energy, "inside": o.inside_of} for o in self.objects.values()],
        }

    def public_view(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "tick": self.tick,
            "held": self.held,
            "objects": [o.copy_pose() for o in self.objects.values() if not o.hidden],
        }

    def _reachability(self) -> np.ndarray:
        body = self.body()
        reach = np.zeros(8)
        i = 0
        for obj in self.objects.values():
            if obj.object_id == self.body_id:
                continue
            d = float(np.linalg.norm(obj.position - body.position))
            if i < reach.size:
                reach[i] = max(0.0, 1.0 - d / 2.5)
                i += 1
        return reach

    def _audio_frame(self) -> AudioFrame:
        samples = np.zeros(self.cfg.audio_dim)
        energy = 0.0
        bearing = 0.0
        speech = 0.0
        for sig, en, br in self.audio_bus[-4:]:
            s = np.zeros(self.cfg.audio_dim)
            s[: min(s.size, sig.size)] = sig[: min(s.size, sig.size)]
            samples += s
            energy += en
            bearing = br
            speech = max(speech, 0.8)
        self.audio_bus = self.audio_bus[-2:]
        if energy == 0:
            return empty_audio(self.tick, self.cfg.audio_dim)
        return AudioFrame(samples=samples, event_energy=energy, source_bearing=bearing, speech_like=speech, tick=self.tick)

    def _resolve_collisions(self) -> None:
        ids = list(self.objects)
        for i, a_id in enumerate(ids):
            a = self.objects[a_id]
            if a.hidden or a.held_by:
                continue
            for b_id in ids[i + 1 :]:
                b = self.objects[b_id]
                if b.hidden or b.held_by:
                    continue
                a.position, b.position = resolve_circles(a.position, a.radius, b.position, b.radius)
                a.position = clamp_world(a.position, a.radius, self.width, self.height)
                b.position = clamp_world(b.position, b.radius, self.width, self.height)

    def _update_devices(self) -> None:
        switches = [o for o in self.objects.values() if o.kind == "switch"]
        lights = [o for o in self.objects.values() if o.kind == "light"]
        if switches and lights:
            lights[0].light_state = switches[0].switch_state
        for box in [o for o in self.objects.values() if o.kind == "box"]:
            for obj in self.objects.values():
                if obj.object_id in {box.object_id, self.body_id}:
                    continue
                if obj.inside_of == box.object_id:
                    obj.hidden = box.open_state < 0.5
                    obj.position = box.position.copy()

    def _handle_manipulation(self, command: MotorCommand) -> None:
        body = self.body()
        if command.gripper > 0.5 and self.held is None:
            target = self._nearest_interactable()
            if target and target.graspable:
                self.held = target.object_id
                target.held_by = self.body_id
                self.events.append(WorldEvent(self.tick, "grasp", {"id": target.object_id}))
        elif command.gripper < -0.2 and self.held:
            dropped = self.objects[self.held]
            dropped.held_by = None
            self.events.append(WorldEvent(self.tick, "release", {"id": self.held}))
            self.held = None
        if command.interact > 0.5:
            target = self._nearest_interactable(radius=1.1)
            if target is None:
                return
            if target.kind == "switch":
                target.switch_state = 1.0 - target.switch_state
                self.events.append(WorldEvent(self.tick, "toggle", {"id": target.object_id, "state": target.switch_state}))
                self.audio_bus.append((np.ones(8) * target.switch_state, 0.8, 0.0))
            elif target.kind == "box":
                target.open_state = 1.0 - target.open_state
                self.events.append(WorldEvent(self.tick, "box", {"id": target.object_id, "open": target.open_state}))
            elif target.kind == "door":
                target.open_state = 1.0 - target.open_state

    def _nearest_interactable(self, radius: float = 0.85) -> SimObject | None:
        body = self.body()
        best = None
        best_d = 1e9
        facing = np.array([np.cos(body.heading), np.sin(body.heading)])
        for obj in self.objects.values():
            if obj.object_id == self.body_id or obj.hidden:
                continue
            delta = obj.position - body.position
            d = float(np.linalg.norm(delta))
            if d < radius and d < best_d and float(np.dot(delta / (d + 1e-6), facing)) > -0.2:
                best, best_d = obj, d
        return best

    def _step_other_agents(self, dt: float) -> None:
        body = self.body()
        for obj in self.objects.values():
            if not obj.is_agent or obj.object_id == self.body_id:
                continue
            if obj.agent_goal is None:
                continue
            delta = obj.agent_goal - obj.position
            dist = float(np.linalg.norm(delta)) + 1e-6
            if dist < 0.3:
                obj.agent_goal = np.array(
                    [float(self.rng.uniform(1.5, self.width - 1.5)), float(self.rng.uniform(1.5, self.height - 1.5))]
                )
            acc = 3.2 * delta / dist
            # false-belief support: if this agent didn't see a hidden move, keep old goal
            obj.position, obj.velocity = integrate(obj.position, obj.velocity, acc, dt, friction=1.8)
            obj.position = clamp_world(obj.position, obj.radius, self.width, self.height)
            obj.heading = float(np.arctan2(obj.velocity[1], obj.velocity[0] + 1e-6))
            # simple social spacing
            if float(np.linalg.norm(obj.position - body.position)) < 0.6:
                obj.position = obj.position + 0.05 * (obj.position - body.position)
