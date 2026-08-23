from embodiment.body import EmbodiedBody
from planning.action import primitive_vector
from shared.config import load_config
from shared.rng import RNG
from shared.types import ActionKind
from simulation.scenarios import ScenarioName
from simulation.world import SimulatedWorld


def test_world_steps_and_switch():
    cfg = load_config("development", seed=1)
    world = SimulatedWorld(cfg, RNG(1), ScenarioName.SWITCH_LIGHT)
    body = EmbodiedBody(cfg)
    obs0 = world.observe()
    assert obs0.sensors.vision.size == cfg.vision_h * cfg.vision_w * 3
    world.objects["switch_0"].switch_state = 1.0
    world._update_devices()
    assert world.objects["light_0"].light_state == 1.0
    world.step(body.decode(primitive_vector(ActionKind.MOVE, cfg.action_dim), ActionKind.MOVE, 1))
    assert world.tick == 1


def test_hidden_object_scenario():
    cfg = load_config("development", seed=2)
    world = SimulatedWorld(cfg, RNG(2), ScenarioName.HIDDEN_OBJECT)
    assert any(o.hidden for o in world.objects.values())
