"""Motor contract documentation and validation."""

from shared.contracts import MotorCommand


def validate_motor(cmd: MotorCommand) -> MotorCommand:
    cmd.linear_velocity = max(-1.0, min(1.0, cmd.linear_velocity))
    cmd.angular_velocity = max(-1.0, min(1.0, cmd.angular_velocity))
    cmd.gripper = max(-1.0, min(1.0, cmd.gripper))
    cmd.interact = max(0.0, min(1.0, cmd.interact))
    cmd.speak = max(0.0, min(1.0, cmd.speak))
    return cmd
