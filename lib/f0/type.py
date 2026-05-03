from typing import Literal

PitchMethod = Literal["pm", "harvest", "crepe", "rmvpe", "fcpe", "dio"]
PITCH_METHODS: list[PitchMethod] = ["pm", "harvest", "crepe", "rmvpe", "fcpe"]
ALL_PITCH_METHODS: tuple[PitchMethod, ...] = (
    "pm",
    "harvest",
    "crepe",
    "rmvpe",
    "fcpe",
    "dio",
)
