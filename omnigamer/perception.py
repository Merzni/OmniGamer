"""Structured visual evidence. Model reports remain fallible, not ground truth."""
from typing import Literal, Optional
from pydantic import BaseModel, Field


class VisibleItem(BaseModel):
    slot: int = Field(ge=1, le=9)
    name: str = Field(min_length=1, max_length=40)
    count: int = Field(ge=1, le=999)


class FrameFacts(BaseModel):
    scene: Literal["menu", "loading", "gameplay", "unknown"] = "unknown"
    pitch: Literal["up", "down", "level", "unknown"] = "unknown"
    open_ground: Optional[bool] = None
    safe_footing: Optional[bool] = None
    visible_items_readable: bool = False
    items: list[VisibleItem] = Field(default_factory=list, max_length=9)
    hand_item: str = Field(default="unknown", max_length=40)
    menu_id: str = Field(default="unknown", max_length=64)
    menu_state: str = Field(default="default", max_length=64)
    selection_visible: Optional[bool] = None
    play_enabled: Optional[bool] = None

    def item_count(self, target: str) -> Optional[int]:
        if not self.visible_items_readable or len({item.slot for item in self.items}) != len(self.items):
            return None
        def canonical(name):
            value = name.casefold().strip().replace(" ", "_")
            if value in {"log", "wood", "wood_log", "oak_log", "birch_log", "spruce_log", "бревно", "древесина"}:
                return "log"
            return value
        return sum(item.count for item in self.items if canonical(item.name) == canonical(target))


class VisualAssessment(BaseModel):
    before: FrameFacts = Field(default_factory=FrameFacts)
    after: FrameFacts = Field(default_factory=FrameFacts)
    result: Literal["progress", "blocked", "regressed", "unknown"] = "unknown"
    confidence: float = Field(default=0, ge=0, le=1, allow_inf_nan=False)
    evidence: str = Field(default="", max_length=500)
    uncertainty: str = Field(default="", max_length=300)
    menu_action_succeeded: Optional[bool] = None
    evidence_source: Literal["visual_model","visual_model_with_geometry","screen_geometry"] = "visual_model"

    @property
    def usable(self):
        return self.confidence >= 0.85 and len(self.evidence.strip()) >= 8


def hud_crop(frame):
    """Magnify the visible lower-center HUD, not the unopened inventory."""
    width, height = frame.size
    return frame.crop((int(width * .25), int(height * .80), int(width * .75), height))
