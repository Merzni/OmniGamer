from PIL import Image, ImageDraw


class SimState:
    def __init__(self, profile, target_after_turns=1):
        self.profile = profile
        self.heading = 0
        self.target_after_turns = target_after_turns
        self.forward_pulses = 0
        self.item = False
        self.actions = 0
        self.detector_available = True

    def target_geometry(self):
        """Model a target shifting across view on turns and growing on forward motion."""
        offset = self.heading - self.target_after_turns
        center = min(0.9, max(0.1, 0.5 - 0.12 * offset))
        half_width = min(0.42, 0.1 + 0.03 * self.forward_pulses)
        return center, half_width


class SimCapture:
    def __init__(self, state, size=(320, 180)):
        self.state, self.size = state, size

    def grab_frame(self):
        image = Image.new("RGB", self.size, (28, 38, 48))
        draw = ImageDraw.Draw(image)
        if self.state.heading >= self.state.target_after_turns:
            center, half_width = self.state.target_geometry()
            center_px = int(center * self.size[0])
            half_px = int(half_width * self.size[0] / 2)
            draw.rectangle((center_px - half_px, 35, center_px + half_px, 150),
                           fill=(80, 160, 75))
        for region in self.state.profile.regions.values():
            box = (int(region.x1 * self.size[0]), int(region.y1 * self.size[1]),
                   int(region.x2 * self.size[0]) - 1, int(region.y2 * self.size[1]) - 1)
            draw.rectangle(box, fill=(25, 25, 25))
            if self.state.item:
                inset = max(1, min(self.size) // 50)
                draw.rectangle((box[0] + inset, box[1] + inset,
                                box[2] - inset, box[3] - inset), fill=(245, 190, 35))
        return image


class SimDetector:
    def __init__(self, state):
        self.state = state

    def health(self):
        return {"ok": True, "model": "sim"}

    def detect(self, image, queries, score_min=None, max_boxes=None):
        if not self.state.detector_available:
            return None
        boxes = []
        if self.state.heading >= self.state.target_after_turns:
            center, half_width = self.state.target_geometry()
            boxes.append({"x1": center - half_width, "y1": 0.2,
                          "x2": center + half_width, "y2": 0.8,
                          "score": 0.99, "query": queries[0]})
        return {"boxes": boxes}


class SimControls:
    def __init__(self, state):
        self.state = state
        self.released = 0

    def press_key(self, key, duration):
        self.state.actions += 1
        if key == self.state.profile.move_keys["forward"]:
            self.state.forward_pulses += 1
            return
        action = next((a for a in self.state.profile.actions.values()
                       if a.kind == "key" and a.value == key), None)
        if action:
            self.state.item = True

    def hold_mouse(self, button, duration):
        self.state.actions += 1
        action = next((a for a in self.state.profile.actions.values()
                       if a.kind == "mouse" and a.value == button), None)
        if action:
            self.state.item = True

    def move_camera(self, dx, dy):
        self.state.actions += 1
        if dx > 0:
            self.state.heading += 1
        elif dx < 0:
            self.state.heading = max(0, self.state.heading - 1)

    def release_all(self):
        self.released += 1


class SimRecorder:
    def __init__(self):
        self.rows, self.finished = [], []

    def __call__(self, row, before=None, after=None):
        self.rows.append(row)

    def finish(self, result, finalize_run=False):
        self.finished.append(result)


def make_profile_a():
    from omnigamer.profiles import GameProfile
    return GameProfile(game_id="sim-alpha", camera_type="first_person",
        move_keys={"forward": "w", "backward": "s", "left": "a", "right": "d"},
        actions={"interact": {"kind": "mouse", "value": "left", "duration": 0.05}},
        regions={"panel": {"x1": 0.78, "y1": 0.82, "x2": 0.98, "y2": 0.98}},
        thresholds={"region_change": 0.08},
        motion_limits={"max_action_seconds": 0.75, "max_run_seconds": 15, "max_actions": 8,
                       "max_turn_counts": 5, "max_sweep_counts": 10})


def make_profile_b():
    from omnigamer.profiles import GameProfile
    return GameProfile(game_id="sim-beta", camera_type="first_person",
        move_keys={"forward": "up", "backward": "down", "left": "left", "right": "right"},
        actions={"activate": {"kind": "key", "value": "enter", "duration": 0.05}},
        regions={"status_strip": {"x1": 0.02, "y1": 0.82, "x2": 0.24, "y2": 0.98}},
        thresholds={"region_change": 0.08},
        motion_limits={"max_action_seconds": 0.75, "max_run_seconds": 15, "max_actions": 8,
                       "max_turn_counts": 5, "max_sweep_counts": 10})
