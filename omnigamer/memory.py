"""Episodic memory, screen state fingerprinting, and trial-and-error learning."""
import sqlite3
import json
from contextlib import closing, contextmanager
from uuid import uuid4
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict
from PIL import Image

class GameMemory:
    """Manages persistent experience, state transitions, and dead-click detection."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._memory_uri = None
        self._memory_anchor = None
        if str(db_path) == ":memory:":
            self._memory_uri = f"file:omnigamer-{uuid4().hex}?mode=memory&cache=shared"
            self._memory_anchor = sqlite3.connect(self._memory_uri, uri=True, check_same_thread=False)
        self.last_learning_match_count = 0
        self._init_db()

    @contextmanager
    def _connection(self):
        # sqlite3's own context manager commits/rolls back but does not close.
        connection = (sqlite3.connect(self._memory_uri, uri=True, check_same_thread=False)
                      if self._memory_uri else sqlite3.connect(self.db_path))
        with closing(connection) as conn:
            with conn:
                yield conn

    def _init_db(self):
        with self._connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL,
                    state_hash TEXT,
                    action_type TEXT,
                    x INTEGER,
                    y INTEGER,
                    key TEXT,
                    screen_delta REAL,
                    effective INTEGER,
                    commentary TEXT,
                    intent TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS dead_clicks (
                    state_hash TEXT,
                    x INTEGER,
                    y INTEGER,
                    fail_count INTEGER,
                    PRIMARY KEY (state_hash, x, y)
                )
            """)
            # Additive migration: unattributed old experiences stay in legacy.
            columns = {row[1] for row in conn.execute("PRAGMA table_info(experiences)")}
            additions = {
                "game_id": "TEXT NOT NULL DEFAULT 'legacy'",
                "next_state_hash": "TEXT", "duration": "REAL",
                "x2": "INTEGER", "y2": "INTEGER", "observation": "TEXT",
                "session_id": "TEXT", "visual_signature": "TEXT",
                "learnable": "INTEGER NOT NULL DEFAULT 1",
                "goal_id": "TEXT", "outcome_status": "TEXT NOT NULL DEFAULT 'unverified'",
                "assessment_json": "TEXT", "action_source": "TEXT", "confidence": "REAL",
                "timings_json": "TEXT", "policy_version": "INTEGER NOT NULL DEFAULT 1",
            }
            for name, declaration in additions.items():
                if name not in columns:
                    conn.execute(f"ALTER TABLE experiences ADD COLUMN {name} {declaration}")
            conn.execute("CREATE INDEX IF NOT EXISTS ix_experience_game_state "
                         "ON experiences(game_id, state_hash, id)")
            conn.execute("CREATE INDEX IF NOT EXISTS ix_experience_game_recent "
                         "ON experiences(game_id, id)")
            conn.execute("CREATE TABLE IF NOT EXISTS agent_state (game_id TEXT PRIMARY KEY, value TEXT NOT NULL)")
            conn.execute("""CREATE TABLE IF NOT EXISTS visual_checks (
                id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp REAL, game_id TEXT, session_id TEXT,
                step INTEGER, phase TEXT, goal_id TEXT, assessment TEXT, error TEXT)""")
            conn.execute("""CREATE TABLE IF NOT EXISTS goal_lessons (
                id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp REAL, game_id TEXT, goal_id TEXT,
                first_check_id INTEGER, second_check_id INTEGER, evidence TEXT,
                verification_source TEXT NOT NULL DEFAULT 'two_visual_model_checks',
                CHECK(first_check_id != second_check_id))""")
            conn.execute("""CREATE TABLE IF NOT EXISTS play_lessons (
                id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp REAL, game_id TEXT,
                first_check_id INTEGER, second_check_id INTEGER, evidence TEXT,
                verification_source TEXT NOT NULL DEFAULT 'two_visual_model_checks',
                CHECK(first_check_id != second_check_id))""")
            conn.execute("""CREATE TABLE IF NOT EXISTS failed_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp REAL, game_id TEXT, session_id TEXT,
                step INTEGER, phase TEXT, goal_id TEXT, error TEXT, metadata TEXT)""")
            from .menu_memory import MenuMemory
            MenuMemory.create_schema(conn)
            conn.commit()

    @staticmethod
    def compute_state_hash(img: Image.Image) -> str:
        """Compute a fast difference-hash (dHash) for screen state clustering."""
        resized = img.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
        arr = np.array(resized)
        diff = arr[:, 1:] > arr[:, :-1]
        bits = "".join(["1" if b else "0" for b in diff.flatten()])
        return hex(int(bits, 2))[2:].zfill(16)

    @staticmethod
    def compute_screen_delta(img1: Image.Image, img2: Image.Image) -> float:
        """Calculate normalized mean absolute pixel difference between two consecutive frames."""
        gray1 = np.array(img1.convert("L").resize((64, 64), Image.Resampling.BILINEAR), dtype=float)
        gray2 = np.array(img2.convert("L").resize((64, 64), Image.Resampling.BILINEAR), dtype=float)
        diff = np.abs(gray1 - gray2).mean() / 255.0
        return float(diff)

    def record_experience(
        self,
        timestamp: float,
        state_hash: str,
        action_type: str,
        x: Optional[int],
        y: Optional[int],
        key: Optional[str],
        screen_delta: float,
        effective: bool,
        commentary: str,
        intent: str,
        *,
        game_id: str = "legacy",
        next_state_hash: Optional[str] = None,
        duration: Optional[float] = None,
        x2: Optional[int] = None,
        y2: Optional[int] = None,
        observation: str = "",
        session_id: str = "",
        visual_signature: Optional[str] = None,
        learnable: bool = True,
        goal_id: Optional[str] = None,
        outcome_status: str = "unverified",
        assessment: Optional[Dict] = None,
        action_source: str = "vlm",
        confidence: Optional[float] = None,
        timings: Optional[Dict] = None,
        policy_version: int = 1,
    ):
        with self._connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO experiences (
                    timestamp, state_hash, action_type, x, y, key,
                    screen_delta, effective, commentary, intent,
                    game_id, next_state_hash, duration, x2, y2, observation,
                    session_id, visual_signature, learnable
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp, state_hash, action_type, x, y, key,
                    screen_delta, 1 if effective else 0, commentary, intent,
                    game_id, next_state_hash, duration, x2, y2, observation,
                    session_id, visual_signature, int(learnable),
                )
            )
            experience_id = cursor.lastrowid
            conn.execute("UPDATE experiences SET goal_id=?, outcome_status=?, assessment_json=?, "
                         "action_source=?, confidence=?, timings_json=?, policy_version=? WHERE id=?",
                         (goal_id, outcome_status, json.dumps(assessment) if assessment else None,
                          action_source, confidence, json.dumps(timings) if timings else None,
                          policy_version, experience_id))

            # If it was an ineffective click, track dead click region
            if game_id == "legacy" and action_type == "click" and not effective and x is not None and y is not None:
                # Snap to grid of 50 to avoid exact pixel redundancy
                grid_x = (x // 50) * 50
                grid_y = (y // 50) * 50
                conn.execute(
                    """
                    INSERT INTO dead_clicks (state_hash, x, y, fail_count)
                    VALUES (?, ?, ?, 1)
                    ON CONFLICT(state_hash, x, y) DO UPDATE SET fail_count = fail_count + 1
                    """,
                    (state_hash, grid_x, grid_y)
                )
            conn.commit()
            return experience_id

    def load_supervisor(self, game_id: str) -> Optional[Dict]:
        with self._connection() as conn:
            row = conn.execute("SELECT value FROM agent_state WHERE game_id=?", (game_id,)).fetchone()
        return json.loads(row[0]) if row else None

    def save_supervisor(self, game_id: str, state: Dict):
        with self._connection() as conn:
            conn.execute("INSERT INTO agent_state VALUES (?,?) ON CONFLICT(game_id) DO UPDATE SET value=excluded.value",
                         (game_id, json.dumps(state, ensure_ascii=False)))

    def record_visual_check(self, timestamp, game_id, session_id, step, phase, goal_id, assessment, error=""):
        with self._connection() as conn:
            return conn.execute("INSERT INTO visual_checks(timestamp,game_id,session_id,step,phase,goal_id,assessment,error) "
                                "VALUES (?,?,?,?,?,?,?,?)", (timestamp,game_id,session_id,step,phase,goal_id,
                                                           json.dumps(assessment, ensure_ascii=False),error)).lastrowid

    def record_goal_lesson(self, timestamp, game_id, goal_id, first_check_id, second_check_id, evidence):
        if first_check_id == second_check_id or first_check_id is None or second_check_id is None:
            raise ValueError("Two distinct post-action visual checks are required")
        if evidence.get("status") != "visual_confirmed" or not evidence.get("completed") or evidence.get("goal_id") != goal_id:
            raise ValueError("A completed goal with provenance is required")
        from .supervision import GoalSupervisor
        predicate = GoalSupervisor(include_resource_goal=False)
        predicate.set_goal(goal_id, evidence.get("target_item", "log"))
        predicate.baseline = evidence.get("baseline")
        with self._connection() as conn:
            for check_id in (first_check_id, second_check_id):
                row = conn.execute("SELECT game_id,goal_id,phase,assessment,error FROM visual_checks WHERE id=?",
                                   (check_id,)).fetchone()
                if not row or row[:3] != (game_id, goal_id, "post") or row[4]:
                    raise ValueError("Visual confirmation does not match this game and goal")
                from .perception import VisualAssessment
                assessment = VisualAssessment.model_validate_json(row[3])
                if (not assessment.usable or assessment.result != "progress"
                        or not predicate._met(assessment.after)):
                    raise ValueError("Uncertain assessment cannot become a confirmed lesson")
            conn.execute("INSERT INTO goal_lessons(timestamp,game_id,goal_id,first_check_id,second_check_id,evidence) "
                         "VALUES (?,?,?,?,?,?)", (timestamp,game_id,goal_id,first_check_id,second_check_id,
                                                json.dumps(evidence, ensure_ascii=False)))

    def record_failure(self, timestamp, game_id, session_id, step, phase, goal_id, error, metadata):
        with self._connection() as conn:
            conn.execute("INSERT INTO failed_attempts(timestamp,game_id,session_id,step,phase,goal_id,error,metadata) "
                         "VALUES (?,?,?,?,?,?,?,?)", (timestamp,game_id,session_id,step,phase,goal_id,error,
                                                    json.dumps(metadata, ensure_ascii=False)))

    def record_play_lesson(self, timestamp, game_id, event):
        from .perception import VisualAssessment
        from .supervision import PlaySupervisor
        first, second = event.get("first_check_id"), event.get("second_check_id")
        if not first or not second or first == second or not event.get("verified_change"):
            raise ValueError("A checked outcome needs two distinct visual checks")
        with self._connection() as conn:
            checks = []
            for check_id in (first, second):
                row = conn.execute("SELECT game_id,goal_id,phase,assessment,error FROM visual_checks WHERE id=?",
                                   (check_id,)).fetchone()
                if not row or row[:3] != (game_id,"free_play","post") or row[4]:
                    raise ValueError("Visual checks must belong to this continuous-play profile")
                assessment = VisualAssessment.model_validate_json(row[3])
                if not assessment.usable or assessment.result != "progress":
                    raise ValueError("Uncertain/regressed outcomes cannot be confirmed")
                checks.append(assessment)
            if (PlaySupervisor._candidate(checks[0]) != event.get("change")
                    or not PlaySupervisor._still_visible(event["change"], checks[1].after)):
                raise ValueError("Visual evidence does not support this outcome")
            conn.execute("INSERT INTO play_lessons(timestamp,game_id,first_check_id,second_check_id,evidence) "
                         "VALUES (?,?,?,?,?)", (timestamp,game_id,first,second,json.dumps(event,ensure_ascii=False)))

    @staticmethod
    def compute_visual_signature(img: Image.Image) -> str:
        """Color thumbnail plus aspect ratio guards against dHash collisions."""
        thumb = img.convert("RGB").resize((8, 8), Image.Resampling.BILINEAR)
        return f"{img.width / img.height:.4f}:" + thumb.tobytes().hex()

    @staticmethod
    def _visually_similar(left: Optional[str], right: Optional[str]) -> bool:
        if not left or not right:
            return True  # Compatibility with manually recorded/legacy episodes.
        try:
            ratio_a, pixels_a = left.split(":")
            ratio_b, pixels_b = right.split(":")
            a, b = bytes.fromhex(pixels_a), bytes.fromhex(pixels_b)
            return (len(a) == len(b) == 192
                    and abs(float(ratio_a) / float(ratio_b) - 1) <= 0.05
                    and sum(abs(x - y) for x, y in zip(a, b)) / (192 * 255) <= 0.08)
        except (ValueError, ZeroDivisionError):
            return False

    @staticmethod
    def _hash_distance(left: str, right: str) -> int:
        try:
            if len(left) != 16 or len(right) != 16:
                return 65
            return (int(left, 16) ^ int(right, 16)).bit_count()
        except (ValueError, TypeError):
            return 65

    def get_recent_experiences(self, game_id: str, limit: int = 10) -> List[Dict]:
        with self._connection() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT *, screen_delta AS delta FROM experiences "
                "WHERE game_id = ? ORDER BY id DESC LIMIT ?", (game_id, limit)
            ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def _matching_experiences(self, state_hash: str, game_id: str,
                              visual_signature: Optional[str] = None,
                              max_distance: int = 4) -> List[Dict]:
        with self._connection() as conn:
            conn.row_factory = sqlite3.Row
            conn.create_function("hash_distance", 2, self._hash_distance)
            hashes = [row[0] for row in conn.execute(
                "SELECT state_hash FROM experiences WHERE game_id = ? AND learnable = 1 "
                "AND action_type NOT IN ('look', 'wait') "
                "GROUP BY state_hash HAVING hash_distance(state_hash, ?) <= ? "
                "ORDER BY hash_distance(state_hash, ?) LIMIT 8",
                (game_id, state_hash, max_distance, state_hash)
            )]
            if not hashes:
                return []
            # Bound each state's contribution; old exact matches cannot be
            # crowded out by newer observations of neighboring screens.
            result = []
            for matched_hash in hashes:
                rows = conn.execute(
                    "SELECT * FROM experiences WHERE game_id = ? AND state_hash = ? "
                    "AND learnable = 1 AND action_type NOT IN ('look', 'wait') "
                    "ORDER BY id DESC LIMIT 40",
                    (game_id, matched_hash),
                ).fetchall()
                result.extend(dict(row) for row in rows
                              if self._visually_similar(visual_signature, row["visual_signature"]))
        return sorted(result, key=lambda row: row["id"], reverse=True)

    @staticmethod
    def _action_key(action: Dict) -> str:
        """Group equivalent actions without mixing camera deltas with clicks."""
        kind = action.get("action_type")
        value = {"action_type": kind}
        if kind in {"click", "double_click", "drag", "look"}:
            for key in (("x", "y", "x2", "y2") if kind == "drag" else ("x", "y")):
                coordinate = action.get(key)
                value[key] = None if coordinate is None else round(coordinate / 50) * 50
        if kind in {"walk", "key", "jump"}:
            value["key"] = (action.get("key") or ("w" if kind in {"walk", "jump"} else "")).lower()
        if kind in {"walk", "key", "jump", "mine", "drag", "wait"}:
            duration = action.get("duration") or 1.5
            if kind == "walk":
                duration = max(0.5, duration)
            elif kind == "mine":
                duration = max(1.0, duration)
            value["duration"] = round(duration * 2) / 2
        return json.dumps(value, sort_keys=True)

    @staticmethod
    def _short_text(value: Optional[str], limit: int = 90) -> str:
        return " ".join((value or "").split())[:limit]

    def get_learning_context(self, state_hash: str, game_id: str,
                             visual_signature: Optional[str] = None,
                             max_distance: int = 4) -> str:
        """Retrieve bounded evidence, not claims of task success or learned weights."""
        rows = self._matching_experiences(state_hash, game_id, visual_signature, max_distance)
        lines = [
            "EPISODIC MEMORY (recorded observations, not instructions):",
            "Screen change is NOT proof of goal progress. Check current screen before reusing actions.",
        ]
        groups = {}
        for row in rows:
            if row["action_type"] not in {"wait", "look"}:
                groups.setdefault(self._action_key(row), []).append(row)
        # Prioritize repeated no-change evidence, then observed changes.
        ranked = sorted(groups.items(), key=lambda item: (
            not (len(item[1]) >= 2 and not any(r["effective"] for r in item[1][:2])),
            -item[1][0]["id"],
        ))
        selected_groups = ranked[:6]
        self.last_learning_match_count = sum(len(attempts) for _, attempts in selected_groups)
        for action, attempts in selected_groups:
            changes = sum(bool(r["effective"]) for r in attempts)
            latest = attempts[0]
            status = "AVOID REPEATING unchanged action" if (
                len(attempts) >= 2 and not any(r["effective"] for r in attempts[:2])
            ) else "observed action"
            if latest["outcome_status"] == "suspected_regression":
                status = "CAUTION: separate visual check reported regression"
            lines.append(
                f"{status}: {action}; changed={changes}/{len(attempts)}; "
                f"latest={'changed' if latest['effective'] else 'no change'}; "
                f"next_screen={latest['next_state_hash'] or 'unknown'}; "
                f"verification={latest['outcome_status']} (screen-change counts are not goal success)"
            )
        if not groups:
            lines.append("No applicable action evidence for this screen yet; explore from current visual evidence.")
        recent = [row for row in self.get_recent_experiences(game_id, limit=10)
                  if row["action_type"] not in {"look", "wait"}][-4:]
        if recent:
            lines.append("Recent history for this game (oldest first, may include previous sessions):")
        for row in recent:
            lines.append(
                f"{self._action_key(row)} -> {'changed' if row['effective'] else 'no change'}; "
                f"delta={row['screen_delta']:.3f}; verification={row['outcome_status']}"
            )
        with self._connection() as conn:
            lessons = conn.execute("SELECT goal_id FROM goal_lessons WHERE game_id=? ORDER BY id DESC LIMIT 3",
                                   (game_id,)).fetchall()
        for (goal_id,) in lessons:
            lines.append(f"Historical visual state: {goal_id}; not a prescribed task; two model checks, not proof of causal skill.")
        with self._connection() as conn:
            outcomes = conn.execute("SELECT evidence FROM play_lessons WHERE game_id=? ORDER BY id DESC LIMIT 4",
                                    (game_id,)).fetchall()
        for (evidence,) in outcomes:
            try:
                outcome = json.loads(evidence)
            except (TypeError, ValueError):
                continue
            change = outcome.get("change") if isinstance(outcome, dict) else None
            if not isinstance(change, dict):
                continue
            lines.append("Previously checked play outcome (not proof of a causal skill): " +
                         json.dumps(change, ensure_ascii=False))
            candidate_action = outcome.get("candidate_action")
            if isinstance(candidate_action, dict):
                action_json = json.dumps(candidate_action, ensure_ascii=False, sort_keys=True)
                lines.append("Action immediately before the first visual candidate "
                             "(temporal association only; not proof of cause): " + action_json[:320])
        return "\n".join(lines)

    def get_repetition_warning(self, state_hash: str, game_id: str, action: Dict,
                               visual_signature: Optional[str] = None) -> str:
        """Replan only on exact-screen evidence of two latest unchanged attempts."""
        if action.get("action_type") == "wait":
            return ""
        key = self._action_key(action)
        attempts = [row for row in self._matching_experiences(
            state_hash, game_id, visual_signature, max_distance=0
        ) if self._action_key(row) == key]
        if len(attempts) >= 2 and not any(row["effective"] for row in attempts[:2]):
            return (f"REPLAN: {key} had no observable effect in its last two attempts on this screen. "
                    "Choose a different action or target based on the current image; do not repeat it.")
        return ""

    def get_recent_failure_hint(self, game_id: str, visual_signature: str,
                                action: Optional[Dict] = None) -> str:
        """Catch consecutive UI failures despite animated-background dHash drift.

        Require the last two real actions to be clicks at the same target and
        thumbnails similar to the current scene. Never apply this to gameplay
        movement, delayed actions, or successes. Waits cannot erase evidence.
        """
        recent = [row for row in self.get_recent_experiences(game_id, 10)
                  if row["learnable"] and row["action_type"] != "wait"]
        if len(recent) < 2:
            return ""
        previous, latest = recent[-2:]
        if (latest["action_type"] not in {"click", "double_click"}
                or self._action_key(previous) != self._action_key(latest)
                or previous["effective"] or latest["effective"]
                or not all(row["visual_signature"] and self._visually_similar(
                    visual_signature, row["visual_signature"]) for row in (previous, latest))):
            return ""
        key = self._action_key(latest)
        if action is not None and self._action_key(action) != key:
            return ""
        return (f"\nREPLAN: recent repeated {key} produced no observable effect on this similar screen, "
                "even though the animated background changed its hash. Inspect selection/highlight and "
                "button state; try a different target or an appropriate keyboard action, not the same click.")

    def get_navigation_hint(self, game_id: str, visual_signature: str,
                            max_delta: float = 0.015, action: Optional[Dict] = None) -> str:
        recent = [row for row in self.get_recent_experiences(game_id, 8)
                  if row["learnable"] and row["action_type"] != "wait"]
        if len(recent) < 2:
            return ""
        previous, latest = recent[-2:]
        direction = latest.get("key") or "w"
        if (any(row["action_type"] != "walk" or (row.get("key") or "w") != direction
                or row["screen_delta"] > max_delta or row["effective"]
                or not row["visual_signature"]
                or not self._visually_similar(visual_signature, row["visual_signature"])
                for row in (previous, latest))):
            return ""
        if action is not None and not (
            action.get("action_type") == "walk" and (action.get("key") or "w") == direction
        ):
            return ""
        return (f"\nNAVIGATION REPLAN: walking '{direction}' twice barely changed the view. "
                "Possible obstacle, not proof of progress. Inspect the ground/obstacle at the crosshair. "
                "Use jump with key='w' for a low step, look with a direction/amount to inspect a route, "
                "or choose another direction; do not walk into the same obstacle again.")

    def get_dead_click_hints(self, state_hash: str) -> str:
        """Return warning string about inactive regions on this screen."""
        with self._connection() as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT x, y, fail_count FROM dead_clicks WHERE state_hash = ? AND fail_count >= 2",
                (state_hash,)
            )
            rows = cur.fetchall()
            if not rows:
                return ""
            spots = [f"near ({r[0]}, {r[1]}) [failed {r[2]}x]" for r in rows[:4]]
            return "Avoid dead-clicking regions that had no effect: " + ", ".join(spots)

    def check_stuck_loop(self, recent_history: List[Dict]) -> Optional[str]:
        """Detect if agent is repeating same actions in a loop without state advancement."""
        if len(recent_history) < 2:
            return None
        
        last = recent_history[-1]
        prev = recent_history[-2]
        
        # Check if last 2 actions targeted nearly identical coordinates or repeated failed intent
        if last.get("action_type") in ("click", "double_click") and prev.get("action_type") in ("click", "double_click"):
            lx, ly = last.get("x") or 0, last.get("y") or 0
            px, py = prev.get("x") or 0, prev.get("y") or 0
            dist = ((lx - px)**2 + (ly - py)**2)**0.5
            
            if dist < 90 and not last.get("effective", False):
                return (
                    f"⚠️ STUCK ALERT: You have clicked near ({lx}, {ly}) multiple times, "
                    "with no observable screen change on the latest attempt. "
                    "Inspect the current image and try a different target or action; "
                    "do not assume a disabled button or invent a target coordinate."
                )
                
        # Check if stuck in passive wait loop
        if last.get("action_type") == "wait" and prev.get("action_type") == "wait":
            return (
                "⚠️ STUCK ALERT: You have waited multiple steps in a row! "
                "If the current image is not a loading screen, choose an action suited "
                "to the visible menu or gameplay rather than waiting again."
            )
        return None

    def get_stats(self, game_id: Optional[str] = None) -> Dict:
        """Return high-level summary of what the bot has experienced and learned."""
        with self._connection() as conn:
            cur = conn.cursor()
            where = " WHERE game_id = ?" if game_id is not None else ""
            params = (game_id,) if game_id is not None else ()
            cur.execute("SELECT COUNT(*), SUM(effective) FROM experiences" + where, params)
            total, effective = cur.fetchone()
            total = total or 0
            effective = effective or 0
            
            cur.execute("SELECT COUNT(DISTINCT state_hash) FROM experiences" + where, params)
            unique_states = cur.fetchone()[0] or 0
            cur.execute("SELECT COUNT(*) FROM (SELECT state_hash, next_state_hash FROM experiences" +
                        (where + " AND " if where else " WHERE ") +
                        "learnable = 1 AND effective = 1 AND next_state_hash IS NOT NULL "
                        "AND next_state_hash != state_hash GROUP BY game_id, state_hash, next_state_hash)", params)
            transitions = cur.fetchone()[0]
            lesson_where = " WHERE game_id=?" if game_id is not None else ""
            confirmed = conn.execute("SELECT COUNT(*) FROM goal_lessons" + lesson_where, params).fetchone()[0]
            failures = conn.execute("SELECT COUNT(*) FROM failed_attempts" + lesson_where, params).fetchone()[0]
            outcomes = conn.execute("SELECT COUNT(*) FROM play_lessons" + lesson_where, params).fetchone()[0]
            
            return {
                "total_actions": total,
                "effective_actions": effective,
                "success_rate": (effective / total) if total > 0 else 0.0,
                "unique_screens_discovered": unique_states,
                "observed_transitions": transitions,
                "visually_confirmed_goals": confirmed,
                "failed_attempts": failures,
                "checked_play_outcomes": outcomes,
            }
