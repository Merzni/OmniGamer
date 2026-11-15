"""Remember verified menu transitions and replay one visually matched step at a time."""
import json
import time
import re
import sqlite3
from collections import deque
import numpy as np
from PIL import Image


def menu_key(scene, menu_id="unknown", menu_state="default"):
    if scene in {"gameplay", "loading"}:
        return scene
    if scene != "menu":
        return None
    def clean(value):
        return re.sub(r"[^\w-]+","_",str(value).strip().casefold())[:64]
    identity = clean(menu_id)
    aliases = {"main":"main_menu","world_selection":"world_select","select_world":"world_select",
               "pause":"pause_menu","главное_меню":"main_menu"}
    identity = aliases.get(identity,identity)
    if identity in {"", "unknown", "menu"}:
        return None
    state = clean(menu_state) or "default"
    state = {"selected":"world_selected","none_selected":"no_selection"}.get(state,state)
    if state in {"unknown","unknown_selection"}:
        return None
    return f"menu:{identity}:{state}"


def facts_key(facts):
    return menu_key(facts.scene,facts.menu_id,facts.menu_state)


class MenuMemory:
    def __init__(self, memory):
        self.memory = memory

    @staticmethod
    def create_schema(conn):
        conn.execute("""CREATE TABLE IF NOT EXISTS menu_actions (
            id INTEGER PRIMARY KEY AUTOINCREMENT, game_id TEXT, source_key TEXT, destination_key TEXT,
            action_json TEXT, anchor_json TEXT, target_label TEXT, successes INTEGER DEFAULT 1,
            failures INTEGER DEFAULT 0, disabled INTEGER DEFAULT 0, first_check_id INTEGER,
            confirm_check_id INTEGER, last_used REAL, created REAL)""")
        conn.execute("CREATE INDEX IF NOT EXISTS ix_menu_game_source ON menu_actions(game_id,source_key,disabled)")

    @staticmethod
    def anchor(frame, action):
        x = action.get("x") if action.get("x") is not None else 500
        y = action.get("y") if action.get("y") is not None else 250
        w,h=frame.size
        box=(max(0,int((x-130)*w/1000)),max(0,int((y-35)*h/1000)),
             min(w,int((x+130)*w/1000)),min(h,int((y+35)*h/1000)))
        patch=frame.convert("RGB").crop(box).resize((96,32),Image.Resampling.BILINEAR)
        rgb=np.array(patch,dtype=np.float32)
        gray=rgb.mean(axis=2)
        edges=np.zeros(gray.shape,dtype=bool)
        edges[:,1:] |= np.abs(gray[:,1:]-gray[:,:-1])>25
        edges[1:,:] |= np.abs(gray[1:,:]-gray[:-1,:])>25
        return {"rgb":np.array(patch.resize((12,4))).tobytes().hex(),
                "edges":np.packbits(edges).tobytes().hex(),"aspect":w/h}

    @staticmethod
    def matches(left,right):
        if abs(left["aspect"]/right["aspect"]-1)>.04:
            return False
        a=np.frombuffer(bytes.fromhex(left["rgb"]),dtype=np.uint8).astype(float)
        b=np.frombuffer(bytes.fromhex(right["rgb"]),dtype=np.uint8).astype(float)
        if np.abs(a-b).mean()/255>.06:
            return False
        ea=np.unpackbits(np.frombuffer(bytes.fromhex(left["edges"]),dtype=np.uint8)).astype(bool)
        eb=np.unpackbits(np.frombuffer(bytes.fromhex(right["edges"]),dtype=np.uint8)).astype(bool)
        union=(ea|eb).sum()
        return union>=12 and (ea&eb).sum()/union>=.65

    def remember(self, game_id, source_frame, action, first_id, confirm_id):
        from .perception import VisualAssessment
        if first_id==confirm_id:
            raise ValueError("A menu action requires two different checks")
        if action.get("action_type") not in {"click","double_click","key"}:
            return None
        with self.memory._connection() as conn:
            checks=[]
            for check_id,phase in ((first_id,"post"),(confirm_id,"menu_confirm")):
                row=conn.execute("SELECT game_id,phase,assessment,error FROM visual_checks WHERE id=?",(check_id,)).fetchone()
                if not row or row[0]!=game_id or row[1]!=phase or row[3]:
                    raise ValueError("Menu evidence belongs to another game/phase")
                check=VisualAssessment.model_validate_json(row[2])
                if not check.usable or check.result=="regressed":
                    return None
                checks.append(check)
            initial,confirmed=checks
            if initial.menu_action_succeeded is not True:
                return None
            source=facts_key(initial.before)
            target=facts_key(initial.after)
            current=facts_key(confirmed.after)
            if not source or not source.startswith("menu:") or not target or source==target:
                return None
            if target=="loading" and current=="gameplay": target="gameplay"
            if target!=current:
                return None
            allowed={key:action.get(key) for key in ("action_type","x","y","x2","y2","key","duration",
                                                    "intent","target_label","expected_result")}
            encoded=json.dumps(allowed,ensure_ascii=False,sort_keys=True)
            anchor=json.dumps(self.anchor(source_frame,allowed))
            existing=conn.execute("SELECT id FROM menu_actions WHERE game_id=? AND source_key=? AND action_json=? "
                                  "AND destination_key=? ORDER BY id DESC LIMIT 1",(game_id,source,encoded,target)).fetchone()
            if existing:
                conn.execute("UPDATE menu_actions SET successes=successes+1,disabled=0,anchor_json=?,"
                             "first_check_id=?,confirm_check_id=? WHERE id=?",(anchor,first_id,confirm_id,existing[0]))
                return existing[0]
            return conn.execute("INSERT INTO menu_actions(game_id,source_key,destination_key,action_json,anchor_json,"
                                "target_label,first_check_id,confirm_check_id,created) VALUES (?,?,?,?,?,?,?,?,?)",
                                (game_id,source,target,encoded,anchor,action.get("target_label",""),first_id,confirm_id,time.time())).lastrowid

    def recall(self, game_id, frame, observation):
        source=menu_key(observation.scene_type,observation.menu_id,observation.menu_state)
        if not source or observation.visual_confidence<.85:
            return None
        with self.memory._connection() as conn:
            conn.row_factory=sqlite3.Row
            rows=[dict(r) for r in conn.execute("SELECT * FROM menu_actions WHERE game_id=? AND disabled=0 "
                                               "ORDER BY successes DESC,id DESC LIMIT 200",(game_id,))]
        # Prefer an already learned route back into gameplay, never execute it blindly.
        graph={}
        for row in rows:
            label=row["target_label"].casefold()
            if any(word in label for word in ("delete","quit","exit","удал","выход")):
                continue
            edges=graph.setdefault(row["source_key"],[])
            if len(edges)<4:edges.append(row)
        queue=deque([(source,[],{source})]);routes=[]
        while queue:
            node,path,visited=queue.popleft()
            if len(path)>=4:continue
            for edge in graph.get(node,[]):
                target=edge["destination_key"]
                if target in visited:continue
                route=path+[edge]
                if target=="gameplay":routes.append(route)
                else:queue.append((target,route,visited|{target}))
        candidates=routes or [[edge] for edge in graph.get(source,[])]
        for route in candidates:
            edge=route[0];action=json.loads(edge["action_json"])
            if self.matches(json.loads(edge["anchor_json"]),self.anchor(frame,action)):
                return {"id":edge["id"],"action":action,"source":source,"destination":edge["destination_key"],
                        "successes":edge["successes"],"route":[e["destination_key"] for e in route]}
        return None

    def invalidate(self, action_id):
        with self.memory._connection() as conn:
            conn.execute("UPDATE menu_actions SET failures=failures+1,disabled=1,last_used=? WHERE id=?",
                         (time.time(),action_id))
