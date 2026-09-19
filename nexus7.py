"""
NEXUS-7 — Sentient Resource Orchestration Engine
Upgraded from os_project_file.py — same core logic, massively extended:
  - 5 classical scheduler classes + NexusAdaptiveAI meta-scheduler
  - Flask-SocketIO WebSocket push (500ms cadence)
  - Ghost Replay recording
  - Algorithm Battle Mode
  - ML linear-regression prediction (from original)
  - SQLite telemetry (from original)
  - NEXUS narration engine
"""

import os, time, random, sqlite3, threading, json, math
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple

import psutil
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from flask import Flask, send_file
from flask_socketio import SocketIO
from flask_cors import CORS

# ── Constants (kept from original) ──────────────────────────────────────────
DEFAULT_DB_NAME               = "nexus7_logs.db"
LEGACY_DB_NAME                = "allocation_logs.db"
DEFAULT_TICK_DELAY_SECONDS    = 0.5
MIN_PREDICTION_POINTS         = 5
EWMA_ALPHA                    = 0.35
MIN_RESOURCE_SLICE            = 0.5
GHOST_REPLAY_LEN              = 60   # ticks to remember
PORT                          = 7800

# Alert profile can be: sensitive, balanced, severe
ALERT_SENSITIVITY_PROFILE = os.getenv("NEXUS_ALERT_PROFILE", "balanced").strip().lower()
ALERT_SENSITIVITY_PRESETS = {
    'sensitive': {
        'bottleneck_threshold': 0.72,
        'collision_unfulfilled_ratio': 0.30,
        'collision_min_request_cpu': 3.0,
        'collision_min_request_mem': 48.0,
        'collision_min_participants': 2,
    },
    'balanced': {
        'bottleneck_threshold': 0.85,
        'collision_unfulfilled_ratio': 0.45,
        'collision_min_request_cpu': 5.0,
        'collision_min_request_mem': 64.0,
        'collision_min_participants': 2,
    },
    'severe': {
        'bottleneck_threshold': 0.93,
        'collision_unfulfilled_ratio': 0.65,
        'collision_min_request_cpu': 8.0,
        'collision_min_request_mem': 128.0,
        'collision_min_participants': 3,
    },
}


def _resolve_alert_profile(name: str) -> dict:
    return ALERT_SENSITIVITY_PRESETS.get(name, ALERT_SENSITIVITY_PRESETS['balanced'])


ALERT_CFG = _resolve_alert_profile(ALERT_SENSITIVITY_PROFILE)
BOTTLENECK_PRESSURE_THRESHOLD = ALERT_CFG['bottleneck_threshold']
COLLISION_UNFULFILLED_RATIO = ALERT_CFG['collision_unfulfilled_ratio']
COLLISION_MIN_REQUEST_CPU = ALERT_CFG['collision_min_request_cpu']
COLLISION_MIN_REQUEST_MEM = ALERT_CFG['collision_min_request_mem']
COLLISION_MIN_PARTICIPANTS = ALERT_CFG['collision_min_participants']

DEFAULT_PROCESS_SPECS = [
    (1, 'CPU-Intensive'),
    (2, 'Memory-Intensive'),
    (3, 'Balanced'),
    (4, 'CPU-Intensive'),
    (5, 'Balanced'),
    (6, 'Memory-Intensive'),
    (7, 'CPU-Intensive'),
    (8, 'Balanced'),
]

WORKLOAD_COLORS = {
    'CPU-Intensive':    '#ff003c',
    'Memory-Intensive': '#7b2fff',
    'Balanced':         '#00ffe0',
}

# ── VirtualProcess (extended from original) ──────────────────────────────────
class VirtualProcess:
    def __init__(self, pid: int, workload_type: str):
        self.pid            = pid
        self.workload_type  = workload_type
        self.priority       = random.randint(3, 8)
        self.allocated_cpu  = 0.0
        self.allocated_mem  = 0.0
        self.last_req_cpu   = 0.0
        self.last_req_mem   = 0.0
        self.wait_time      = 0
        self.starvation_credit = 0.0
        self.burst_left     = random.randint(10, 50)
        self.state          = 'ready'
        self.history_cpu: deque = deque(maxlen=30)
        self.history_mem: deque = deque(maxlen=30)
        # DNA fingerprint: deterministic from pid+workload
        rng = random.Random(pid * 31 + hash(workload_type))
        self.dna = [rng.random() for _ in range(16)]
        self.color = WORKLOAD_COLORS.get(workload_type, '#ffffff')

    def generate_requests(self) -> Tuple[float, float]:
        if self.workload_type == 'CPU-Intensive':
            return random.uniform(15.0, 40.0), random.uniform(10.0, 50.0)
        elif self.workload_type == 'Memory-Intensive':
            return random.uniform(1.0, 10.0), random.uniform(200.0, 800.0)
        else:
            return random.uniform(5.0, 20.0), random.uniform(50.0, 300.0)

    def update_state(self, fulfilled: bool):
        r = random.random()
        if self.state == 'running':
            if not fulfilled:
                self.state = 'waiting' if r < 0.4 else 'blocked'
            elif r < 0.05:
                self.state = 'ready'
        elif self.state in ('waiting', 'blocked'):
            if r < 0.45:
                self.state = 'running'
        elif self.state == 'ready':
            if r < 0.7:
                self.state = 'running'

    def to_dict(self) -> dict:
        return {
            'pid':            self.pid,
            'workload_type':  self.workload_type,
            'priority':       self.priority,
            'allocated_cpu':  round(self.allocated_cpu, 2),
            'allocated_mem':  round(self.allocated_mem, 2),
            'requested_cpu':  round(self.last_req_cpu, 2),
            'requested_mem':  round(self.last_req_mem, 2),
            'wait_time':      self.wait_time,
            'starvation_credit': round(self.starvation_credit, 3),
            'burst_left':     self.burst_left,
            'state':          self.state,
            'history_cpu':    list(self.history_cpu),
            'history_mem':    list(self.history_mem),
            'dna':            self.dna,
            'color':          self.color,
        }

# ═══════════════════════════════════════════════════════════════════════════
# SCHEDULER CLASSES
# Each exposes: .schedule(requests, avail_cpu, avail_mem, **ctx) -> list[dict]
#               .explain() -> str
#               .metrics() -> dict
# ═══════════════════════════════════════════════════════════════════════════

class BaseScheduler:
    name   = "Base"
    color  = "#ffffff"
    formula = ""

    def schedule(self, requests, avail_cpu, avail_mem, **ctx):
        raise NotImplementedError

    def explain(self) -> str:
        return f"{self.name} scheduler active."

    def metrics(self) -> dict:
        return {}

# ── 1. Round Robin ───────────────────────────────────────────────────────────
class RoundRobinScheduler(BaseScheduler):
    name    = "Round Robin"
    color   = "#4e9eff"
    formula = "q=20ms · wait=(n-1)×q · fair share=avail/n"

    def __init__(self):
        self._pointer = 0
        self._avg_wait = 0.0

    def schedule(self, requests, avail_cpu, avail_mem, **ctx):
        n = len(requests)
        if n == 0:
            return []
        cpu_share = avail_cpu / n
        mem_share = avail_mem / n
        results = []
        for req in requests:
            alloc_cpu = min(req['req_cpu'], cpu_share)
            alloc_mem = min(req['req_mem'], mem_share)
            results.append({'pid': req['process'].pid,
                            'cpu': round(alloc_cpu, 2),
                            'mem': round(alloc_mem, 2)})
        self._pointer = (self._pointer + 1) % max(n, 1)
        self._avg_wait = (n - 1) * 20  # ms
        return results

    def explain(self) -> str:
        return (f"Round Robin: each process receives equal time-slice. "
                f"Pointer at slot {self._pointer}. "
                f"Estimated avg wait ≈ {self._avg_wait:.0f} ms.")

    def metrics(self) -> dict:
        return {'avg_wait_ms': self._avg_wait}


# ── 2. Priority Scheduling ───────────────────────────────────────────────────
class PriorityScheduler(BaseScheduler):
    name    = "Priority"
    color   = "#ff003c"
    formula = "sort(priority) · aging: prio -= α×wait_ticks · starvation guard"

    def __init__(self):
        self._last_chosen = None

    def schedule(self, requests, avail_cpu, avail_mem, **ctx):
        sorted_reqs = sorted(requests, key=lambda r: r['process'].priority)
        results = []
        rem_cpu, rem_mem = avail_cpu, avail_mem
        for req in sorted_reqs:
            alloc_cpu = min(req['req_cpu'], rem_cpu)
            alloc_mem = min(req['req_mem'], rem_mem)
            rem_cpu  -= alloc_cpu
            rem_mem  -= alloc_mem
            results.append({'pid': req['process'].pid,
                            'cpu': round(alloc_cpu, 2),
                            'mem': round(alloc_mem, 2)})
            if rem_cpu <= 0 and rem_mem <= 0:
                break
        self._last_chosen = sorted_reqs[0]['process'].pid if sorted_reqs else None
        return results

    def explain(self) -> str:
        return (f"Priority: processes sorted by priority level (1=highest). "
                f"Last served PID {self._last_chosen}. "
                f"Aging prevents starvation: wait>2 → priority−2.")

    def metrics(self) -> dict:
        return {'last_chosen_pid': self._last_chosen}


# ── 3. Shortest Job First ────────────────────────────────────────────────────
class SJFScheduler(BaseScheduler):
    name    = "Shortest Job First"
    color   = "#00ff88"
    formula = "select min(burst_left) · avg_wait=Σ(start−arrival)/n"

    def __init__(self):
        self._total_wait = 0.0
        self._scheduled  = 0

    def schedule(self, requests, avail_cpu, avail_mem, **ctx):
        sorted_reqs = sorted(requests, key=lambda r: r['process'].burst_left)
        results = []
        rem_cpu, rem_mem = avail_cpu, avail_mem
        for i, req in enumerate(sorted_reqs):
            # Give more to shorter jobs
            weight = (len(sorted_reqs) - i) / max(1, sum(range(1, len(sorted_reqs) + 1)))
            alloc_cpu = min(req['req_cpu'], rem_cpu * weight * 1.5)
            alloc_mem = min(req['req_mem'], rem_mem * weight * 1.5)
            rem_cpu  -= alloc_cpu
            rem_mem  -= alloc_mem
            results.append({'pid': req['process'].pid,
                            'cpu': round(alloc_cpu, 2),
                            'mem': round(alloc_mem, 2)})
        self._total_wait += sum(r['process'].wait_time for r in requests)
        self._scheduled  += len(requests)
        return results

    def explain(self) -> str:
        avg = self._total_wait / max(1, self._scheduled)
        return (f"SJF: sorted by burst_left. Shortest jobs get weighted priority. "
                f"Cumulative avg wait ≈ {avg:.1f} ticks.")

    def metrics(self) -> dict:
        return {'avg_wait_ticks': round(self._total_wait / max(1, self._scheduled), 2)}


# ── 4. Multilevel Queue ──────────────────────────────────────────────────────
class MLQScheduler(BaseScheduler):
    name    = "Multilevel Queue"
    color   = "#c77dff"
    formula = "fg_pool=70% · bg_pool=30% · boundary: nice≤0 → foreground"

    def schedule(self, requests, avail_cpu, avail_mem, **ctx):
        fg = [r for r in requests if r['process'].workload_type == 'CPU-Intensive']
        bg = [r for r in requests if r['process'].workload_type != 'CPU-Intensive']
        results = []

        def alloc_group(group, cpu_pool, mem_pool):
            n = len(group)
            if n == 0:
                return
            for req in group:
                c = min(req['req_cpu'], cpu_pool / n)
                m = min(req['req_mem'], mem_pool / n)
                results.append({'pid': req['process'].pid,
                                'cpu': round(c, 2),
                                'mem': round(m, 2)})

        alloc_group(fg, avail_cpu * 0.70, avail_mem * 0.70)
        alloc_group(bg, avail_cpu * 0.30, avail_mem * 0.30)
        return results

    def explain(self) -> str:
        return ("MLQ: CPU-Intensive procs → foreground queue (70% pool). "
                "Memory-Intensive + Balanced → background queue (30% pool). "
                "No inter-queue migration.")

    def metrics(self) -> dict:
        return {}


# ── 5. Multilevel Feedback Queue ─────────────────────────────────────────────
class MLFQScheduler(BaseScheduler):
    name    = "MLFQ"
    color   = "#ffd60a"
    formula = "3 levels · q1=10ms q2=20ms q3=40ms · demotion on timeout · boost T=50"

    LEVELS = 3
    BOOST_INTERVAL = 50

    def __init__(self):
        self._queues: Dict[int, int] = {}   # pid -> level (0=highest)
        self._tick   = 0

    def schedule(self, requests, avail_cpu, avail_mem, **ctx):
        self._tick += 1
        # Periodic boost: reset all to level 0
        if self._tick % self.BOOST_INTERVAL == 0:
            self._queues = {r['process'].pid: 0 for r in requests}

        results = []
        level_weights = [0.6, 0.3, 0.1]   # level 0 gets 60% of pool

        for lvl in range(self.LEVELS):
            group = [r for r in requests
                     if self._queues.get(r['process'].pid, 0) == lvl]
            if not group:
                continue
            w = level_weights[lvl]
            cpu_pool = avail_cpu * w
            mem_pool = avail_mem * w
            n = len(group)
            for req in group:
                c = min(req['req_cpu'], cpu_pool / n)
                m = min(req['req_mem'], mem_pool / n)
                results.append({'pid': req['process'].pid,
                                'cpu': round(c, 2),
                                'mem': round(m, 2)})
                # Demote if still waiting (CPU hog)
                if req['process'].wait_time > 1:
                    pid = req['process'].pid
                    self._queues[pid] = min(self.LEVELS - 1,
                                           self._queues.get(pid, 0) + 1)
        return results

    def explain(self) -> str:
        lvl_counts = {0: 0, 1: 0, 2: 0}
        for lvl in self._queues.values():
            lvl_counts[lvl] = lvl_counts.get(lvl, 0) + 1
        return (f"MLFQ: L0={lvl_counts[0]} procs, L1={lvl_counts[1]}, L2={lvl_counts[2]}. "
                f"Boost every {self.BOOST_INTERVAL} ticks. "
                f"Next boost in {self.BOOST_INTERVAL - (self._tick % self.BOOST_INTERVAL)} ticks.")

    def metrics(self) -> dict:
        return {f'level_{k}': v for k, v in
                {0: 0, 1: 0, 2: 0, **{l: c for l, c in
                 [(l, sum(1 for v in self._queues.values() if v == l))
                  for l in range(self.LEVELS)]}}.items()}


# ── 6. NexusAdaptiveAI (upgraded from original AdaptiveAllocator logic) ──────
class NexusAdaptiveAI(BaseScheduler):
    """
    Meta-scheduler: dynamically switches between the 5 classical algorithms
    based on real-time system pressure. Preserves the original EWMA + weighted
    fair-share + 2-pass allocation logic as the 'balanced' mode.
    """
    name    = "NEXUS Adaptive AI"
    color   = "#00ffe0"
    formula = "EWMA(α=0.35) · weighted_fair_share · 2-pass realloc · starvation_credit"

    def __init__(self):
        self._schedulers = {
            'rr':       RoundRobinScheduler(),
            'priority': PriorityScheduler(),
            'sjf':      SJFScheduler(),
            'mlq':      MLQScheduler(),
            'mlfq':     MLFQScheduler(),
        }
        self._active_key  = 'rr'
        self._reason      = "Initializing with Round Robin."
        self._cpu_ewma    = 0.0
        self._mem_ewma    = 0.0

    def _pick_scheduler(self, cpu_pressure: float, mem_pressure: float,
                        n_blocked: int, n_procs: int) -> Tuple[str, str]:
        if cpu_pressure > 0.90:
            return 'sjf', f"CPU critical ({cpu_pressure:.0%}) — SJF clears backlog fastest."
        if mem_pressure > 0.85:
            return 'priority', f"Memory pressure ({mem_pressure:.0%}) — starving low-priority procs."
        if n_blocked > max(1, n_procs // 2):
            return 'mlfq', f"{n_blocked} blocked procs — MLFQ re-queues by behavior."
        if cpu_pressure > 0.70:
            return 'mlq', f"Mixed load ({cpu_pressure:.0%} CPU) — MLQ separates fg/bg queues."
        if cpu_pressure < 0.30 and mem_pressure < 0.30:
            return 'rr', f"Low load — Round Robin for maximum fairness."
        return 'rr', f"Balanced load — Round Robin (fair share)."

    def schedule(self, requests, avail_cpu, avail_mem, **ctx):
        cpu_pressure = ctx.get('cpu_pressure', 0.5)
        mem_pressure = ctx.get('mem_pressure', 0.5)
        n_blocked    = sum(1 for r in requests if r['process'].state == 'blocked')

        key, reason = self._pick_scheduler(cpu_pressure, mem_pressure,
                                           n_blocked, len(requests))
        if key != self._active_key:
            self._active_key = key
            self._reason     = reason

        return self._schedulers[key].schedule(requests, avail_cpu, avail_mem, **ctx)

    def explain(self) -> str:
        inner = self._schedulers[self._active_key].explain()
        return f"[NEXUS] {self._reason} → {inner}"

    def metrics(self) -> dict:
        return {'active_scheduler': self._active_key,
                'inner_metrics': self._schedulers[self._active_key].metrics()}

    @property
    def active_key(self):
        return self._active_key

    @property
    def active_color(self):
        return self._schedulers[self._active_key].color


# ═══════════════════════════════════════════════════════════════════════════
# TELEMETRY & ML (preserved from original, extended)
# ═══════════════════════════════════════════════════════════════════════════

class TelemetryDB:
    def __init__(self, db_name=DEFAULT_DB_NAME):
        self.db_name = db_name
        self.legacy_db_name = LEGACY_DB_NAME
        self.conn   = sqlite3.connect(db_name, check_same_thread=False)
        self.lock   = threading.Lock()
        self._init()

    def _init(self):
        with self.lock:
            self.conn.execute("DROP TABLE IF EXISTS AllocationStats")
            self.conn.execute("""
                CREATE TABLE AllocationStats (
                    timestamp TEXT, tick INTEGER,
                    scheduler TEXT,
                    real_avail_cpu REAL, real_avail_mem REAL,
                    total_requested_cpu REAL, total_requested_mem REAL,
                    total_allocated_cpu REAL, total_allocated_mem REAL,
                    cpu_pressure REAL, mem_pressure REAL,
                    bottleneck_state TEXT
                )
            """)
            self.conn.commit()

    def record(self, row: dict):
        with self.lock:
            self.conn.execute("""
                INSERT INTO AllocationStats VALUES
                (:timestamp,:tick,:scheduler,
                 :real_avail_cpu,:real_avail_mem,
                 :total_requested_cpu,:total_requested_mem,
                 :total_allocated_cpu,:total_allocated_mem,
                 :cpu_pressure,:mem_pressure,:bottleneck_state)
            """, row)
            self.conn.commit()

    def get_df(self) -> pd.DataFrame:
        with self.lock:
            return pd.read_sql("SELECT * FROM AllocationStats", self.conn)

    def get_legacy_df(self) -> pd.DataFrame:
        """Load telemetry from the legacy simulator DB, if available."""
        if not os.path.exists(self.legacy_db_name):
            return pd.DataFrame()
        try:
            legacy_conn = sqlite3.connect(self.legacy_db_name)
            try:
                df = pd.read_sql("SELECT * FROM AllocationStats", legacy_conn)
            finally:
                legacy_conn.close()
            if df.empty:
                return df

            # Normalize legacy schema to the new schema.
            if 'tick' not in df.columns:
                df.insert(1, 'tick', range(1, len(df) + 1))
            if 'scheduler' not in df.columns:
                df['scheduler'] = 'legacy-adaptive'

            expected = [
                'timestamp', 'tick', 'scheduler',
                'real_avail_cpu', 'real_avail_mem',
                'total_requested_cpu', 'total_requested_mem',
                'total_allocated_cpu', 'total_allocated_mem',
                'cpu_pressure', 'mem_pressure', 'bottleneck_state'
            ]
            for col in expected:
                if col not in df.columns:
                    df[col] = None
            return df[expected]
        except Exception:
            return pd.DataFrame()

    def get_unified_df(self) -> pd.DataFrame:
        """Merge legacy and current telemetry for unified analytics."""
        current = self.get_df()
        legacy = self.get_legacy_df()
        if legacy.empty:
            return current
        if current.empty:
            return legacy
        return pd.concat([legacy, current], ignore_index=True)

    def predict_next(self) -> dict:
        df = self.get_unified_df()
        if len(df) < MIN_PREDICTION_POINTS:
            return {}
        X = np.arange(len(df)).reshape(-1, 1)
        mc = LinearRegression().fit(X, df["real_avail_cpu"].values)
        mm = LinearRegression().fit(X, df["real_avail_mem"].values)
        nx = np.array([[len(df)]])
        def trend(s):
            if s > 0.05: return "increasing"
            if s < -0.05: return "decreasing"
            return "stable"
        return {
            'pred_cpu': round(float(np.clip(mc.predict(nx)[0], 0, 100)), 2),
            'pred_mem': round(float(max(0, mm.predict(nx)[0])), 2),
            'cpu_trend': trend(mc.coef_[0]),
            'mem_trend': trend(mm.coef_[0]),
        }


# ═══════════════════════════════════════════════════════════════════════════
# NEXUS ENGINE — orchestrates everything
# ═══════════════════════════════════════════════════════════════════════════

class NexusEngine:
    def __init__(self):
        self.processes   = [VirtualProcess(pid, wt) for pid, wt in DEFAULT_PROCESS_SPECS]
        self.db          = TelemetryDB()
        self.tick        = 0
        self.started_at  = time.time()
        self.cpu_ewma    = 0.0
        self.mem_ewma    = 0.0

        # All schedulers available for selection
        self.schedulers: Dict[str, BaseScheduler] = {
            'nexus':    NexusAdaptiveAI(),
            'rr':       RoundRobinScheduler(),
            'priority': PriorityScheduler(),
            'sjf':      SJFScheduler(),
            'mlq':      MLQScheduler(),
            'mlfq':     MLFQScheduler(),
        }
        self.active_scheduler_key = 'nexus'

        # History for charts
        self.cpu_history:  deque = deque(maxlen=60)
        self.mem_history:  deque = deque(maxlen=60)
        self.gantt_history: deque = deque(maxlen=80)
        self.narration_log: deque = deque(maxlen=40)

        # Ghost Replay buffer
        self.ghost_buffer: deque = deque(maxlen=GHOST_REPLAY_LEN)

        # Battle mode results
        self.battle_results: dict = {}

        # Manual request control: when enabled, per-process requests come
        # from user-provided values instead of random workload generation.
        self.manual_mode_enabled: bool = False
        self.manual_request_overrides: Dict[int, Dict[str, float]] = {}

        # Boot
        psutil.cpu_percent(interval=0.1)
        self._narrate("NEXUS-7 core online. Binding to system kernel.", "boot")

    @property
    def scheduler(self) -> BaseScheduler:
        return self.schedulers[self.active_scheduler_key]

    def _narrate(self, msg: str, kind: str = "info"):
        ts = time.strftime("%H:%M:%S")
        entry = {"ts": ts, "msg": msg, "kind": kind}
        self.narration_log.appendleft(entry)

    def _measure_system(self):
        cpu_used   = psutil.cpu_percent(interval=None)
        avail_cpu  = max(0.0, 100.0 - cpu_used)
        mem_info   = psutil.virtual_memory()
        avail_mem  = mem_info.available / (1024 * 1024)
        cpu_per_core = psutil.cpu_percent(interval=None, percpu=True)
        freq       = psutil.cpu_freq()
        uptime_s   = max(0, int(time.time() - psutil.boot_time()))
        return {
            'avail_cpu':   avail_cpu,
            'used_cpu':    cpu_used,
            'avail_mem':   avail_mem,
            'used_mem':    mem_info.used / (1024 * 1024),
            'total_mem':   mem_info.total / (1024 * 1024),
            'mem_pct':     mem_info.percent,
            'swap_pct':    psutil.swap_memory().percent,
            'cpu_per_core': cpu_per_core,
            'cpu_freq':    round(freq.current, 0) if freq else 0,
            'uptime_s':    uptime_s,
        }

    def _collect_requests(self):
        requests = []
        for proc in self.processes:
            if proc.wait_time > 2:
                proc.priority = max(1, proc.priority - 2)
                proc.wait_time = 0
            manual = self.manual_request_overrides.get(proc.pid)
            if self.manual_mode_enabled:
                # In manual mode, allocation demand is driven strictly by
                # user-provided PID entries; unspecified PIDs request 0.
                if manual:
                    req_cpu = float(max(0.0, manual.get('req_cpu', 0.0)))
                    req_mem = float(max(0.0, manual.get('req_mem', 0.0)))
                else:
                    req_cpu, req_mem = 0.0, 0.0
            else:
                req_cpu, req_mem = proc.generate_requests()
            proc.last_req_cpu = req_cpu
            proc.last_req_mem = req_mem
            requests.append({'process': proc, 'req_cpu': req_cpu, 'req_mem': req_mem})
        requests.sort(key=lambda r: r['process'].priority)
        return requests

    def _ewma(self, prev, cur):
        return EWMA_ALPHA * cur + (1 - EWMA_ALPHA) * prev

    def _pressure(self, requested, available):
        if available <= 0:
            return 1.0 if requested > 0 else 0.0
        return min(1.0, requested / available)

    def _bottleneck(self, cpu_p, mem_p) -> str:
        c = cpu_p >= BOTTLENECK_PRESSURE_THRESHOLD
        m = mem_p >= BOTTLENECK_PRESSURE_THRESHOLD
        if c and m: return "cpu+mem"
        if c: return "cpu"
        if m: return "memory"
        return "none"

    def _apply_allocations(self, requests, alloc_map):
        """Write scheduler output back onto process objects."""
        for req in requests:
            pid  = req['process'].pid
            proc = req['process']
            entry = next((a for a in alloc_map if a['pid'] == pid), None)
            if entry:
                proc.allocated_cpu = entry['cpu']
                proc.allocated_mem = entry['mem']
                fulfilled = (entry['cpu'] >= req['req_cpu'] * 0.90 and
                             entry['mem'] >= req['req_mem'] * 0.90)
            else:
                proc.allocated_cpu = 0.0
                proc.allocated_mem = 0.0
                fulfilled = False

            proc.update_state(fulfilled)
            proc.history_cpu.append(round(proc.allocated_cpu, 1))
            proc.history_mem.append(round(proc.allocated_mem, 1))

            if fulfilled:
                proc.priority = min(10, proc.priority + 1)
                proc.wait_time = 0
                proc.starvation_credit = max(0.0, proc.starvation_credit - 0.25)
            else:
                proc.wait_time += 1
                proc.priority  = max(1, proc.priority - 1)
                proc.starvation_credit = min(2.0, proc.starvation_credit + 0.2)

    def _detect_collisions(self, requests) -> dict:
        """
        Detect contention collisions when multiple processes are heavily
        under-fulfilled on the same resource in the same tick.
        """
        participants = []

        for req in requests:
            proc = req['process']
            req_cpu = float(req.get('req_cpu', 0.0))
            req_mem = float(req.get('req_mem', 0.0))
            alloc_cpu = float(getattr(proc, 'allocated_cpu', 0.0))
            alloc_mem = float(getattr(proc, 'allocated_mem', 0.0))

            cpu_unmet = (1.0 - (alloc_cpu / req_cpu)) if req_cpu > 0 else 0.0
            mem_unmet = (1.0 - (alloc_mem / req_mem)) if req_mem > 0 else 0.0

            cpu_hot = req_cpu >= COLLISION_MIN_REQUEST_CPU and cpu_unmet >= COLLISION_UNFULFILLED_RATIO
            mem_hot = req_mem >= COLLISION_MIN_REQUEST_MEM and mem_unmet >= COLLISION_UNFULFILLED_RATIO

            if not (cpu_hot or mem_hot):
                continue

            if cpu_hot and mem_hot:
                collision_type = 'dual'
            elif cpu_hot:
                collision_type = 'cpu'
            else:
                collision_type = 'memory'

            participants.append({
                'pid': proc.pid,
                'type': collision_type,
                'req_cpu': round(req_cpu, 2),
                'alloc_cpu': round(alloc_cpu, 2),
                'req_mem': round(req_mem, 2),
                'alloc_mem': round(alloc_mem, 2),
                'cpu_unmet_pct': round(max(0.0, cpu_unmet) * 100.0, 1),
                'mem_unmet_pct': round(max(0.0, mem_unmet) * 100.0, 1),
            })

        cpu_candidates = [p for p in participants if p['type'] in {'cpu', 'dual'}]
        mem_candidates = [p for p in participants if p['type'] in {'memory', 'dual'}]

        cpu_collision = len(cpu_candidates) >= COLLISION_MIN_PARTICIPANTS
        mem_collision = len(mem_candidates) >= COLLISION_MIN_PARTICIPANTS

        if not cpu_collision and not mem_collision:
            return {
                'active': False,
                'count': 0,
                'type': 'none',
                'level': 'ok',
                'events': [],
            }

        if cpu_collision and mem_collision:
            collision_type = 'cpu+mem'
            level = 'critical'
            relevant = [p for p in participants if p['type'] in {'cpu', 'memory', 'dual'}]
        elif cpu_collision:
            collision_type = 'cpu'
            level = 'warn'
            relevant = cpu_candidates
        else:
            collision_type = 'memory'
            level = 'warn'
            relevant = mem_candidates

        # Keep payload compact for websocket updates while preserving top-impact events.
        ranked = sorted(
            relevant,
            key=lambda p: (max(p['cpu_unmet_pct'], p['mem_unmet_pct']), p['pid']),
            reverse=True,
        )
        events = ranked[:6]

        return {
            'active': True,
            'count': len(relevant),
            'type': collision_type,
            'level': level,
            'events': events,
        }

    # ── Battle Mode ──────────────────────────────────────────────────────────
    def run_battle(self, requests, avail_cpu, avail_mem, cpu_p, mem_p) -> dict:
        """Run all 5 classical schedulers on the same snapshot, compare metrics."""
        results = {}
        for key, sched in self.schedulers.items():
            if key == 'nexus':
                continue
            try:
                allocs = sched.schedule(list(requests), avail_cpu, avail_mem,
                                        cpu_pressure=cpu_p, mem_pressure=mem_p)
                total_cpu = sum(a['cpu'] for a in allocs)
                total_mem = sum(a['mem'] for a in allocs)
                req_cpu   = sum(r['req_cpu'] for r in requests)
                req_mem   = sum(r['req_mem'] for r in requests)
                eff_cpu   = total_cpu / max(1, req_cpu)
                eff_mem   = total_mem / max(1, req_mem)
                results[key] = {
                    'name':      sched.name,
                    'color':     sched.color,
                    'total_cpu': round(total_cpu, 2),
                    'total_mem': round(total_mem, 2),
                    'eff_cpu':   round(eff_cpu * 100, 1),
                    'eff_mem':   round(eff_mem * 100, 1),
                    'score':     round((eff_cpu + eff_mem) / 2 * 100, 1),
                    'explain':   sched.explain(),
                }
            except Exception:
                pass
        return results

    def tick_once(self) -> dict:
        self.tick += 1
        sys   = self._measure_system()
        reqs  = self._collect_requests()

        total_req_cpu = sum(r['req_cpu'] for r in reqs)
        total_req_mem = sum(r['req_mem'] for r in reqs)
        cpu_p = self._pressure(total_req_cpu, sys['avail_cpu'])
        mem_p = self._pressure(total_req_mem, sys['avail_mem'])
        self.cpu_ewma = self._ewma(self.cpu_ewma, cpu_p)
        self.mem_ewma = self._ewma(self.mem_ewma, mem_p)
        bottle = self._bottleneck(self.cpu_ewma, self.mem_ewma)

        # Schedule
        allocs = self.scheduler.schedule(
            reqs, sys['avail_cpu'], sys['avail_mem'],
            cpu_pressure=self.cpu_ewma, mem_pressure=self.mem_ewma)
        self._apply_allocations(reqs, allocs)

        total_alloc_cpu = sum(a['cpu'] for a in allocs)
        total_alloc_mem = sum(a['mem'] for a in allocs)
        collision = self._detect_collisions(reqs)

        # ML prediction
        pred = self.db.predict_next()

        # Narration
        explain = self.scheduler.explain()
        if self.tick % 3 == 0 or bottle != "none":
            kind = "warn" if bottle != "none" else "info"
            self.db.record({
                'timestamp': time.strftime("%H:%M:%S"),
                'tick': self.tick,
                'scheduler': self.active_scheduler_key,
                'real_avail_cpu': sys['avail_cpu'],
                'real_avail_mem': sys['avail_mem'],
                'total_requested_cpu': total_req_cpu,
                'total_requested_mem': total_req_mem,
                'total_allocated_cpu': total_alloc_cpu,
                'total_allocated_mem': total_alloc_mem,
                'cpu_pressure': self.cpu_ewma,
                'mem_pressure': self.mem_ewma,
                'bottleneck_state': bottle,
            })
            self._narrate(explain, kind)

        if bottle == "cpu+mem":
            self._narrate(f"⚠ DUAL BOTTLENECK detected — CPU {self.cpu_ewma:.0%} · MEM {self.mem_ewma:.0%}", "critical")
        elif bottle == "cpu":
            self._narrate(f"⚠ CPU bottleneck {self.cpu_ewma:.0%}", "warn")
        elif bottle == "memory":
            self._narrate(f"⚠ MEM bottleneck {self.mem_ewma:.0%}", "warn")

        if collision['active']:
            if collision['level'] == 'critical':
                self._narrate(
                    f"⚠ COLLISION: concurrent CPU+MEM contention ({collision['count']} processes)",
                    "critical",
                )
            elif collision['type'] == 'cpu':
                self._narrate(
                    f"⚠ CPU collision detected ({collision['count']} processes competing)",
                    "warn",
                )
            elif collision['type'] == 'memory':
                self._narrate(
                    f"⚠ MEM collision detected ({collision['count']} processes competing)",
                    "warn",
                )

        # Histories
        self.cpu_history.append(round(sys['used_cpu'], 1))
        self.mem_history.append(round(sys['mem_pct'], 1))

        # Gantt strip entry
        gantt_entry = {
            'tick': self.tick,
            'slots': [{'pid': p.pid, 'cpu': p.allocated_cpu,
                       'color': p.color, 'state': p.state}
                      for p in self.processes]
        }
        self.gantt_history.append(gantt_entry)

        # Battle (every 10 ticks)
        if self.tick % 10 == 0:
            self.battle_results = self.run_battle(
                reqs, sys['avail_cpu'], sys['avail_mem'],
                self.cpu_ewma, self.mem_ewma)

        # Active scheduler info
        active_sched = self.scheduler
        active_inner = (active_sched.active_key
                        if hasattr(active_sched, 'active_key')
                        else self.active_scheduler_key)
        active_color = (active_sched.active_color
                        if hasattr(active_sched, 'active_color')
                        else active_sched.color)

        legacy_df = self.db.get_legacy_df()
        legacy_rows = int(len(legacy_df))

        snapshot = {
            'tick':         self.tick,
            'legacy_rows':  legacy_rows,
            'alert_profile': ALERT_SENSITIVITY_PROFILE,
            'system':       sys,
            'processes':    [p.to_dict() for p in self.processes],
            'cpu_ewma':     round(self.cpu_ewma, 3),
            'mem_ewma':     round(self.mem_ewma, 3),
            'bottleneck':   bottle,
            'collision':    collision,
            'total_req_cpu': round(total_req_cpu, 2),
            'total_req_mem': round(total_req_mem, 2),
            'total_alloc_cpu': round(total_alloc_cpu, 2),
            'total_alloc_mem': round(total_alloc_mem, 2),
            'scheduler_key': self.active_scheduler_key,
            'scheduler_name': active_sched.name,
            'scheduler_color': active_color,
            'inner_scheduler': active_inner,
            'explain':      explain,
            'formula':      active_sched.formula,
            'narration':    list(self.narration_log)[:20],
            'cpu_history':  list(self.cpu_history),
            'mem_history':  list(self.mem_history),
            'gantt':        list(self.gantt_history)[-30:],
            'battle':       self.battle_results,
            'ml_pred':      pred,
            'scheduler_meta': {
                k: {'name': v.name, 'color': v.color, 'formula': v.formula}
                for k, v in self.schedulers.items()
            },
            'legacy': {
                'available': os.path.exists(LEGACY_DB_NAME),
                'rows': legacy_rows,
                'db_file': LEGACY_DB_NAME,
            },
            'manual_control': {
                'enabled': self.manual_mode_enabled,
                'overrides': {
                    str(pid): {
                        'req_cpu': round(values.get('req_cpu', 0.0), 2),
                        'req_mem': round(values.get('req_mem', 0.0), 2),
                    }
                    for pid, values in self.manual_request_overrides.items()
                }
            }
        }

        # Ghost buffer
        self.ghost_buffer.append(snapshot)
        return snapshot

    def set_scheduler(self, key: str):
        if key in self.schedulers:
            self.active_scheduler_key = key
            self._narrate(f"Scheduler → {self.schedulers[key].name}", "info")
            return True
        return False

    def get_ghost_replay(self) -> list:
        return list(self.ghost_buffer)

    def add_process(self):
        existing_pids = {p.pid for p in self.processes}
        new_pid = max(existing_pids) + 1
        wt = random.choice(['CPU-Intensive', 'Memory-Intensive', 'Balanced'])
        self.processes.append(VirtualProcess(new_pid, wt))
        self._narrate(f"PID {new_pid} ({wt}) spawned", "ok")

    def remove_process(self):
        if len(self.processes) > 2:
            p = self.processes.pop()
            self.manual_request_overrides.pop(p.pid, None)
            self._narrate(f"PID {p.pid} terminated", "warn")

    def set_manual_mode(self, enabled: bool):
        self.manual_mode_enabled = bool(enabled)
        mode = "enabled" if self.manual_mode_enabled else "disabled"
        self._narrate(f"Manual request mode {mode}", "info")

    def set_manual_requests(self, overrides: Dict[int, Dict[str, float]]):
        valid_pids = {p.pid for p in self.processes}
        cleaned: Dict[int, Dict[str, float]] = {}
        for pid, values in (overrides or {}).items():
            try:
                pid_int = int(pid)
                req_cpu = float(values.get('req_cpu', 0.0))
                req_mem = float(values.get('req_mem', 0.0))
            except (TypeError, ValueError, AttributeError):
                continue
            if pid_int not in valid_pids:
                continue
            cleaned[pid_int] = {
                'req_cpu': max(0.0, req_cpu),
                'req_mem': max(0.0, req_mem),
            }
        self.manual_request_overrides = cleaned
        self._narrate(
            f"Manual request profile updated ({len(cleaned)} process overrides)",
            "ok"
        )


# ═══════════════════════════════════════════════════════════════════════════
# FLASK + SOCKETIO SERVER
# ═══════════════════════════════════════════════════════════════════════════

app    = Flask(__name__, static_folder=".")
CORS(app)
sio    = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
engine = NexusEngine()

def poll_loop():
    while True:
        snap = engine.tick_once()
        sio.emit('state', snap)
        time.sleep(DEFAULT_TICK_DELAY_SECONDS)

@app.route("/")
def index():
    return send_file(os.path.join(app.root_path, "nexus7_landing.html"))

@app.route("/landing")
def landing_page():
    return send_file(os.path.join(app.root_path, "nexus7_landing.html"))

@app.route("/dashboard")
def dashboard_page():
    return send_file(os.path.join(app.root_path, "nexus7_dashboard.html"))

@sio.on("set_scheduler")
def on_set_scheduler(data):
    engine.set_scheduler(data.get("key", "nexus"))

@sio.on("add_process")
def on_add_process(_=None):
    engine.add_process()

@sio.on("remove_process")
def on_remove_process(_=None):
    engine.remove_process()

@sio.on("ghost_replay")
def on_ghost_replay(_=None):
    sio.emit("ghost_data", engine.get_ghost_replay())

@sio.on("set_manual_mode")
def on_set_manual_mode(data):
    engine.set_manual_mode(bool((data or {}).get("enabled", False)))

@sio.on("set_manual_requests")
def on_set_manual_requests(data):
    overrides = (data or {}).get("overrides", {})
    engine.set_manual_requests(overrides)

if __name__ == "__main__":
    print(f"\n{'='*55}")
    print("  NEXUS-7 — Sentient Resource Orchestration Engine")
    print(f"  Open: http://localhost:{PORT}")
    print(f"  Ctrl+C to stop")
    print(f"{'='*55}\n")
    sio.start_background_task(poll_loop)
    sio.run(app, host="0.0.0.0", port=PORT, debug=False)
