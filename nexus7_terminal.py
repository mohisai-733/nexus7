import os
import queue
import random
import shlex
import sqlite3
import threading
import time
import ctypes
from collections import deque

import numpy as np
import pandas as pd
import psutil
from sklearn.linear_model import LinearRegression


DEFAULT_DB_NAME = "allocation_logs_terminal.db"
DEFAULT_TICK_DELAY_SECONDS = 1.0
MIN_PREDICTION_POINTS = 5
EWMA_ALPHA = 0.35
MIN_RESOURCE_SLICE = 0.5

# Alert profile can be: sensitive, balanced, severe
ALERT_SENSITIVITY_PROFILE = os.getenv(
    "NEXUS_TERMINAL_ALERT_PROFILE",
    os.getenv("NEXUS_ALERT_PROFILE", "balanced"),
).strip().lower()
ALERT_SENSITIVITY_PRESETS = {
    "sensitive": {
        "bottleneck_threshold": 0.72,
        "collision_unfulfilled_ratio": 0.30,
        "collision_min_request_cpu": 3.0,
        "collision_min_request_mem": 48.0,
        "collision_min_participants": 2,
    },
    "balanced": {
        "bottleneck_threshold": 0.85,
        "collision_unfulfilled_ratio": 0.45,
        "collision_min_request_cpu": 5.0,
        "collision_min_request_mem": 64.0,
        "collision_min_participants": 2,
    },
    "severe": {
        "bottleneck_threshold": 0.93,
        "collision_unfulfilled_ratio": 0.65,
        "collision_min_request_cpu": 8.0,
        "collision_min_request_mem": 128.0,
        "collision_min_participants": 3,
    },
}


def _resolve_alert_profile(name):
    return ALERT_SENSITIVITY_PRESETS.get(name, ALERT_SENSITIVITY_PRESETS["balanced"])


ALERT_CFG = _resolve_alert_profile(ALERT_SENSITIVITY_PROFILE)
BOTTLENECK_PRESSURE_THRESHOLD = ALERT_CFG["bottleneck_threshold"]
COLLISION_UNFULFILLED_RATIO = ALERT_CFG["collision_unfulfilled_ratio"]
COLLISION_MIN_REQUEST_CPU = ALERT_CFG["collision_min_request_cpu"]
COLLISION_MIN_REQUEST_MEM = ALERT_CFG["collision_min_request_mem"]
COLLISION_MIN_PARTICIPANTS = ALERT_CFG["collision_min_participants"]

DEFAULT_PROCESS_SPECS = [
    (1, "CPU-Intensive"),
    (2, "Memory-Intensive"),
    (3, "Balanced"),
    (4, "CPU-Intensive"),
    (5, "Balanced"),
]


class VirtualProcess:
    def __init__(self, pid, workload_type):
        self.pid = pid
        self.workload_type = workload_type
        self.priority = 5
        self.allocated_cpu = 0.0
        self.allocated_mem = 0.0
        self.requested_cpu = 0.0
        self.requested_mem = 0.0
        self.wait_time = 0
        self.starvation_credit = 0.0

    def generate_requests(self):
        if self.workload_type == "CPU-Intensive":
            return random.uniform(15.0, 40.0), random.uniform(10.0, 50.0)
        if self.workload_type == "Memory-Intensive":
            return random.uniform(1.0, 10.0), random.uniform(200.0, 800.0)
        return random.uniform(5.0, 20.0), random.uniform(50.0, 300.0)


class AdaptiveAllocator:
    def __init__(self, db_name=DEFAULT_DB_NAME):
        self.db_name = db_name
        self.processes = [VirtualProcess(pid, workload_type) for pid, workload_type in DEFAULT_PROCESS_SPECS]
        self.conn = sqlite3.connect(self.db_name, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.cpu_pressure_ewma = 0.0
        self.mem_pressure_ewma = 0.0
        self.manual_mode_enabled = False
        self.manual_request_overrides = {}
        self._init_db()
        psutil.cpu_percent(interval=0.1)

    def _init_db(self):
        self.cursor.execute("DROP TABLE IF EXISTS AllocationStats")
        self.cursor.execute(
            """
            CREATE TABLE AllocationStats (
                timestamp TEXT,
                tick INTEGER,
                real_avail_cpu REAL,
                real_avail_mem REAL,
                total_requested_cpu REAL,
                total_requested_mem REAL,
                total_allocated_cpu REAL,
                total_allocated_mem REAL,
                cpu_pressure REAL,
                mem_pressure REAL,
                bottleneck_state TEXT,
                mode TEXT
            )
            """
        )
        self.conn.commit()

    def close(self):
        if self.conn:
            self.conn.close()

    def set_manual_mode(self, enabled):
        self.manual_mode_enabled = bool(enabled)

    def set_manual_request(self, pid, req_cpu, req_mem):
        self.manual_request_overrides[int(pid)] = {
            "req_cpu": max(0.0, float(req_cpu)),
            "req_mem": max(0.0, float(req_mem)),
        }

    def remove_manual_request(self, pid):
        self.manual_request_overrides.pop(int(pid), None)

    def clear_manual_requests(self):
        self.manual_request_overrides = {}

    def _measure_system_capacity(self):
        real_used_cpu = psutil.cpu_percent(interval=None)
        real_avail_cpu = max(0.0, 100.0 - real_used_cpu)
        mem_info = psutil.virtual_memory()
        real_avail_mem_mb = mem_info.available / (1024 * 1024)
        return real_avail_cpu, real_avail_mem_mb

    def _collect_requests(self):
        requests = []
        total_req_cpu = 0.0
        total_req_mem = 0.0
        for process in self.processes:
            if process.wait_time > 2:
                process.priority = max(1, process.priority - 2)
                process.wait_time = 0

            manual = self.manual_request_overrides.get(process.pid)
            if self.manual_mode_enabled:
                if manual:
                    req_cpu = float(max(0.0, manual.get("req_cpu", 0.0)))
                    req_mem = float(max(0.0, manual.get("req_mem", 0.0)))
                else:
                    req_cpu, req_mem = 0.0, 0.0
            else:
                req_cpu, req_mem = process.generate_requests()

            process.requested_cpu = req_cpu
            process.requested_mem = req_mem
            requests.append({"process": process, "req_cpu": req_cpu, "req_mem": req_mem})
            total_req_cpu += req_cpu
            total_req_mem += req_mem

        requests.sort(key=lambda item: item["process"].priority)
        return requests, total_req_cpu, total_req_mem

    def _update_ewma(self, previous, current):
        return (EWMA_ALPHA * current) + ((1 - EWMA_ALPHA) * previous)

    def _pressure(self, total_requested, total_available):
        if total_available <= 0:
            return 1.0 if total_requested > 0 else 0.0
        return min(1.0, total_requested / total_available)

    def _get_bottleneck_state(self, cpu_pressure, mem_pressure):
        cpu_hot = cpu_pressure >= BOTTLENECK_PRESSURE_THRESHOLD
        mem_hot = mem_pressure >= BOTTLENECK_PRESSURE_THRESHOLD
        if cpu_hot and mem_hot:
            return "cpu+mem"
        if cpu_hot:
            return "cpu"
        if mem_hot:
            return "memory"
        return "none"

    def _compute_process_weight(self, process, cpu_pressure, mem_pressure):
        fairness_boost = 1.0 + (0.35 * process.wait_time) + process.starvation_credit
        priority_boost = (11 - process.priority) / 10.0

        workload_bias = 1.0
        if process.workload_type == "CPU-Intensive" and cpu_pressure > 0.7:
            workload_bias += 0.25
        elif process.workload_type == "Memory-Intensive" and mem_pressure > 0.7:
            workload_bias += 0.25
        elif process.workload_type == "Balanced":
            workload_bias += 0.1
        return max(0.1, fairness_boost * priority_boost * workload_bias)

    def _detect_collisions(self, requests):
        participants = []

        for req in requests:
            process = req["process"]
            req_cpu = float(req.get("req_cpu", 0.0))
            req_mem = float(req.get("req_mem", 0.0))
            alloc_cpu = float(getattr(process, "allocated_cpu", 0.0))
            alloc_mem = float(getattr(process, "allocated_mem", 0.0))

            cpu_unmet = (1.0 - (alloc_cpu / req_cpu)) if req_cpu > 0 else 0.0
            mem_unmet = (1.0 - (alloc_mem / req_mem)) if req_mem > 0 else 0.0

            cpu_hot = req_cpu >= COLLISION_MIN_REQUEST_CPU and cpu_unmet >= COLLISION_UNFULFILLED_RATIO
            mem_hot = req_mem >= COLLISION_MIN_REQUEST_MEM and mem_unmet >= COLLISION_UNFULFILLED_RATIO

            if not (cpu_hot or mem_hot):
                continue

            if cpu_hot and mem_hot:
                collision_type = "dual"
            elif cpu_hot:
                collision_type = "cpu"
            else:
                collision_type = "memory"

            participants.append(
                {
                    "pid": process.pid,
                    "type": collision_type,
                    "req_cpu": round(req_cpu, 2),
                    "alloc_cpu": round(alloc_cpu, 2),
                    "req_mem": round(req_mem, 2),
                    "alloc_mem": round(alloc_mem, 2),
                    "cpu_unmet_pct": round(max(0.0, cpu_unmet) * 100.0, 1),
                    "mem_unmet_pct": round(max(0.0, mem_unmet) * 100.0, 1),
                }
            )

        cpu_candidates = [p for p in participants if p["type"] in {"cpu", "dual"}]
        mem_candidates = [p for p in participants if p["type"] in {"memory", "dual"}]

        cpu_collision = len(cpu_candidates) >= COLLISION_MIN_PARTICIPANTS
        mem_collision = len(mem_candidates) >= COLLISION_MIN_PARTICIPANTS

        if not cpu_collision and not mem_collision:
            return {
                "active": False,
                "count": 0,
                "type": "none",
                "level": "ok",
                "events": [],
            }

        if cpu_collision and mem_collision:
            collision_type = "cpu+mem"
            level = "critical"
            relevant = [p for p in participants if p["type"] in {"cpu", "memory", "dual"}]
        elif cpu_collision:
            collision_type = "cpu"
            level = "warn"
            relevant = cpu_candidates
        else:
            collision_type = "memory"
            level = "warn"
            relevant = mem_candidates

        ranked = sorted(
            relevant,
            key=lambda p: (max(p["cpu_unmet_pct"], p["mem_unmet_pct"]), p["pid"]),
            reverse=True,
        )
        events = ranked[:6]

        return {
            "active": True,
            "count": len(relevant),
            "type": collision_type,
            "level": level,
            "events": events,
        }

    def _adaptive_allocate(self, requests, avail_cpu_pool, avail_mem_pool, cpu_pressure, mem_pressure):
        if not requests:
            return 0.0, 0.0

        weights = {
            request["process"].pid: self._compute_process_weight(request["process"], cpu_pressure, mem_pressure)
            for request in requests
        }
        weight_sum = sum(weights.values())

        process_allocations = {}
        for request in requests:
            process = request["process"]
            process_allocations[process.pid] = {
                "cpu": 0.0,
                "mem": 0.0,
                "req_cpu": request["req_cpu"],
                "req_mem": request["req_mem"],
            }

        for request in requests:
            process = request["process"]
            pid = process.pid
            weight = weights[pid]

            cpu_target = avail_cpu_pool * (weight / weight_sum)
            mem_target = avail_mem_pool * (weight / weight_sum)

            cpu_alloc = min(request["req_cpu"], max(MIN_RESOURCE_SLICE, cpu_target))
            mem_alloc = min(request["req_mem"], max(MIN_RESOURCE_SLICE, mem_target))

            cpu_alloc = min(cpu_alloc, avail_cpu_pool)
            mem_alloc = min(mem_alloc, avail_mem_pool)

            process_allocations[pid]["cpu"] += cpu_alloc
            process_allocations[pid]["mem"] += mem_alloc
            avail_cpu_pool -= cpu_alloc
            avail_mem_pool -= mem_alloc

        for _ in range(2):
            if avail_cpu_pool <= 0 and avail_mem_pool <= 0:
                break

            cpu_unmet_weight = sum(
                max(0.0, process_allocations[r["process"].pid]["req_cpu"] - process_allocations[r["process"].pid]["cpu"])
                * weights[r["process"].pid]
                for r in requests
            )
            mem_unmet_weight = sum(
                max(0.0, process_allocations[r["process"].pid]["req_mem"] - process_allocations[r["process"].pid]["mem"])
                * weights[r["process"].pid]
                for r in requests
            )

            for request in requests:
                pid = request["process"].pid
                entry = process_allocations[pid]
                weight = weights[pid]

                cpu_unmet = max(0.0, entry["req_cpu"] - entry["cpu"])
                mem_unmet = max(0.0, entry["req_mem"] - entry["mem"])

                if avail_cpu_pool > 0 and cpu_unmet > 0 and cpu_unmet_weight > 0:
                    cpu_share = avail_cpu_pool * ((cpu_unmet * weight) / cpu_unmet_weight)
                    cpu_gain = min(cpu_unmet, cpu_share, avail_cpu_pool)
                    entry["cpu"] += cpu_gain
                    avail_cpu_pool -= cpu_gain

                if avail_mem_pool > 0 and mem_unmet > 0 and mem_unmet_weight > 0:
                    mem_share = avail_mem_pool * ((mem_unmet * weight) / mem_unmet_weight)
                    mem_gain = min(mem_unmet, mem_share, avail_mem_pool)
                    entry["mem"] += mem_gain
                    avail_mem_pool -= mem_gain

        total_alloc_cpu = 0.0
        total_alloc_mem = 0.0
        for request in requests:
            process = request["process"]
            pid = process.pid
            alloc_cpu = process_allocations[pid]["cpu"]
            alloc_mem = process_allocations[pid]["mem"]

            process.allocated_cpu = alloc_cpu
            process.allocated_mem = alloc_mem
            total_alloc_cpu += alloc_cpu
            total_alloc_mem += alloc_mem

            cpu_fulfilled = alloc_cpu >= request["req_cpu"] * 0.95
            mem_fulfilled = alloc_mem >= request["req_mem"] * 0.95
            if cpu_fulfilled and mem_fulfilled:
                process.priority = min(10, process.priority + 1)
                process.wait_time = 0
                process.starvation_credit = max(0.0, process.starvation_credit - 0.25)
            else:
                process.wait_time += 1
                process.priority = max(1, process.priority - 1)
                process.starvation_credit = min(2.0, process.starvation_credit + 0.2)

        return total_alloc_cpu, total_alloc_mem

    def _record_telemetry(
        self,
        tick,
        real_avail_cpu,
        real_avail_mem_mb,
        total_req_cpu,
        total_req_mem,
        total_alloc_cpu,
        total_alloc_mem,
        cpu_pressure,
        mem_pressure,
        bottleneck_state,
    ):
        timestamp = pd.Timestamp.now().strftime("%H:%M:%S")
        self.cursor.execute(
            "INSERT INTO AllocationStats VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                timestamp,
                tick,
                real_avail_cpu,
                real_avail_mem_mb,
                total_req_cpu,
                total_req_mem,
                total_alloc_cpu,
                total_alloc_mem,
                cpu_pressure,
                mem_pressure,
                bottleneck_state,
                "manual" if self.manual_mode_enabled else "auto",
            ),
        )
        self.conn.commit()

    def predict_next_tick(self):
        df = pd.read_sql("SELECT * FROM AllocationStats", self.conn)
        if len(df) < MIN_PREDICTION_POINTS:
            return None

        X = np.arange(len(df)).reshape(-1, 1)
        model_cpu = LinearRegression().fit(X, df["real_avail_cpu"].values)
        model_mem = LinearRegression().fit(X, df["real_avail_mem"].values)

        next_t = np.array([[len(df)]])
        pred_cpu = max(0, min(100, model_cpu.predict(next_t)[0]))
        pred_mem = max(0, model_mem.predict(next_t)[0])
        return float(pred_cpu), float(pred_mem)

    def allocate_resources(self, tick):
        real_avail_cpu, real_avail_mem_mb = self._measure_system_capacity()
        requests, total_req_cpu, total_req_mem = self._collect_requests()

        cpu_pressure = self._pressure(total_req_cpu, real_avail_cpu)
        mem_pressure = self._pressure(total_req_mem, real_avail_mem_mb)
        self.cpu_pressure_ewma = self._update_ewma(self.cpu_pressure_ewma, cpu_pressure)
        self.mem_pressure_ewma = self._update_ewma(self.mem_pressure_ewma, mem_pressure)
        bottleneck_state = self._get_bottleneck_state(self.cpu_pressure_ewma, self.mem_pressure_ewma)

        total_alloc_cpu, total_alloc_mem = self._adaptive_allocate(
            requests,
            real_avail_cpu,
            real_avail_mem_mb,
            self.cpu_pressure_ewma,
            self.mem_pressure_ewma,
        )
        collision = self._detect_collisions(requests)

        self._record_telemetry(
            tick,
            real_avail_cpu,
            real_avail_mem_mb,
            total_req_cpu,
            total_req_mem,
            total_alloc_cpu,
            total_alloc_mem,
            self.cpu_pressure_ewma,
            self.mem_pressure_ewma,
            bottleneck_state,
        )

        return {
            "tick": tick,
            "mode": "manual" if self.manual_mode_enabled else "auto",
            "alert_profile": ALERT_SENSITIVITY_PROFILE,
            "real_avail_cpu": real_avail_cpu,
            "real_avail_mem": real_avail_mem_mb,
            "cpu_pressure": self.cpu_pressure_ewma,
            "mem_pressure": self.mem_pressure_ewma,
            "bottleneck": bottleneck_state,
            "collision": collision,
            "total_req_cpu": total_req_cpu,
            "total_req_mem": total_req_mem,
            "total_alloc_cpu": total_alloc_cpu,
            "total_alloc_mem": total_alloc_mem,
            "processes": [
                {
                    "pid": p.pid,
                    "workload": p.workload_type,
                    "priority": p.priority,
                    "req_cpu": p.requested_cpu,
                    "req_mem": p.requested_mem,
                    "alloc_cpu": p.allocated_cpu,
                    "alloc_mem": p.allocated_mem,
                    "wait": p.wait_time,
                }
                for p in self.processes
            ],
            "manual_overrides": dict(self.manual_request_overrides),
        }


class InteractiveTerminalRunner:
    def __init__(self, allocator, tick_delay=DEFAULT_TICK_DELAY_SECONDS):
        self.allocator = allocator
        self.tick_delay = max(0.2, float(tick_delay))
        self.command_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.messages = deque(maxlen=8)
        self.tick = 0
        self.color_enabled = self._enable_color_support()
        self.C = {
            "reset": "\033[0m",
            "bold": "\033[1m",
            "cyan": "\033[36m",
            "blue": "\033[94m",
            "green": "\033[92m",
            "yellow": "\033[93m",
            "red": "\033[91m",
            "magenta": "\033[95m",
            "dim": "\033[2m",
            "white": "\033[97m",
        }

    def _enable_color_support(self):
        if os.environ.get("NO_COLOR"):
            return False
        if os.name != "nt":
            return True
        try:
            kernel32 = ctypes.windll.kernel32
            h_out = kernel32.GetStdHandle(-11)
            mode = ctypes.c_ulong()
            if kernel32.GetConsoleMode(h_out, ctypes.byref(mode)) == 0:
                return False
            new_mode = mode.value | 0x0004
            if kernel32.SetConsoleMode(h_out, new_mode) == 0:
                return False
            return True
        except Exception:
            return False

    def _c(self, text, color=None, bold=False):
        if not self.color_enabled:
            return text
        prefix = ""
        if bold:
            prefix += self.C["bold"]
        if color and color in self.C:
            prefix += self.C[color]
        return f"{prefix}{text}{self.C['reset']}"

    def _state_color(self, state):
        if state == "cpu+mem":
            return "red"
        if state in {"cpu", "memory"}:
            return "yellow"
        return "green"

    def _collision_color(self, collision):
        if not collision or not collision.get("active"):
            return "green"
        if collision.get("level") == "critical":
            return "red"
        return "yellow"

    def _collision_banner(self, collision):
        if not collision or not collision.get("active"):
            return "Collision: none"
        ctype = str(collision.get("type", "unknown")).upper()
        count = int(collision.get("count", 0))
        return f"Collision: {ctype} ({count} contenders)"

    def _add_collision_tick_msg(self, snapshot):
        collision = snapshot.get("collision") or {}
        if not collision.get("active"):
            return
        if collision.get("level") == "critical":
            self._add_msg(
                f"ALERT tick {snapshot['tick']}: COLLISION CPU+MEM ({collision.get('count', 0)} contenders)"
            )
            return
        ctype = str(collision.get("type", "unknown")).upper()
        self._add_msg(
            f"ALERT tick {snapshot['tick']}: COLLISION {ctype} ({collision.get('count', 0)} contenders)"
        )

    def _workload_color(self, workload):
        if workload.startswith("CPU"):
            return "red"
        if workload.startswith("Memo"):
            return "magenta"
        return "cyan"

    def _input_worker(self):
        while not self.stop_event.is_set():
            try:
                raw = input("cmd> ").strip()
            except EOFError:
                self.command_queue.put("quit")
                break
            if raw:
                self.command_queue.put(raw)

    def _add_msg(self, msg):
        ts = time.strftime("%H:%M:%S")
        self.messages.appendleft(f"[{ts}] {msg}")

    def _format_bar(self, value, cap, width=28, ch="#"):
        if cap <= 0:
            cap = 1.0
        pct = max(0.0, min(1.0, value / cap))
        fill = int(round(pct * width))
        return (ch * fill) + ("." * (width - fill))

    def _render(self, snapshot):
        clear_cmd = "cls" if os.name == "nt" else "clear"
        os.system(clear_cmd)

        accent = self._state_color(snapshot["bottleneck"])
        width = 92
        line = "=" * width
        project_title = "ADVANCE RESOURCE ALLOCATION IN MULTI PROGRAMMING SYSTEMS"
        subtitle = "Terminal-Based Interactive Simulator"
        status = f"Real-time Adaptive Allocation | Mode: {snapshot['mode'].upper()} | Tick {snapshot['tick']}"

        print(self._c(line, accent, bold=True))
        print(self._c(project_title.center(width), "red", bold=True))
        print(self._c(subtitle.center(width), "white", bold=True))
        print(self._c(status.center(width), "yellow" if snapshot["mode"] == "manual" else "green"))
        print(self._c(line, accent, bold=True))
        print(
            self._c(f"Tick {snapshot['tick']:4d}", "blue", bold=True)
            + " | "
            + self._c(f"Mode: {snapshot['mode'].upper():6s}", "yellow" if snapshot["mode"] == "manual" else "green", bold=True)
            + " | "
            + self._c(f"Bottleneck: {snapshot['bottleneck']:7s}", accent, bold=True)
            + " | "
            + self._c(f"CPU Pressure: {snapshot['cpu_pressure']*100:5.1f}%", "cyan")
            + " | "
            + self._c(f"MEM Pressure: {snapshot['mem_pressure']*100:5.1f}%", "magenta")
        )
        print(
            self._c("Alert Profile: ", "dim")
            + self._c(snapshot.get("alert_profile", "balanced").upper(), "white", bold=True)
            + " | "
            + self._c(self._collision_banner(snapshot.get("collision", {})), self._collision_color(snapshot.get("collision", {})), bold=True)
        )
        print(
            self._c("System Avail -> ", "dim")
            + self._c(f"CPU: {snapshot['real_avail_cpu']:6.2f}%", "cyan")
            + " | "
            + self._c(f"MEM: {snapshot['real_avail_mem']:9.2f} MB", "magenta")
        )
        print(
            self._c("Totals      -> ", "dim")
            + self._c(f"ReqCPU: {snapshot['total_req_cpu']:6.2f}", "yellow")
            + " | "
            + self._c(f"AllocCPU: {snapshot['total_alloc_cpu']:6.2f}", "green")
            + " | "
            + self._c(f"ReqMEM: {snapshot['total_req_mem']:9.2f}", "yellow")
            + " | "
            + self._c(f"AllocMEM: {snapshot['total_alloc_mem']:9.2f}", "green")
        )

        max_req_cpu = max(1.0, max((p["req_cpu"] for p in snapshot["processes"]), default=1.0))
        max_req_mem = max(1.0, max((p["req_mem"] for p in snapshot["processes"]), default=1.0))

        print(self._c("-" * 92, "dim"))
        print(self._c("Per-Process Demographs", "white", bold=True))
        print(self._c("Legend: CPU[# req, + alloc]  MEM[* req, = alloc]", "dim"))
        print(self._c("-" * 92, "dim"))
        for p in sorted(snapshot["processes"], key=lambda row: row["pid"]):
            cpu_req_bar = self._format_bar(p["req_cpu"], max_req_cpu, ch="#")
            cpu_alloc_bar = self._format_bar(p["alloc_cpu"], max_req_cpu, ch="+")
            mem_req_bar = self._format_bar(p["req_mem"], max_req_mem, ch="*")
            mem_alloc_bar = self._format_bar(p["alloc_mem"], max_req_mem, ch="=")

            print(
                self._c(f"PID {p['pid']:2d}", "blue", bold=True)
                + " "
                + self._c(f"{p['workload'][:4]:4s}", self._workload_color(p["workload"]))
                + f" Prio:{p['priority']:2d} Wait:{p['wait']:2d} | "
                + self._c(f"CPU R:{p['req_cpu']:5.1f}", "yellow")
                + " "
                + self._c(f"A:{p['alloc_cpu']:5.1f}", "green")
            )
            print("   CPU req  [" + self._c(cpu_req_bar, "yellow") + "]")
            print("   CPU alloc[" + self._c(cpu_alloc_bar, "green") + "]")
            print("   MEM req  [" + self._c(mem_req_bar, "magenta") + "]  " + self._c(f"R:{p['req_mem']:7.1f}", "yellow"))
            print("   MEM alloc[" + self._c(mem_alloc_bar, "cyan") + "]  " + self._c(f"A:{p['alloc_mem']:7.1f}", "green"))

        print(self._c("-" * 92, "dim"))
        print(self._c("Manual Overrides:", "white", bold=True))
        if snapshot["manual_overrides"]:
            for pid, values in sorted(snapshot["manual_overrides"].items(), key=lambda x: x[0]):
                print(
                    "  "
                    + self._c(f"PID {pid}", "blue")
                    + ": "
                    + self._c(f"CPU {values['req_cpu']:.1f}%", "yellow")
                    + " | "
                    + self._c(f"MEM {values['req_mem']:.1f} MB", "magenta")
                )
        else:
            print("  " + self._c("(none)", "dim"))

        pred = self.allocator.predict_next_tick()
        if pred:
            print(
                self._c("ML Next Tick Estimate -> ", "dim")
                + self._c(f"CPU Avail: {pred[0]:.1f}%", "cyan")
                + " | "
                + self._c(f"MEM Avail: {pred[1]:.1f} MB", "magenta")
            )

        print(self._c("-" * 92, "dim"))
        print(self._c("Commands: help | mode manual|auto | set <pid> <cpu%> <memMB> | del <pid> | clear | quit", "white", bold=True))
        if self.messages:
            print(self._c("Messages:", "white", bold=True))
            for msg in self.messages:
                print("  " + self._c(msg, "dim"))

    def _handle_command(self, raw):
        parts = shlex.split(raw)
        if not parts:
            return
        cmd = parts[0].lower()

        try:
            if cmd in {"quit", "exit", "q"}:
                self._add_msg("Stopping terminal simulation.")
                self.stop_event.set()
                return

            if cmd == "help":
                self._add_msg("help | mode manual|auto | set <pid> <cpu%> <memMB> | del <pid> | clear | quit")
                return

            if cmd == "mode" and len(parts) == 2:
                target = parts[1].lower()
                if target not in {"manual", "auto"}:
                    self._add_msg("Invalid mode. Use: mode manual|auto")
                    return
                self.allocator.set_manual_mode(target == "manual")
                self._add_msg(f"Mode set to {target}.")
                return

            if cmd == "set" and len(parts) == 4:
                pid = int(parts[1])
                cpu = float(parts[2])
                mem = float(parts[3])
                if pid not in {p.pid for p in self.allocator.processes}:
                    self._add_msg(f"PID {pid} not found.")
                    return
                self.allocator.set_manual_request(pid, cpu, mem)
                self._add_msg(f"Manual override set for PID {pid}: CPU {cpu:.1f}, MEM {mem:.1f}")
                return

            if cmd == "del" and len(parts) == 2:
                pid = int(parts[1])
                self.allocator.remove_manual_request(pid)
                self._add_msg(f"Manual override removed for PID {pid}")
                return

            if cmd == "clear":
                self.allocator.clear_manual_requests()
                self._add_msg("All manual overrides cleared.")
                return

            self._add_msg("Unknown command. Type: help")
        except ValueError:
            self._add_msg("Invalid number format in command.")

    def run(self):
        self._add_msg("Terminal simulation started.")
        thread = threading.Thread(target=self._input_worker, daemon=True)
        thread.start()

        try:
            while not self.stop_event.is_set():
                while True:
                    try:
                        cmd = self.command_queue.get_nowait()
                    except queue.Empty:
                        break
                    self._handle_command(cmd)

                if self.stop_event.is_set():
                    break

                self.tick += 1
                snapshot = self.allocator.allocate_resources(self.tick)
                self._add_collision_tick_msg(snapshot)
                self._render(snapshot)
                time.sleep(self.tick_delay)
        finally:
            self.stop_event.set()


def main():
    allocator = AdaptiveAllocator()
    runner = InteractiveTerminalRunner(allocator, tick_delay=DEFAULT_TICK_DELAY_SECONDS)
    try:
        runner.run()
    finally:
        allocator.close()


if __name__ == "__main__":
    main()
