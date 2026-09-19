import psutil
import time
import random
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

DEFAULT_DB_NAME = "allocation_logs.db"
DEFAULT_TICK_COUNT = 15
DEFAULT_TICK_DELAY_SECONDS = 1
MIN_PREDICTION_POINTS = 5
EWMA_ALPHA = 0.35
BOTTLENECK_PRESSURE_THRESHOLD = 0.85
MIN_RESOURCE_SLICE = 0.5
DEFAULT_PROCESS_SPECS = [
    (1, 'CPU-Intensive'),
    (2, 'Memory-Intensive'),
    (3, 'Balanced'),
    (4, 'CPU-Intensive'),
    (5, 'Balanced')
]

class VirtualProcess:
    """Simulates a process with specific resource demands."""
    def __init__(self, pid, workload_type):
        self.pid = pid
        self.workload_type = workload_type # 'CPU-Intensive', 'Memory-Intensive', 'Balanced'
        self.priority = 5 # 1 (Highest) to 10 (Lowest)
        self.allocated_cpu = 0.0
        self.allocated_mem = 0.0 # in MB
        self.wait_time = 0
        self.starvation_credit = 0.0

    def generate_requests(self):
        # Generate resource requests based on workload profile
        if self.workload_type == 'CPU-Intensive':
            req_cpu = random.uniform(15.0, 40.0) # High CPU demand
            req_mem = random.uniform(10.0, 50.0)
        elif self.workload_type == 'Memory-Intensive':
            req_cpu = random.uniform(1.0, 10.0)
            req_mem = random.uniform(200.0, 800.0) # High Mem demand
        else: # Balanced
            req_cpu = random.uniform(5.0, 20.0)
            req_mem = random.uniform(50.0, 300.0)
            
        return req_cpu, req_mem

class AdaptiveAllocator:
    """
    Simulates an OS scheduler that adaptively allocates real available 
    system resources to virtual processes.
    """
    def __init__(self, db_name=DEFAULT_DB_NAME):
        self.db_name = db_name
        self.processes = self._create_default_processes()
        self.conn = None
        self.cursor = None
        self.cpu_pressure_ewma = 0.0
        self.mem_pressure_ewma = 0.0
        self._setup_db()
        psutil.cpu_percent(interval=0.1) # Prime the CPU monitor

    def _create_default_processes(self):
        return [VirtualProcess(pid, workload_type) for pid, workload_type in DEFAULT_PROCESS_SPECS]

    def _setup_db(self):
        self.conn = sqlite3.connect(self.db_name)
        self.cursor = self.conn.cursor()
        self._initialize_allocation_stats_table()
        self.conn.commit()

    def _initialize_allocation_stats_table(self):
        self.cursor.execute("DROP TABLE IF EXISTS AllocationStats")
        self.cursor.execute("""
            CREATE TABLE AllocationStats (
                timestamp TEXT,
                real_avail_cpu REAL,
                real_avail_mem REAL,
                total_requested_cpu REAL,
                total_requested_mem REAL,
                total_allocated_cpu REAL,
                total_allocated_mem REAL,
                cpu_pressure REAL,
                mem_pressure REAL,
                bottleneck_state TEXT
            )
        """)

    def _measure_system_capacity(self):
        real_used_cpu = psutil.cpu_percent(interval=None)
        real_avail_cpu = max(0.0, 100.0 - real_used_cpu)

        mem_info = psutil.virtual_memory()
        real_avail_mem_mb = mem_info.available / (1024 * 1024)

        return real_avail_cpu, real_avail_mem_mb

    def _collect_requests(self):
        requests = []
        total_req_cpu = 0
        total_req_mem = 0

        for process in self.processes:
            if process.wait_time > 2:
                process.priority = max(1, process.priority - 2)
                process.wait_time = 0

            req_cpu, req_mem = process.generate_requests()
            requests.append({'process': process, 'req_cpu': req_cpu, 'req_mem': req_mem})
            total_req_cpu += req_cpu
            total_req_mem += req_mem

        requests.sort(key=lambda item: item['process'].priority)
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
        if process.workload_type == 'CPU-Intensive' and cpu_pressure > 0.7:
            workload_bias += 0.25
        elif process.workload_type == 'Memory-Intensive' and mem_pressure > 0.7:
            workload_bias += 0.25
        elif process.workload_type == 'Balanced':
            workload_bias += 0.1

        return max(0.1, fairness_boost * priority_boost * workload_bias)

    def _adaptive_allocate(self, requests, avail_cpu_pool, avail_mem_pool, cpu_pressure, mem_pressure):
        if not requests:
            return 0.0, 0.0

        weights = {
            request['process'].pid: self._compute_process_weight(request['process'], cpu_pressure, mem_pressure)
            for request in requests
        }
        weight_sum = sum(weights.values())

        process_allocations = {}
        for request in requests:
            process = request['process']
            process_allocations[process.pid] = {
                'cpu': 0.0,
                'mem': 0.0,
                'req_cpu': request['req_cpu'],
                'req_mem': request['req_mem'],
                'process': process,
            }

        # Pass 1: weighted fair-share allocation under current pressure.
        for request in requests:
            process = request['process']
            pid = process.pid
            weight = weights[pid]

            cpu_target = avail_cpu_pool * (weight / weight_sum)
            mem_target = avail_mem_pool * (weight / weight_sum)

            cpu_alloc = min(request['req_cpu'], max(MIN_RESOURCE_SLICE, cpu_target))
            mem_alloc = min(request['req_mem'], max(MIN_RESOURCE_SLICE, mem_target))

            cpu_alloc = min(cpu_alloc, avail_cpu_pool)
            mem_alloc = min(mem_alloc, avail_mem_pool)

            process_allocations[pid]['cpu'] += cpu_alloc
            process_allocations[pid]['mem'] += mem_alloc
            avail_cpu_pool -= cpu_alloc
            avail_mem_pool -= mem_alloc

        # Pass 2: reallocate leftover resources to highest unmet weighted demand.
        for _ in range(2):
            if avail_cpu_pool <= 0 and avail_mem_pool <= 0:
                break

            cpu_unmet_weight = sum(
                max(0.0, process_allocations[r['process'].pid]['req_cpu'] - process_allocations[r['process'].pid]['cpu'])
                * weights[r['process'].pid]
                for r in requests
            )
            mem_unmet_weight = sum(
                max(0.0, process_allocations[r['process'].pid]['req_mem'] - process_allocations[r['process'].pid]['mem'])
                * weights[r['process'].pid]
                for r in requests
            )

            for request in requests:
                process = request['process']
                pid = process.pid
                entry = process_allocations[pid]
                weight = weights[pid]

                cpu_unmet = max(0.0, entry['req_cpu'] - entry['cpu'])
                mem_unmet = max(0.0, entry['req_mem'] - entry['mem'])

                if avail_cpu_pool > 0 and cpu_unmet > 0 and cpu_unmet_weight > 0:
                    cpu_share = avail_cpu_pool * ((cpu_unmet * weight) / cpu_unmet_weight)
                    cpu_gain = min(cpu_unmet, cpu_share, avail_cpu_pool)
                    entry['cpu'] += cpu_gain
                    avail_cpu_pool -= cpu_gain

                if avail_mem_pool > 0 and mem_unmet > 0 and mem_unmet_weight > 0:
                    mem_share = avail_mem_pool * ((mem_unmet * weight) / mem_unmet_weight)
                    mem_gain = min(mem_unmet, mem_share, avail_mem_pool)
                    entry['mem'] += mem_gain
                    avail_mem_pool -= mem_gain

        total_alloc_cpu = 0.0
        total_alloc_mem = 0.0
        for request in requests:
            process = request['process']
            pid = process.pid
            alloc_cpu = process_allocations[pid]['cpu']
            alloc_mem = process_allocations[pid]['mem']

            process.allocated_cpu = alloc_cpu
            process.allocated_mem = alloc_mem
            total_alloc_cpu += alloc_cpu
            total_alloc_mem += alloc_mem

            cpu_fulfilled = alloc_cpu >= request['req_cpu'] * 0.95
            mem_fulfilled = alloc_mem >= request['req_mem'] * 0.95

            if cpu_fulfilled and mem_fulfilled:
                process.priority = min(10, process.priority + 1)
                process.wait_time = 0
                process.starvation_credit = max(0.0, process.starvation_credit - 0.25)
            else:
                process.wait_time += 1
                process.priority = max(1, process.priority - 1)
                process.starvation_credit = min(2.0, process.starvation_credit + 0.2)

        return total_alloc_cpu, total_alloc_mem

    def _apply_request_allocation(self, request, avail_cpu_pool, avail_mem_pool):
        process = request['process']
        cpu_needed = request['req_cpu']
        mem_needed = request['req_mem']

        if avail_cpu_pool >= cpu_needed:
            process.allocated_cpu = cpu_needed
            avail_cpu_pool -= cpu_needed
        else:
            process.allocated_cpu = avail_cpu_pool
            avail_cpu_pool = 0

        if avail_mem_pool >= mem_needed:
            process.allocated_mem = mem_needed
            avail_mem_pool -= mem_needed
        else:
            process.allocated_mem = avail_mem_pool
            avail_mem_pool = 0

        if process.allocated_cpu < cpu_needed or process.allocated_mem < mem_needed:
            process.wait_time += 1
        else:
            process.priority = min(10, process.priority + 1)
            process.wait_time = 0

        return avail_cpu_pool, avail_mem_pool

    def _print_request_status(self, process, cpu_needed, mem_needed):
        now_ts = pd.Timestamp.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"[{now_ts}] PID {process.pid} ({process.workload_type[:4]:^4}) | Prio: {process.priority:2d} | "
              f"Req: {cpu_needed:4.1f}% CPU, {mem_needed:4.1f}MB Mem | "
              f"Alloc: {process.allocated_cpu:4.1f}% CPU, {process.allocated_mem:4.1f}MB Mem")

    def _record_telemetry(self, real_avail_cpu, real_avail_mem_mb, total_req_cpu, total_req_mem,
                          total_alloc_cpu, total_alloc_mem, cpu_pressure, mem_pressure,
                          bottleneck_state):
        timestamp = pd.Timestamp.now().strftime("%H:%M:%S")
        try:
            self.cursor.execute("INSERT INTO AllocationStats VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                                (timestamp, real_avail_cpu, real_avail_mem_mb,
                                 total_req_cpu, total_req_mem, total_alloc_cpu, total_alloc_mem,
                                 cpu_pressure, mem_pressure, bottleneck_state))
            self.conn.commit()
        except sqlite3.Error as error:
            print(f"Telemetry write failed: {error}")

    def _describe_trend(self, slope):
        if slope > 0.05:
            return "increasing"
        if slope < -0.05:
            return "decreasing"
        return "stable"

    def allocate_resources(self, tick):
        real_avail_cpu, real_avail_mem_mb = self._measure_system_capacity()
        requests, total_req_cpu, total_req_mem = self._collect_requests()

        cpu_pressure = self._pressure(total_req_cpu, real_avail_cpu)
        mem_pressure = self._pressure(total_req_mem, real_avail_mem_mb)
        self.cpu_pressure_ewma = self._update_ewma(self.cpu_pressure_ewma, cpu_pressure)
        self.mem_pressure_ewma = self._update_ewma(self.mem_pressure_ewma, mem_pressure)
        bottleneck_state = self._get_bottleneck_state(self.cpu_pressure_ewma, self.mem_pressure_ewma)

        print(
            f"\n--- Tick {tick} | Real Avail: [CPU: {real_avail_cpu:.1f}%] [Mem: {real_avail_mem_mb:.1f} MB] "
            f"| Pressure: [CPU: {self.cpu_pressure_ewma:.2f}] [Mem: {self.mem_pressure_ewma:.2f}] "
            f"| Bottleneck: {bottleneck_state} ---"
        )

        total_alloc_cpu, total_alloc_mem = self._adaptive_allocate(
            requests,
            real_avail_cpu,
            real_avail_mem_mb,
            self.cpu_pressure_ewma,
            self.mem_pressure_ewma,
        )

        for request in requests:
            process = request['process']
            self._print_request_status(process, request['req_cpu'], request['req_mem'])

        self._record_telemetry(
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

    def get_data(self):
        try:
            return pd.read_sql("SELECT * FROM AllocationStats", self.conn)
        except (sqlite3.Error, ValueError) as error:
            print(f"Unable to read allocation data: {error}")
            return pd.DataFrame()

    def predict_next_tick(self):
        """Uses ML to predict system resource availability trend for next tick."""
        df = self.get_data()
        if len(df) < MIN_PREDICTION_POINTS:
            print("\n[ML Predict] Not enough data to generate a trend prediction.")
            return
        
        X = np.arange(len(df)).reshape(-1, 1)
        model_cpu = LinearRegression().fit(X, df["real_avail_cpu"].values)
        model_mem = LinearRegression().fit(X, df["real_avail_mem"].values)
        
        next_t = np.array([[len(df)]])
        pred_cpu = max(0, min(100, model_cpu.predict(next_t)[0]))
        pred_mem = max(0, model_mem.predict(next_t)[0])
        cpu_trend = self._describe_trend(model_cpu.coef_[0])
        mem_trend = self._describe_trend(model_mem.coef_[0])

        print(
            "\n[ML Predict] Trend for Next Tick -> "
            f"Est. Avail CPU: {pred_cpu:.1f}% ({cpu_trend}), "
            f"Est. Avail Mem: {pred_mem:.1f} MB ({mem_trend})"
        )

    def plot_analytics(self):
        df = self.get_data()
        if df.empty:
            print("No telemetry data available for plotting.")
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # CPU Subplot
        ax1.plot(df.index, df["real_avail_cpu"], label="Real Avail CPU Capacity", linestyle="--", color="grey", linewidth=2)
        ax1.plot(df.index, df["total_requested_cpu"], label="Requested CPU (All Procs)", color="orange", marker='o')
        ax1.plot(df.index, df["total_allocated_cpu"], label="Allocated CPU (All Procs)", color="blue", marker='s')
        ax1.set_ylabel("CPU (%)")
        ax1.set_title("Adaptive CPU Allocation vs Real Constraints")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Memory Subplot
        ax2.plot(df.index, df["real_avail_mem"], label="Real Avail Mem Capacity", linestyle="--", color="grey", linewidth=2)
        ax2.plot(df.index, df["total_requested_mem"], label="Requested Mem (All Procs)", color="red", marker='o')
        ax2.plot(df.index, df["total_allocated_mem"], label="Allocated Mem (All Procs)", color="green", marker='s')
        ax2.set_ylabel("Memory (MB)")
        ax2.set_title("Adaptive Memory Allocation vs Real Constraints")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.xlabel("Simulation Tick")
        fig.suptitle("Adaptive Resource Allocation Analysis", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.show()

    def close(self):
        if self.conn:
            try:
                self.conn.close()
            finally:
                self.conn = None
                self.cursor = None


def validate_runtime_config():
    if DEFAULT_TICK_COUNT <= 0:
        raise ValueError("DEFAULT_TICK_COUNT must be greater than 0")
    if DEFAULT_TICK_DELAY_SECONDS < 0:
        raise ValueError("DEFAULT_TICK_DELAY_SECONDS cannot be negative")
    if len(DEFAULT_PROCESS_SPECS) == 0:
        raise ValueError("DEFAULT_PROCESS_SPECS must include at least one process")
    if not (0 < EWMA_ALPHA <= 1):
        raise ValueError("EWMA_ALPHA must be in the range (0, 1]")
    if not (0 < BOTTLENECK_PRESSURE_THRESHOLD <= 1):
        raise ValueError("BOTTLENECK_PRESSURE_THRESHOLD must be in the range (0, 1]")
    if MIN_RESOURCE_SLICE < 0:
        raise ValueError("MIN_RESOURCE_SLICE cannot be negative")

def main():
    print("Initializing Adaptive OS Resource Allocation Simulation...")
    print("Binding simulated requests to REAL system availability bounds.\n")

    validate_runtime_config()
    print("Runtime configuration validation passed.")
    
    allocator = AdaptiveAllocator()
    try:
        for tick in range(1, DEFAULT_TICK_COUNT + 1):
            allocator.allocate_resources(tick)
            time.sleep(DEFAULT_TICK_DELAY_SECONDS)
            
        allocator.predict_next_tick()
        
        print("\nOpening analytical plot (Close the plot window to exit program)...")
        allocator.plot_analytics()
    except Exception as e:
        print(f"Error during simulation: {e}")
    finally:
        allocator.close()

if __name__ == "__main__":
    main()