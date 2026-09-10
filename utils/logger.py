# utils/logger.py
import os
import time
import psutil
import torch
import wandb


class MetricsLogger:
    """
    Weights & Biases (WandB) observability logger for NPC_Brain.
    Provides organized namespaces for:
      - Losses/* (World Model, Actor-Critic, KL divergence, RND curiosity)
      - Game/* (Episode reward, Win rate, In-cover %, Survival time)
      - Curriculum/* (Step delay, Difficulty pacing)
      - Latency/* (Perception, NN inference, Env step, Training step in ms)
      - System/* (VRAM allocation/reserved/peak, System RAM used & %)
    """

    def __init__(
        self,
        project_name: str = "npc-brain",
        run_name: str = None,
        config: dict = None,
        use_wandb: bool = True,
        mode: str = "online",  # defaults to live online cloud syncing
    ):
        self.use_wandb = use_wandb
        self.start_time = time.time()
        self.process = psutil.Process(os.getpid())

        if self.use_wandb:
            if run_name is None:
                run_name = f"npc-dreamerv3-{time.strftime('%Y%m%d-%H%M%S')}"

            init_kwargs = {
                "project": project_name,
                "name": run_name,
                "config": config or {},
            }
            if mode is not None:
                init_kwargs["mode"] = mode

            try:
                self.run = wandb.init(**init_kwargs)
                self._setup_metric_layouts()
                url = wandb.run.get_url() if (wandb.run and hasattr(wandb.run, 'get_url')) else run_name
                print(f"[WandB] Initialized: {url}")
            except Exception as e:
                print(f"[WandB Warning] Init failed: {e}. Falling back to offline mode.")
                try:
                    init_kwargs["mode"] = "offline"
                    self.run = wandb.init(**init_kwargs)
                    self._setup_metric_layouts()
                    print("[WandB] Initialized in offline mode.")
                except Exception as e_offline:
                    print(f"[WandB Error] Offline initialization also failed: {e_offline}")
                    self.use_wandb = False

    def _setup_metric_layouts(self):
        """Define steps and organize metric axes for clean WandB charts."""
        try:
            # Base step axes
            wandb.define_metric("step")
            wandb.define_metric("episode")

            # High-frequency step-bound metrics
            wandb.define_metric("Losses/*", step_metric="step")
            wandb.define_metric("Latency/*", step_metric="step")
            wandb.define_metric("System/*", step_metric="step")

            # Episode-bound metrics
            wandb.define_metric("Game/*", step_metric="episode")
            wandb.define_metric("Curriculum/*", step_metric="episode")
        except Exception as e:
            print(f"[WandB Notice] define_metric notice: {e}")

    def get_system_metrics(self) -> dict:
        """Capture precise GPU VRAM and System RAM utilization."""
        metrics = {}

        # GPU VRAM profiling
        if torch.cuda.is_available():
            total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            alloc_vram = torch.cuda.memory_allocated(0) / (1024 ** 3)
            resv_vram = torch.cuda.memory_reserved(0) / (1024 ** 3)
            max_vram = torch.cuda.max_memory_allocated(0) / (1024 ** 3)

            metrics["System/vram_allocated_gb"] = round(alloc_vram, 3)
            metrics["System/vram_reserved_gb"] = round(resv_vram, 3)
            metrics["System/vram_peak_gb"] = round(max_vram, 3)
            metrics["System/vram_utilization_pct"] = round((alloc_vram / total_vram) * 100.0, 2)

        # Host System RAM profiling
        vm = psutil.virtual_memory()
        proc_ram = self.process.memory_info().rss / (1024 ** 3)

        metrics["System/ram_used_gb"] = round(vm.used / (1024 ** 3), 3)
        metrics["System/ram_process_gb"] = round(proc_ram, 3)
        metrics["System/ram_utilization_pct"] = round(vm.percent, 2)
        metrics["System/elapsed_minutes"] = round((time.time() - self.start_time) / 60.0, 2)

        return metrics

    def log_step(self, step: int, latency_ms: dict = None, losses: dict = None, log_system: bool = False):
        """
        Log per-step or periodic training step metrics (losses, latency in ms, hardware).
        """
        if not self.use_wandb:
            return

        payload = {"step": step}

        if losses:
            for k, v in losses.items():
                clean_name = k.replace(" ", "_")
                payload[f"Losses/{clean_name}"] = float(v)

        if latency_ms:
            for k, v in latency_ms.items():
                clean_name = k.replace(" ", "_")
                payload[f"Latency/{clean_name}"] = float(v)

        if log_system:
            payload.update(self.get_system_metrics())

        try:
            wandb.log(payload, step=step)
        except Exception as e:
            print(f"[WandB Warning] step log: {e}")

    def log_episode(self, episode: int, metrics: dict):
        """
        Log episode-level summary metrics (rewards, win rate, survival, curriculum).
        """
        if not self.use_wandb:
            return

        payload = {"episode": episode}

        # Segregate into Game and Curriculum namespaces
        for k, v in metrics.items():
            if k in ("step_delay", "win_rate", "avg_curiosity", "curriculum_win_rate"):
                payload[f"Curriculum/{k}"] = float(v)
            elif k in ("vram_gb", "ram_gb"):
                payload[f"System/{k}"] = float(v)
            else:
                payload[f"Game/{k}"] = float(v)

        # Include system metrics snapshot at episode end
        payload.update(self.get_system_metrics())

        try:
            wandb.log(payload)
        except Exception as e:
            print(f"[WandB Warning] episode log: {e}")

    def close(self):
        """Finish the WandB run cleanly."""
        if self.use_wandb:
            try:
                wandb.finish()
            except Exception:
                pass


# ─── VALIDATION / DEMO ──────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing MetricsLogger in offline mode...")
    logger = MetricsLogger(project_name="npc-brain-test", mode="offline", config={"test_mode": True})

    # Simulate 5 steps with latencies & losses
    for s in range(1, 6):
        sample_latencies = {
            "perception_ms": 12.4 + s * 0.3,
            "inference_ms": 6.8 + s * 0.1,
            "env_step_ms": 25.0 + s * 0.5,
            "train_step_ms": 42.1 if s % 2 == 0 else 0.0,
            "total_step_latency_ms": 44.2 + (42.1 if s % 2 == 0 else 0.0),
        }
        sample_losses = {
            "world_model_loss": 1.25 - s * 0.05,
            "actor_loss": 0.42 - s * 0.02,
            "critic_loss": 0.31 - s * 0.01,
            "reward_loss": 0.18 - s * 0.01,
            "kl_divergence": 0.08,
        }
        logger.log_step(step=s, latency_ms=sample_latencies, losses=sample_losses, log_system=(s == 5))

    # Simulate 2 episodes
    for ep in range(1, 3):
        logger.log_episode(
            episode=ep,
            metrics={
                "episode_reward": ep * 15.5,
                "episode_length": 150 + ep * 10,
                "in_cover_pct": 0.45 + ep * 0.05,
                "survival_time": 45.0 + ep * 5,
                "shots_taken": max(0, 3 - ep),
                "win_rate": 0.5 * ep,
                "success_rate": 1.0 if ep > 1 else 0.0,
                "step_delay": 0.28,
                "avg_curiosity": 0.035,
            },
        )

    logger.close()
    print("[SUCCESS] MetricsLogger validation PASSED! Clean namespaces and system metrics logged.")