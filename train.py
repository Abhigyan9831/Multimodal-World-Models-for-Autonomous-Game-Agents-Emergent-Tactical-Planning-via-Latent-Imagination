
import torch
import numpy as np
import time
import os
import signal

os.environ['PYTHONPATH'] = r'E:\NPC_Brain'

from environment.ue5_env import UE5Env
from dreamerv3.config import DreamerConfig
from dreamerv3.trainer import DreamerTrainer
from utils.replay_buffer import ReplayBuffer
from utils.rewards import RewardFunction
from utils.logger import MetricsLogger
from utils.opponent_model import PlayerHistoryTracker
from utils.rnd import RNDModule
from utils.curriculum import CurriculumManager


# ── CONFIG ────────────────────────────────────────────────────────────────────

TOTAL_EPISODES      = 10000
MAX_STEPS_PER_EP    = 200
SAVE_EVERY          = 50
LOG_EVERY           = 1
TRAIN_EVERY         = 5
MIN_BUFFER_EPISODES = 5
RESUME              = True
IDLE_TIMEOUT        = 30    # seconds — pause if player inactive

# Curriculum manages STEP_DELAY automatically; this is the starting value
INITIAL_STEP_DELAY  = 0.3

# RND curiosity weight — how much intrinsic reward to add per step
RND_WEIGHT          = 0.1

ACTION_NAMES = {0:'Fwd', 1:'Back', 2:'Left', 3:'Right', 4:'Stay', 5:'Sprint'}


# ── GRACEFUL SHUTDOWN ─────────────────────────────────────────────────────────

shutdown_flag = False

def handle_shutdown(sig, frame):
    global shutdown_flag
    print("\n Saving and shutting down")
    shutdown_flag = True

signal.signal(signal.SIGINT, handle_shutdown)


# ── MAIN ──────────────────────────────────────────────────────────────────────

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"VRAM:   {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    # ── Initialize ─────────────────────────────────────────────────
    print("\n 1. Connecting to UE5")
    env = UE5Env()

    print("2. Loading World Model")
    config  = DreamerConfig()
    trainer = DreamerTrainer(config, device=str(device))

    print("3. Replay buffer")
    buffer = ReplayBuffer(
        capacity    = config.replay_capacity,
        vec_obs_dim = config.vector_obs_dim,   # 15
        act_dim     = config.act_dim,
        seq_len     = config.seq_len
    )

    print("4. Reward function")
    reward_fn = RewardFunction()

    print("5. Opponent model tracker")
    opp_tracker = PlayerHistoryTracker()

    print("6. RND curiosity module")
    rnd = RNDModule(obs_dim=config.vector_obs_dim, device=str(device))

    print("7. Curriculum manager")
    curriculum = CurriculumManager(initial_delay=INITIAL_STEP_DELAY)

    print("8. Logger (Weights & Biases)")
    wandb_config = {
        "total_episodes": TOTAL_EPISODES,
        "max_steps_per_ep": MAX_STEPS_PER_EP,
        "train_every": TRAIN_EVERY,
        "initial_step_delay": INITIAL_STEP_DELAY,
        "rnd_weight": RND_WEIGHT,
        "batch_size": config.batch_size,
        "seq_len": config.seq_len,
        "obs_dim": config.obs_dim,
        "vector_obs_dim": config.vector_obs_dim,
        "act_dim": config.act_dim,
        "device": str(device),
    }
    logger = MetricsLogger(
        project_name="npc-brain",
        run_name=f"dreamerv3-curriculum-{time.strftime('%Y%m%d-%H%M%S')}",
        config=wandb_config,
        use_wandb=True
    )

    # ── Resume ─────────────────────────────────────────────────────
    start_episode = 0
    if RESUME:
        start_episode = trainer.load_checkpoint()

    print(f"\n Training ->")
    print(f"   Episodes:  {start_episode} -> {TOTAL_EPISODES}")
    print(f"   Obs dim:   {config.obs_dim}  (512 CNN + 15 telemetry)")
    print(f"   Features:  OpponentModel + RND Curiosity + Curriculum + WandB Tracking")
    print(f"   Press Ctrl+C to save and exit\n")

    global_step     = 0
    idle_timer      = time.time()
    training_active = True

    for episode in range(start_episode, TOTAL_EPISODES):
        if shutdown_flag:
            break

        # ── Wait for player activity ────────────────────────────────
        if not training_active:
            print("  Waiting for player activity.")
            while True:
                game_state = env.get_game_state()
                if env.is_player_active(game_state):
                    training_active = True
                    idle_timer = time.time()
                    print(" Player active — resuming!")
                    break
                time.sleep(1.0)
                if shutdown_flag:
                    break

        # ── Reset episode ───────────────────────────────────────────
        frame = env.reset()
        reward_fn.reset()
        opp_tracker.reset()
        prev_state = trainer.model.rssm.initial_state(1, device)
        prev_state = {k: v.float() for k, v in prev_state.items()}

        # Current step delay from curriculum
        step_delay = curriculum.step_delay

        episode_reward     = 0.0
        episode_steps      = 0
        in_cover_count     = 0
        shot_count         = 0
        curiosity_total    = 0.0
        prev_game_state       = None
        took_damage_interrupt = False

        for step in range(MAX_STEPS_PER_EP):
            if shutdown_flag:
                break

            t_step_start = time.perf_counter()

            # ── Game state ────────────────────────────────────────
            game_state = env.get_game_state()

            # ── Idle check ────────────────────────────────────────
            if env.is_player_active(game_state):
                idle_timer = time.time()
            elif time.time() - idle_timer > IDLE_TIMEOUT:
                print(f"\nPlayer idle for {IDLE_TIMEOUT}s — pausing episode")
                training_active = False
                break

            # ── Opponent model: update player history ─────────────
            if game_state.get('got_shot', False):
                opp_tracker.mark_shot()
            opp_hist = opp_tracker.update(game_state)      # (4,) float32

            # ── Stage 1: Capture frame + telemetry (Perception Latency) ───
            t_percept_start = time.perf_counter()
            frame   = env.get_frame()
            vec_obs = env.get_observation(
                game_state, frame_np=frame,
                opponent_history=opp_hist
            )                                               # (15,) float32
            perception_latency_ms = (time.perf_counter() - t_percept_start) * 1000.0

            threat_score  = float(vec_obs[10])             # index 10 = threat
            ai_perception = vec_obs[6:10]                  # (4,)
            actors_seen   = float(ai_perception[0]) * 5.0  # raw count

            # Immediate Damage Reaction: True if engine reported damage or step was interrupted by bullet
            got_shot_now  = game_state.get('got_shot', False) or took_damage_interrupt
            took_damage_interrupt = False   # consumed for this step

            # ── Perception gate ───────────────────────────────────
            px, py = float(vec_obs[0]) * env._pos_scale, float(vec_obs[1]) * env._pos_scale
            nx, ny = float(vec_obs[3]) * env._pos_scale, float(vec_obs[4]) * env._pos_scale
            player_dist = float(np.sqrt((px - nx)**2 + (py - ny)**2))

            can_perceive = (
                ((actors_seen > 0.0 or player_dist < 1200.0) and threat_score > 0.5)
                or got_shot_now
            )

            # ── Build tensors ─────────────────────────────────────
            img_t = torch.tensor(
                frame.astype(np.float32) / 255.0,
                dtype=torch.float32, device=device
            ).permute(2, 0, 1).unsqueeze(0)                # (1,3,128,128)

            vec_t = torch.tensor(
                vec_obs, dtype=torch.float32, device=device
            ).unsqueeze(0)                                  # (1,15)

            # ── RND curiosity reward ──────────────────────────────
            curiosity = rnd.curiosity_reward(vec_t)
            curiosity_total += curiosity

            # ── Stage 2: Action selection (NN Inference Latency) ──
            t_infer_start = time.perf_counter()
            if can_perceive:
                with torch.no_grad():
                    obs_embed             = trainer.model.encode_obs(img_t, vec_t)
                    action, new_state, _  = trainer.model.get_action(obs_embed, prev_state)
                action_int = action.item()
                # Emergency reflex: If hit while staying still, immediately sprint to break line of fire
                if got_shot_now and action_int == 4:
                    action_int = 5  # Sprint
            else:
                action_int = 4          # Stay — NPC holds still
                new_state  = prev_state
            inference_latency_ms = (time.perf_counter() - t_infer_start) * 1000.0

            # ── Stage 3: Execute in UE5 (Env Step Latency) ────────
            t_env_start = time.perf_counter()
            env.move_npc(action_int)
            was_interrupted = env.wait_for_step_or_damage(step_delay)
            if was_interrupted:
                took_damage_interrupt = True
            env_latency_ms = (time.perf_counter() - t_env_start) * 1000.0

            # ── Post-step state ───────────────────────────────────
            game_state = env.get_game_state()
            if was_interrupted:
                game_state['got_shot'] = True

            # ── Reward (extrinsic + curiosity intrinsic) ──────────
            reward, info = reward_fn.compute(
                game_state, action_int, prev_game_state,
                ai_perception=ai_perception,
                threat_score=threat_score,
                player_dist=player_dist
            )
            reward += RND_WEIGHT * curiosity   # intrinsic bonus

            done = game_state['is_dead'] or step == MAX_STEPS_PER_EP - 1

            if info.get('in_cover', False):
                in_cover_count += 1
            if game_state.get('got_shot', False):
                shot_count += 1

            # ── Store transition ──────────────────────────────────
            buffer.add(frame, vec_obs, action_int, reward, done)

            episode_reward += reward
            episode_steps  += 1
            global_step    += 1
            prev_state      = {k: v.detach() for k, v in new_state.items()}
            prev_game_state = game_state

            # ── Live step log ─────────────────────────────────────
            act_name = ACTION_NAMES.get(action_int, str(action_int))
            if was_interrupted or info.get('got_shot'):
                tag = "HIT!"
            elif not can_perceive:
                tag = "BLIND_HOLD"
            elif info.get('holding_cover'):
                tag = "HOLD_COVER"
            elif info.get('in_cover'):
                tag = "IN_COVER"
            else:
                tag = "PATROL"

            state_str = (f"ARMED(seen={actors_seen:.0f})"
                         if threat_score > 0.5
                         else f"SAFE(seen={actors_seen:.0f})")
            print(
                f"\r  [Ep {episode} | Step {step:03d}] "
                f"Act:{act_name:<6} Rew:{reward:+5.2f} ({tag:<10}) "
                f"State:{state_str:<18} EpRew:{episode_reward:+6.2f} "
                f"Cur:{curiosity:.3f}  ",
                end='', flush=True
            )

            # ── Stage 4: Train World Model + Actor-Critic ─────────
            train_latency_ms = 0.0
            trained_this_step = False
            last_losses = getattr(trainer, '_last_losses', None)

            if global_step % TRAIN_EVERY == 0 and buffer.is_ready(MIN_BUFFER_EPISODES):
                batch = buffer.sample(config.batch_size, config.seq_len, device=str(device))
                if batch:
                    t_train_start = time.perf_counter()
                    img_b, vec_b, act_b, rew_b, done_b = batch
                    wm = trainer.train_world_model(img_b, vec_b, act_b, rew_b, done_b)
                    ac = trainer.train_actor_critic(img_b, vec_b, act_b, rew_b)
                    last_losses = {**wm, **ac}
                    trainer._last_losses = last_losses
                    # Also update RND predictor on this batch
                    rnd.update(vec_b[:, 0, :])
                    train_latency_ms = (time.perf_counter() - t_train_start) * 1000.0
                    trained_this_step = True

            total_step_latency_ms = (time.perf_counter() - t_step_start) * 1000.0

            # ── Log step telemetry to WandB ───────────────────────
            step_latencies = {
                "perception_ms": round(perception_latency_ms, 2),
                "inference_ms": round(inference_latency_ms, 2),
                "env_step_ms": round(env_latency_ms, 2),
                "total_step_latency_ms": round(total_step_latency_ms, 2),
            }
            if trained_this_step:
                step_latencies["train_step_ms"] = round(train_latency_ms, 2)

            # Log losses and latency breakdown regularly or on training events
            should_log_step = (trained_this_step or (global_step % 10 == 0))
            if should_log_step:
                logger.log_step(
                    step=global_step,
                    latency_ms=step_latencies,
                    losses=last_losses if trained_this_step else None,
                    log_system=(global_step % 20 == 0)
                )

            torch.cuda.empty_cache()

            if done:
                if game_state['is_dead']:
                    print(f"\n  [DEAD] NPC died at step {step} ")
                break

        # ── Episode end ─────────────────────────────────────────────
        survived     = not game_state['is_dead']
        in_cover_pct = in_cover_count / max(episode_steps, 1)
        avg_curiosity = curiosity_total / max(episode_steps, 1)

        # Curriculum: record win/loss, update difficulty
        curriculum.record(survived)
        step_delay = curriculum.update()

        metrics = {
            'episode_reward':  episode_reward,
            'episode_length':  episode_steps,
            'in_cover_pct':    in_cover_pct,
            'survival_time':   episode_steps * step_delay,
            'shots_taken':     shot_count,
            'buffer_size':     len(buffer),
            'win_rate':        curriculum.win_rate,
            'success_rate':    1.0 if survived else 0.0,
            'step_delay':      step_delay,
            'avg_curiosity':   avg_curiosity,
        }
        logger.log_episode(episode=episode, metrics=metrics)

        losses   = getattr(trainer, '_last_losses', {})
        wm_loss  = losses.get('world_model_loss', 0.0)
        act_loss = losses.get('actor_loss', 0.0)
        crt_loss = losses.get('critic_loss', 0.0)

        outcome = "SURVIVED (WIN)" if survived else "ELIMINATED (LOSS)"

        print(f" EPISODE {episode:<4d} SUMMARY  [{outcome}]")
        print(f"  Performance : Reward: {episode_reward:+7.2f} | Steps: {episode_steps:<3d} | Shots Taken: {shot_count}")
        print(f"  Tactics     : In-Cover: {in_cover_pct:6.1%} | Survival Time: {episode_steps * step_delay:5.1f}s")
        print(f"  Learning    : Win Rate: {curriculum.win_rate:6.1%} | Step Delay: {step_delay:4.2f}s | Curiosity: {avg_curiosity:.4f}")
        if losses:
            print(f"  Model Loss  : WM: {wm_loss:6.4f} | Actor: {act_loss:6.4f} | Critic: {crt_loss:6.4f}")
        else:
            print(f"  Model Loss  : [Replay Buffer Warming Up {len(buffer)}/{config.batch_size * config.seq_len * MIN_BUFFER_EPISODES} steps]")
        print(f"  Buffer Size : {len(buffer)} transitions | VRAM: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")

        # Save after every episode
        trainer.save_checkpoint(episode + 1, metrics={'episode_reward': episode_reward})

    # ── Final save ───────────────────────────────────────────────────
    print("\n Saving final checkpoint")
    trainer.save_checkpoint(episode + 1)
    logger.close()
    env.disconnect()
    print("Training complete")


if __name__ == "__main__":
    train()