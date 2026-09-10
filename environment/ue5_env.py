# environment/ue5_env.py
import unrealcv
import numpy as np
import time
import os
import json
import io
from PIL import Image


class UE5Env:
    def __init__(self, host='localhost', port=9000, target_actor='BP_WAM_Gaurd_C_1', player_actor='BP_ThirdPerson_Pawn_C_0'):
        self.client = unrealcv.Client((host, port))
        self.client.connect()

        if not self.client.isconnected():
            raise ConnectionError(" UnrealCV not connected.")

        print(" UE5Env connected.")
        self.npc_cam_id = 'FusionCameraActor_0'
        self.npc_name        = target_actor
        self.player_name     = player_actor
        self.weapon_name     = 'BP_Weap_AssaultRifle1_C_1'
        self.temp_path       = 'E:/temp_frame.png'
        self.step_size       = 800.0   

        
        self.spawn_x = -10.0
        self.spawn_y = 280.0
        self.spawn_z = 200.0

        # Actions: 0=forward, 1=back, 2=left (strafe), 3=right (strafe), 4=stay, 5=sprint forward
        self.actions = {
            0: (self.step_size,       0.0),             # Forward
            1: (-self.step_size * 0.7, 0.0),            # Backward
            2: (0.0,                  -self.step_size), # Left strafe
            3: (0.0,                  self.step_size),  # Right strafe
            4: (0.0,                  0.0),             # Stay / Crouch
            5: (self.step_size * 1.8, 0.0)              # Sprint Forward (~630 units)
        }

        self.prev_health = 100.0

        
        self._pos_scale = 4000.0

    # ─── Camera ──────────────────────────────────────────────────────
    def sync_spectator_camera(self):
        """Copy NPC eye-socket transform to Camera 0 so vget captures NPC POV."""
        res = self.client.request(f'vbp {self.npc_name} GetCamInfo')
        try:
            data = json.loads(res)

            # UE5 outputs keys: "Return Value Location" or "Location"
            loc_val = data.get('Return Value Location', data.get('Location', ''))
            if isinstance(loc_val, dict):
                lx = float(loc_val.get('X', self.spawn_x))
                ly = float(loc_val.get('Y', self.spawn_y))
                lz = float(loc_val.get('Z', self.spawn_z + 60))
            elif isinstance(loc_val, str) and loc_val.strip():
                # Format: "X=-291.908 Y=280.000 Z=325.256"
                parts = {}
                for item in loc_val.replace('(', '').replace(')', '').split():
                    if '=' in item:
                        k, v = item.split('=', 1)
                        parts[k.strip()] = float(v)
                lx = parts.get('X', self.spawn_x)
                ly = parts.get('Y', self.spawn_y)
                lz = parts.get('Z', self.spawn_z + 60)
            elif isinstance(loc_val, (list, tuple)):
                lx, ly, lz = float(loc_val[0]), float(loc_val[1]), float(loc_val[2])
            else:
                # Fallback to NPC current location + eye offset
                cur_x, cur_y, cur_z = self.get_npc_location()
                lx, ly, lz = cur_x, cur_y, cur_z + 60

            # UE5 outputs keys: "Return Value Rotation" or "Rotation"
            # Format: "P=-20.000000 Y=0.000000 R=-0.000000" or Pitch/Yaw/Roll
            rot_val = data.get('Return Value Rotation', data.get('Rotation', ''))
            pitch, yaw, roll = 0.0, 0.0, 0.0
            if isinstance(rot_val, dict):
                pitch = float(rot_val.get('Pitch', rot_val.get('P', 0.0)))
                yaw   = float(rot_val.get('Yaw',   rot_val.get('Y', 0.0)))
                roll  = float(rot_val.get('Roll',  rot_val.get('R', 0.0)))
            elif isinstance(rot_val, str) and rot_val.strip():
                parts = {}
                for item in rot_val.replace('(', '').replace(')', '').split():
                    if '=' in item:
                        k, v = item.split('=', 1)
                        parts[k.strip()] = float(v)
                pitch = parts.get('Pitch', parts.get('P', 0.0))
                yaw   = parts.get('Yaw',   parts.get('Y', 0.0))
                roll  = parts.get('Roll',  parts.get('R', 0.0))
            elif isinstance(rot_val, (list, tuple)):
                pitch, yaw, roll = float(rot_val[0]), float(rot_val[1]), float(rot_val[2])

            self.client.request(f'vset {self.npc_cam_id}/location {lx:.3f} {ly:.3f} {lz:.3f}')
            self.client.request(f'vset {self.npc_cam_id}/rotation {pitch:.3f} {yaw:.3f} {roll:.3f}')
        except Exception as e:
            print(f"[Warning] sync_spectator_camera: {e}")

    # ─── Frame Capture ───────────────────────────────────────────────
    def get_frame(self):
        """
        Capture a 128x128 RGB frame directly from NPC's SceneCaptureComponent2D -> TextureRenderTarget2D.
        Returns:
            np.ndarray: shape (128, 128, 3), dtype uint8
        """
        res = self.client.request(f'vget /npc/{self.npc_name}/render_target png')
        if isinstance(res, (bytes, bytearray)):
            try:
                img = Image.open(io.BytesIO(res)).convert('RGB')
                if img.size != (128, 128):
                    img = img.resize((128, 128))
                return np.array(img, dtype=np.uint8)
            except Exception as e:
                print(f"[Warning] Frame decode error: {e}")
        elif isinstance(res, str) and not res.startswith('error'):
            if os.path.exists(res):
                img = Image.open(res).convert('RGB').resize((128, 128))
                return np.array(img, dtype=np.uint8)
        else:
            if isinstance(res, str) and res.startswith('error'):
                print(f"[Warning] UnrealCV get_frame error: {res}")

        return np.zeros((128, 128, 3), dtype=np.uint8)

    # ─── Locations ───────────────────────────────────────────────────
    def get_npc_location(self):
        res = self.client.request(f'vget /object/{self.npc_name}/location')
        try:
            x, y, z = map(float, res.strip().split())
            return x, y, z
        except:
            return 0.0, 0.0, 0.0

    def get_player_location(self):
        res = self.client.request(f'vget /object/{self.player_name}/location')
        try:
            x, y, z = map(float, res.strip().split())
            return x, y, z
        except:
            return 0.0, 0.0, 0.0

    def get_weapon_location(self):
        res = self.client.request(f'vget /object/{self.weapon_name}/location')
        try:
            x, y, z = map(float, res.strip().split())
            return x, y, z
        except:
            return None

    def get_player_rotation(self):
        res = self.client.request(f'vget /object/{self.player_name}/rotation')
        try:
            pitch, yaw, roll = map(float, res.strip().split())
            return pitch, yaw, roll
        except:
            return 0.0, 0.0, 0.0

    # ─── Weapon / Threat ─────────────────────────────────────────────
    def get_threat_score(self, game_state=None):
        """
        Direct Ground-Truth Blueprint Check via WeaponCheck().
        Returns:
            1.0 if player has weapon equipped in hand (e.g. 'BP_Weap_AssaultRifle1_C_1')
            0.0 if holstered / unarmed (returns 'None' or empty)
        """
        try:
            res = self.client.request(f'vbp {self.player_name} WeaponCheck')
            if res and not res.startswith('error'):
                data = json.loads(res)
                weap_name = str(data.get('WeaponName', '')).strip()
                if weap_name and weap_name.lower() != 'none':
                    return 1.0
                return 0.0
        except Exception:
            pass

        # Fallback to coordinate dot-product if blueprint call fails
        w_loc = self.get_weapon_location()
        if w_loc is None or game_state is None:
            return 0.0

        px, py = game_state['player_x'], game_state['player_y']
        wx, wy, _ = w_loc
        _, p_yaw, _ = self.get_player_rotation()
        yaw_rad = np.radians(p_yaw)
        fwd_x, fwd_y = np.cos(yaw_rad), np.sin(yaw_rad)
        to_wx, to_wy = wx - px, wy - py
        dot = to_wx * fwd_x + to_wy * fwd_y
        return 1.0 if dot > 0.0 else 0.0

    # ─── NPC Info ────────────────────────────────────────────────────
    def get_npc_info(self):
        res = self.client.request(f'vbp {self.npc_name} GetInfo')
        try:
            data       = json.loads(res)
            health     = float(data.get('Health', 100.0))
            is_dead    = str(data.get('IsDead', 'false')).lower() == 'true'
            max_health = float(data.get('MaxHealth', 100.0))
            return health, is_dead, max_health
        except:
            return 100.0, False, 100.0

    # ─── AI Perception ───────────────────────────────────────────────
    def get_perception_info(self, game_state=None):
        """Returns 4-dim normalised AI Perception vector."""
        res = self.client.request(f'vbp {self.npc_name} GetSenseInfo')
        actors_seen = 0.0
        nearest_dist = 1.0
        sight_angle = 0.5
        damage_received = 0.0
        try:
            data = json.loads(res)
            # Blueprint returns keys: 'ActorSeen' and 'NearestDistance'
            raw_seen = float(data.get('ActorSeen', data.get('actors_seen', 0)))
            actors_seen = raw_seen / 5.0
            raw_dist = float(data.get('NearestDistance', data.get('nearest_distance', 5000)))
            if raw_dist > 0:
                nearest_dist = raw_dist / 5000.0
            sight_angle = float(data.get('sight_angle', 90)) / 180.0
            damage_received = float(data.get('damage_received', 0)) / 100.0
        except:
            pass

        # Geometric sight fallback: If engine perception component hasn't triggered,
        # calculate sight cone directly from NPC transform to player transform
        if actors_seen <= 0.0 and game_state is not None:
            nx, ny = game_state.get('npc_x', 0.0), game_state.get('npc_y', 0.0)
            px, py = game_state.get('player_x', 0.0), game_state.get('player_y', 0.0)
            _, npc_yaw, _ = self.get_npc_rotation()
            dx, dy = px - nx, py - ny
            dist = float(np.sqrt(dx**2 + dy**2))
            angle_to_p = np.degrees(np.arctan2(dy, dx)) - npc_yaw
            angle_diff = (angle_to_p + 180.0) % 360.0 - 180.0
            
            if abs(angle_diff) <= 80.0 and dist <= 3000.0:
                actors_seen = 0.2  # 1.0 / 5.0 = 1 actor detected
                nearest_dist = min(dist / 5000.0, 1.0)
                sight_angle = abs(angle_diff) / 180.0

        return np.array([actors_seen, nearest_dist, sight_angle, damage_received],
                        dtype=np.float32)

    def get_npc_rotation(self):
        res = self.client.request(f'vget /object/{self.npc_name}/rotation')
        try:
            pitch, yaw, roll = map(float, res.strip().split())
            return pitch, yaw, roll
        except:
            return 0.0, 0.0, 0.0

    # ─── Movement ────────────────────────────────────────────────────
    def move_npc(self, action):
        """
        Orders the NPC to move with full character animation blendspaces.
        Uses native Unreal C++ AIController navigation command:
            vset /npc/<name>/moveto <X> <Y> <Z>
        Actions are evaluated RELATIVE to the NPC's current facing direction (yaw):
            0: Forward
            1: Backward
            2: Left (strafe)
            3: Right (strafe)
            4: Stay
            5: Sprint Forward
        """
        if action == 4:
            # Action 4 is stay/crouch in cover: stop moving immediately and hold ground
            cur_x, cur_y, cur_z = self.get_npc_location()
            self.client.request(f'vset /npc/{self.npc_name}/moveto {cur_x:.2f} {cur_y:.2f} {cur_z:.2f}')
            return

        x, y, z = self.get_npc_location()
        pitch, yaw, roll = self.get_npc_rotation()

        forward_dist, right_dist = self.actions[action]

        # Rotate forward/right deltas into World coordinates based on current yaw
        yaw_rad = np.radians(yaw)
        cos_y = np.cos(yaw_rad)
        sin_y = np.sin(yaw_rad)

        world_dx = forward_dist * cos_y - right_dist * sin_y
        world_dy = forward_dist * sin_y + right_dist * cos_y

        target_x = float(np.clip(x + world_dx, -510.0, 3190.0))
        target_y = float(np.clip(y + world_dy, -1000.0, 3400.0))
        target_z = float(z)

        # Send native AIController navigation command (triggers walking/running blendspaces)
        self.client.request(
            f'vset /npc/{self.npc_name}/moveto {target_x:.2f} {target_y:.2f} {target_z:.2f}'
        )

    def wait_for_step_or_damage(self, duration=0.3, check_interval=0.015):
        """
        0ms Damage Interrupt:
        Instead of a blind time.sleep(duration), actively polls health at micro-intervals.
        If a bullet hits the NPC, breaks IMMEDIATELY so the World Model can react without lag.
        Returns:
            interrupted (bool): True if damage woke the agent up early.
        """
        start = time.time()
        while time.time() - start < duration:
            health, is_dead, _ = self.get_npc_info()
            if health < self.prev_health or is_dead:
                return True
            time.sleep(check_interval)
        return False





    # ─── Game State ──────────────────────────────────────────────────
    def get_game_state(self):
        npc_x, npc_y, npc_z         = self.get_npc_location()
        player_x, player_y, player_z = self.get_player_location()
        health, is_dead, max_health  = self.get_npc_info()

        
        out_of_bounds = (
            npc_x < -600 or npc_x > 3300 or
            npc_y < -1100 or npc_y > 3500 or
            npc_z < -500
        )
        if out_of_bounds:
            self.client.request(
                f'vset /object/{self.npc_name}/location '
                f'{self.spawn_x} {self.spawn_y} {self.spawn_z}'
            )

        got_shot = health < self.prev_health
        self.prev_health = health

        return {
            'npc_x': npc_x, 'npc_y': npc_y, 'npc_z': npc_z,
            'player_x': player_x, 'player_y': player_y, 'player_z': player_z,
            'health': health, 'max_health': max_health,
            'is_dead': is_dead, 'alive': not is_dead, 'got_shot': got_shot
        }

    # ─── Observation Vector (15-dim scalar) ──────────────────────────
    def get_observation(self, game_state, frame_np=None, opponent_history=None):
        """
        Returns the 15-dim scalar telemetry vector:
            player_pos (3) + npc_pos (3) + ai_perception (4) + threat (1)
            + opponent_history (4) = 15
        Concatenated externally with the 512-dim CNN visual embedding → 527 total.
        """
        s = self._pos_scale
        player_pos = np.array([
            game_state['player_x'] / s,
            game_state['player_y'] / s,
            game_state['player_z'] / s
        ], dtype=np.float32)
        npc_pos = np.array([
            game_state['npc_x'] / s,
            game_state['npc_y'] / s,
            game_state['npc_z'] / s
        ], dtype=np.float32)
        ai_perception = self.get_perception_info(game_state=game_state)   # (4,)

        # ── Threat score from Engine Ground Truth ─────────
        threat_score = self.get_threat_score(game_state)            # float 0.0 or 1.0
        threat = np.array([threat_score], dtype=np.float32)         # (1,)

        if opponent_history is None:
            opponent_history = np.zeros(4, dtype=np.float32)

        return np.concatenate([player_pos, npc_pos, ai_perception, threat, opponent_history])  # (15,)

    # ─── Player activity check ───────────────────────────────────────
    def is_player_active(self, game_state):
        if not hasattr(self, '_last_player_pos'):
            self._last_player_pos = (game_state['player_x'], game_state['player_y'])
            return True
        dx = game_state['player_x'] - self._last_player_pos[0]
        dy = game_state['player_y'] - self._last_player_pos[1]
        self._last_player_pos = (game_state['player_x'], game_state['player_y'])
        return np.sqrt(dx**2 + dy**2) > 10.0

    # ─── Reset / Disconnect ──────────────────────────────────────────
    def reset(self):
        self.client.request(
            f'vset /object/{self.npc_name}/location '
            f'{self.spawn_x} {self.spawn_y} {self.spawn_z}'
        )
        self.prev_health = 100.0
        time.sleep(0.5)
        return self.get_frame()   # returns single 128x128 uint8 frame

    def disconnect(self):
        self.client.disconnect()
        print("UE5Env disconnected.")


# ─── VALIDATION ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    env = UE5Env()

    print("\n Testing get_frame()")
    frame = env.get_frame()
    print(f"Frame shape: {frame.shape}, dtype: {frame.dtype}")   # (128, 128, 3) uint8

    print("\n Testing get_game_state()")
    state = env.get_game_state()
    print(f"NPC pos:    ({state['npc_x']:.1f}, {state['npc_y']:.1f})")
    print(f"Player pos: ({state['player_x']:.1f}, {state['player_y']:.1f})")

    print("\n Testing get_observation()")
    vec_obs = env.get_observation(state)
    print(f"Vec obs shape: {vec_obs.shape}, values: {vec_obs}")   # (10,)

    print("\n Testing move_npc()")
    for action in range(5):
        env.move_npc(action)
        s2 = env.get_game_state()
        print(f"Action {action}: NPC at ({s2['npc_x']:.1f}, {s2['npc_y']:.1f})")
        time.sleep(0.2)

    print("\n SUCCESS")
    env.disconnect()