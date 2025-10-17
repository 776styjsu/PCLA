# run_dave2_constant_speed.py
import argparse
import json
import random
import sys
import time
from collections import deque
from pathlib import Path
from typing import Deque, Tuple

import carla
import numpy as np
from agents.dave2.dave2_agent import DAVE2Agent
from PIL import Image


def carla_img_to_pil(image: carla.Image) -> Image.Image:
    """Convert CARLA BGRA uint8 image to RGB PIL."""
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    arr = arr.reshape((image.height, image.width, 4))  # BGRA
    rgb = arr[:, :, :3][:, :, ::-1]  # BGR->RGB
    return Image.fromarray(rgb)


def speed_kmh(vehicle: carla.Vehicle) -> float:
    v = vehicle.get_velocity()
    return 3.6 * np.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)


class ConstantSpeedController:
    """
    Very simple PI(D) longitudinal controller to track a target speed (km/h).
    We cap throttle at 0.75 to reduce oscillations; brake is linear in overspeed.
    """

    def __init__(
        self,
        target_kmh=25.0,
        kp=0.08,
        ki=0.02,
        kd=0.00,
        brake_k=0.15,
        throttle_cap=0.75,
    ):
        self.target = float(target_kmh)
        self.kp, self.ki, self.kd = kp, ki, kd
        self.brake_k = brake_k
        self.throttle_cap = throttle_cap
        self.integral = 0.0
        self.prev_err = None

    def step(self, current_kmh: float, dt: float) -> Tuple[float, float]:
        err = self.target - float(current_kmh)
        self.integral += err * dt
        deriv = 0.0 if self.prev_err is None else (err - self.prev_err) / max(dt, 1e-6)
        self.prev_err = err

        u = self.kp * err + self.ki * self.integral + self.kd * deriv
        if err >= 0.0:
            throttle = float(np.clip(u, 0.0, self.throttle_cap))
            brake = 0.0
        else:
            throttle = 0.0
            brake = float(np.clip(-err * self.brake_k, 0.0, 1.0))
        return throttle, brake


def pick_vehicle_bp(world: carla.World) -> carla.ActorBlueprint:
    bps = world.get_blueprint_library().filter("vehicle.*model3*")
    if not bps:
        bps = world.get_blueprint_library().filter("vehicle.*")
    bp = random.choice(bps)
    if bp.has_attribute("color"):
        bp.set_attribute(
            "color", random.choice(bp.get_attribute("color").recommended_values)
        )
    if bp.has_attribute("role_name"):
        bp.set_attribute("role_name", "ego")
    return bp


def attach_rgb_camera(
    world: carla.World, ego: carla.Vehicle, width=200, height=150, fov=90
):
    cam_bp = world.get_blueprint_library().find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(width))
    cam_bp.set_attribute("image_size_y", str(height))
    cam_bp.set_attribute("fov", str(fov))
    # Mount slightly ahead and above the roof for a driver-ish view
    rel_tf = carla.Transform(carla.Location(x=1.6, z=1.7))
    cam = world.spawn_actor(cam_bp, rel_tf, attach_to=ego)
    return cam


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument(
        "--town", default="Town04", help="Town01..Town10 (or any installed map)"
    )
    parser.add_argument(
        "--ckpt",
        required=True,
        help="Path to DAVE2 checkpoint (state dict with ['model'])",
    )
    parser.add_argument("--target-kmh", type=float, default=20.0)
    parser.add_argument(
        "--fps", type=float, default=30.0, help="Simulation fixed step (Hz)"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--seconds", type=int, default=300, help="How long to drive before quitting"
    )
    parser.add_argument(
        "--out", default=None, help="Dir to save logs (JSONL) and frames"
    )
    parser.add_argument(
        "--save-frames", action="store_true", help="Write camera frames to disk"
    )
    parser.add_argument(
        "--record-carla", default=None, help="Use CARLA's built-in recorder to a .log"
    )
    args = parser.parse_args()

    random.seed(args.seed)
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)

    world = client.get_world()
    if args.town and (world.get_map().name.split("/")[-1] != args.town):
        world = client.load_world(args.town)

    log_f = None
    if args.out:
        out = Path(args.out)
        (out / "images").mkdir(parents=True, exist_ok=True)
        log_f = open(out / "measurements.jsonl", "w")

    if args.record_carla:
        client.start_recorder(args.record_carla)

    # Synchronous mode for stable sensor/actuation timing
    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / args.fps
    settings.max_substeps = 1
    world.apply_settings(settings)

    traffic_manager = client.get_trafficmanager()
    traffic_manager.set_synchronous_mode(True)

    ego = None
    camera = None
    spectator = world.get_spectator()
    image_queue: Deque[carla.Image] = deque(maxlen=1)

    actors_to_destroy = []

    try:
        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("No spawn points available on this map.")
        spawn_tf = random.choice(spawn_points)

        vehicle_bp = pick_vehicle_bp(world)
        ego = world.try_spawn_actor(vehicle_bp, spawn_tf)
        if ego is None:
            # Fallback: try others
            for tf in spawn_points:
                ego = world.try_spawn_actor(vehicle_bp, tf)
                if ego:
                    break
        if ego is None:
            raise RuntimeError("Failed to spawn ego vehicle.")

        actors_to_destroy.append(ego)
        ego.set_autopilot(False)

        # Follow the ego with spectator
        spectator.set_transform(
            carla.Transform(
                spawn_tf.location + carla.Location(z=30, x=-15),
                carla.Rotation(pitch=-45, yaw=spawn_tf.rotation.yaw),
            )
        )

        camera = attach_rgb_camera(world, ego, width=320, height=180, fov=90)
        actors_to_destroy.append(camera)
        camera.listen(lambda img: image_queue.append(img))

        # Build your steering agent + speed controller
        steer_agent = DAVE2Agent(ego, ckpt_path=args.ckpt, input_shape=(180, 320))
        speed_ctrl = ConstantSpeedController(target_kmh=args.target_kmh)

        # Let physics settle a frame
        world.tick()

        start_time = time.time()
        steps = 0
        print(
            f"[INFO] Driving in {args.town} at target {args.target_kmh:.1f} km/h ... Ctrl+C to stop."
        )

        while True:
            # Single synchronous step
            snapshot = world.tick()

            # Latest image (if any)
            if not image_queue:
                continue
            img = image_queue[-1]
            pil = carla_img_to_pil(img)

            # Longitudinal control to hold speed
            kmh = speed_kmh(ego)
            throttle, brake = speed_ctrl.step(kmh, dt=settings.fixed_delta_seconds)

            # Lateral control (DAVE2 predicts steer from image)
            ctrl = steer_agent.run_step(pil, throttle=throttle, brake=brake)

            if args.save_frames and args.out:
                img_path = out / "images" / f"{args.town}__{img.frame:06d}.png"
                pil.save(img_path)

            # log one JSON line per frame (speed, control, pose, optional image path)
            if log_f is not None:
                tf = ego.get_transform()
                loc, rot = tf.location, tf.rotation
                rec = {
                    "frame": int(img.frame),
                    "speed_kmh": float(kmh),
                    "control": {
                        "throttle": float(throttle),
                        "steer": float(ctrl.steer),
                        "brake": float(brake),
                    },
                    "image_path": f"images/{args.town}__{img.frame:06d}.png"
                    if args.save_frames
                    else None,
                    "town": args.town,
                    "location": {"x": loc.x, "y": loc.y, "z": loc.z},
                    "rotation": {"pitch": rot.pitch, "yaw": rot.yaw, "roll": rot.roll},
                }
                json.dump(rec, log_f)
                log_f.write("\n")

            ego.apply_control(ctrl)

            # Keep spectator loosely tracking
            if steps % int(args.fps // 2 or 1) == 0:
                ego_tf = ego.get_transform()
                spectator.set_transform(
                    carla.Transform(
                        ego_tf.location + carla.Location(z=30, x=-15),
                        carla.Rotation(pitch=-45, yaw=ego_tf.rotation.yaw),
                    )
                )

            if (time.time() - start_time) > args.seconds:
                print("[INFO] Time limit reached, stopping.")
                break

            if steps % int(args.fps) == 0:
                print(
                    f"[t={steps/settings.fixed_delta_seconds:5.1f}s] speed={kmh:5.1f} km/h  throttle={throttle:4.2f}  brake={brake:4.2f}"
                )
            steps += 1

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted, shutting down.")
    finally:
        # Clean up
        if camera is not None:
            camera.stop()
        for a in actors_to_destroy[::-1]:
            try:
                a.destroy()
            except Exception:
                pass
        world.apply_settings(original_settings)
        traffic_manager.set_synchronous_mode(False)

        if args.record_carla:
            client.stop_recorder()
        if log_f is not None:
            log_f.close()

        print("[INFO] Destroyed actors and restored world settings.")


if __name__ == "__main__":
    main()
