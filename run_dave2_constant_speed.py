# run_dave2_constant_speed.py
import argparse
import cv2
import json
import random
import subprocess
import shlex
import sys
import time
from queue import Queue, Empty
from pathlib import Path
from typing import Tuple

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
    return world.get_blueprint_library().find("vehicle.tesla.model3")


def attach_rgb_camera(world: carla.World, ego: carla.Vehicle, width=320, height=180, fov=90, sensor_tick=None):
    cam_bp = world.get_blueprint_library().find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(width))
    cam_bp.set_attribute("image_size_y", str(height))
    cam_bp.set_attribute("fov", str(fov))
    if sensor_tick is not None:
        cam_bp.set_attribute("sensor_tick", str(sensor_tick))  # one image per sim tick
    rel_tf = carla.Transform(carla.Location(x=1.6, z=1.7))
    cam = world.spawn_actor(cam_bp, rel_tf, attach_to=ego)
    return cam


def get_image_for_frame(q: Queue, frame_id: int, timeout: float = 2.0) -> carla.Image:
    """
    Block until we receive the camera image matching the given simulator frame id.
    Guarantees one frame per tick (no drops/dups in the encoded video).
    """
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            img = q.get(timeout=max(0.0, deadline - time.time()))
            last = img
            if img.frame == frame_id:
                return img
            # If we somehow jumped ahead (shouldn't happen), return the latest we saw.
            if img.frame > frame_id:
                return img
        except Empty:
            pass
    raise RuntimeError(f"Timed out waiting for sensor frame {frame_id} (last={None if last is None else last.frame})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--town", default="Town04", help="Town01..Town10 (or any installed map)")
    parser.add_argument("--ckpt", required=True, help="Path to DAVE2 checkpoint (state dict with ['model'])")
    parser.add_argument("--target-kmh", type=float, default=20.0)
    parser.add_argument("--fps", type=float, default=10.0, help="Simulation fixed step (Hz)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seconds", type=int, default=300, help="How long to drive before quitting")
    parser.add_argument("--out", default=None, help="Dir to save logs (JSONL) and frames")
    parser.add_argument("--save-frames", action="store_true", help="Write camera frames to disk")
    parser.add_argument("--record-carla", default=None, help="Use CARLA's built-in recorder to a .log")
    parser.add_argument("--video", default=None, help="Path to output video (e.g., runs/demo.mp4 or runs/demo.avi)")
    parser.add_argument("--ffmpeg", default=None,
                        help="Write video via ffmpeg pipe to this path (e.g., runs/demo.mp4)")
    parser.add_argument("--ffmpeg-codec", default="libx264",
                        help="ffmpeg codec: libx264 (best), mpeg4 (fallback), libx265, hevc_nvenc, etc.")
    parser.add_argument("--ffmpeg-crf", type=int, default=23,
                        help="ffmpeg CRF (lower=better quality, 17-28 typical) for libx264/265")
    parser.add_argument("--ffmpeg-preset", default="veryfast",
                        help="x264/x265 preset (ultrafast..placebo). Speed/quality tradeoff.")
    args = parser.parse_args()

    random.seed(args.seed)
    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)

    world = client.get_world()
    if args.town and (world.get_map().name.split("/")[-1] != args.town):
        world = client.load_world(args.town)

    log_f = None
    out = None
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
    world.apply_settings(settings)
    sim_fps = 1.0 / settings.fixed_delta_seconds  # authoritative FPS for encoding

    traffic_manager = client.get_trafficmanager()
    traffic_manager.set_synchronous_mode(True)

    ego = None
    camera = None
    spectator = world.get_spectator()
    image_queue: Queue = Queue()

    # --- video writer state ---
    video_writer = None
    video_w = video_h = None
    video_path_actual = None

    ffmpeg_proc = None
    ffmpeg_path_actual = None
    # ---------------------------

    frames_processed = 0
    video_frames_written = 0
    ffmpeg_frames_sent = 0

    actors_to_destroy = []

    try:
        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("No spawn points available on this map.")
        spawn_tf = random.choice(spawn_points)

        vehicle_bp = pick_vehicle_bp(world)
        ego = world.try_spawn_actor(vehicle_bp, spawn_tf)
        if ego is None:
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

        camera = attach_rgb_camera(world, ego, width=320, height=180, fov=90,
                                   sensor_tick=settings.fixed_delta_seconds)
        actors_to_destroy.append(camera)
        camera.listen(image_queue.put)

        # Build steering agent + speed controller
        steer_agent = DAVE2Agent(ego, ckpt_path=args.ckpt, input_shape=(180, 320))
        speed_ctrl = ConstantSpeedController(target_kmh=args.target_kmh)

        # Persist meta (dt & sim_fps) for reproducibility
        if out is not None:
            (out / "meta.txt").write_text(
                f"fixed_delta_seconds={settings.fixed_delta_seconds}\n"
                f"sim_fps={sim_fps}\n"
            )

        # Let physics settle a frame and record baseline sim time
        world.tick()
        start_elapsed = world.get_snapshot().timestamp.elapsed_seconds

        steps = 0
        print(f"[INFO] Driving in {args.town} at target {args.target_kmh:.1f} km/h ... Ctrl+C to stop.")

        while True:
            # Advance exactly one sim tick and fetch the matching camera frame
            frame_id = world.tick()
            snapshot = world.get_snapshot()
            img = get_image_for_frame(image_queue, frame_id)
            pil = carla_img_to_pil(img)
            sim_time = max(0.0, snapshot.timestamp.elapsed_seconds - start_elapsed)
            frames_processed += 1

            # Longitudinal control (hold speed)
            kmh = speed_kmh(ego)
            throttle, brake = speed_ctrl.step(kmh, dt=settings.fixed_delta_seconds)

            # Lateral control (DAVE2 predicts steer from image)
            ctrl = steer_agent.run_step(pil, throttle=throttle, brake=brake)

            # Convert PIL(RGB) -> BGR numpy for video/overlay
            bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            h, w = bgr.shape[:2]

            # Optional overlay for sanity
            # t_sim = steps * settings.fixed_delta_seconds
            # cv2.putText(bgr, f"t={t_sim:6.2f}s v={kmh:4.1f}km/h", (6, 16),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            # --- OpenCV writer (CFR = sim_fps) ---
            if args.video and video_writer is None:
                out_path = Path(args.video)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                fps = float(sim_fps)
                video_w, video_h = w, h

                def try_open(path: Path, fourcc_str: str):
                    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
                    vw = cv2.VideoWriter(str(path), fourcc, fps, (w, h), True)
                    return vw if vw.isOpened() else None

                ext = out_path.suffix.lower()
                video_writer = None
                video_path_actual = str(out_path)

                if ext == ".mp4":
                    # Try H.264, then mp4v
                    video_writer = try_open(out_path, "avc1") or try_open(out_path, "mp4v")
                    if video_writer is None:
                        # Fallback to AVI MJPG
                        fallback = out_path.with_suffix(".avi")
                        print(f"[WARN] Couldn't open MP4 writer (avc1/mp4v). Falling back to AVI(MJPG): {fallback}")
                        video_writer = try_open(fallback, "MJPG")
                        if video_writer is not None:
                            video_path_actual = str(fallback)
                else:
                    video_writer = try_open(out_path, "MJPG")

                if video_writer is None:
                    raise RuntimeError("Failed to open any video writer (avc1/mp4v/MJPG).")
                else:
                    print(f"[INFO] Video writer opened: {video_path_actual} ({w}x{h}@{fps:.1f})")

            if video_writer is not None:
                if (w != video_w or h != video_h):
                    raise RuntimeError(
                        f"Frame size changed from {(video_w, video_h)} to {(w, h)}; all frames must be identical."
                    )
                video_writer.write(bgr)
                video_frames_written += 1

            # --- ffmpeg pipe (CFR = sim_fps) ---
            if args.ffmpeg and ffmpeg_proc is None:
                out_path = Path(args.ffmpeg)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                ffmpeg_path_actual = str(out_path)

                cmd = [
                    "ffmpeg", "-y",
                    "-f", "rawvideo",
                    "-vcodec", "rawvideo",
                    "-pix_fmt", "bgr24",
                    "-s", f"{w}x{h}",
                    "-r", f"{sim_fps}",     # match simulation FPS
                    "-i", "-",              # stdin
                    "-an",
                    "-vsync", "cfr",        # constant frame rate
                ]

                if args.ffmpeg_codec in ("libx264", "libx265"):
                    cmd += [
                        "-c:v", args.ffmpeg_codec,
                        "-pix_fmt", "yuv420p",
                        "-preset", args.ffmpeg_preset,
                        "-crf", str(args.ffmpeg_crf),
                        "-movflags", "+faststart",
                        ffmpeg_path_actual,
                    ]
                else:
                    cmd += ["-c:v", args.ffmpeg_codec, ffmpeg_path_actual]

                print("[INFO] Starting ffmpeg:", " ".join(shlex.quote(c) for c in cmd))
                try:
                    ffmpeg_proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
                except FileNotFoundError:
                    raise RuntimeError("ffmpeg not found. On HPC, try: module load ffmpeg")

            if ffmpeg_proc is not None and ffmpeg_proc.stdin:
                ffmpeg_proc.stdin.write(bgr.tobytes())
                ffmpeg_frames_sent += 1

            # Optional per-frame PNG save
            if args.save_frames and out is not None:
                img_path = out / "images" / f"{args.town}__{img.frame:06d}.png"
                pil.save(img_path)

            # Log JSONL
            if log_f is not None:
                tf = ego.get_transform()
                loc, rot = tf.location, tf.rotation
                rec = {
                    "frame": int(img.frame),
                    "speed_kmh": float(kmh),
                    "control": {"throttle": float(throttle), "steer": float(ctrl.steer), "brake": float(brake)},
                    "image_path": f"images/{args.town}__{img.frame:06d}.png" if args.save_frames else None,
                    "town": args.town,
                    "location": {"x": loc.x, "y": loc.y, "z": loc.z},
                    "rotation": {"pitch": rot.pitch, "yaw": rot.yaw, "roll": rot.roll},
                }
                json.dump(rec, log_f); log_f.write("\n")

            ego.apply_control(ctrl)

            # Keep spectator loosely tracking
            if steps % int(sim_fps // 2 or 1) == 0:
                ego_tf = ego.get_transform()
                spectator.set_transform(
                    carla.Transform(
                        ego_tf.location + carla.Location(z=30, x=-15),
                        carla.Rotation(pitch=-45, yaw=ego_tf.rotation.yaw),
                    )
                )

            if sim_time >= args.seconds:
                print("[INFO] Time limit reached, stopping.")
                if args.out:
                    (Path(args.out) / "done").touch()
                break

            if steps % int(sim_fps) == 0:
                print(f"[t={steps*settings.fixed_delta_seconds:5.1f}s] speed={kmh:5.1f} km/h  "
                      f"throttle={throttle:4.2f}  brake={brake:4.2f}")
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

        if ffmpeg_proc is not None:
            try:
                ffmpeg_proc.stdin.close()
            except Exception:
                pass
            ret = ffmpeg_proc.wait()
            print(f"[INFO] ffmpeg exited with code {ret}. Wrote: {ffmpeg_path_actual}")
        if video_writer is not None:
            video_writer.release()
            print(f"[INFO] Video saved to: {video_path_actual}")
        if args.record_carla:
            client.stop_recorder()
        if log_f is not None:
            log_f.close()

        print("[INFO] Destroyed actors and restored world settings.")


if __name__ == "__main__":
    main()
