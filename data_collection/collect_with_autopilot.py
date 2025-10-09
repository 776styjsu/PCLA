import os, json, time, math, random, argparse, subprocess, signal
from pathlib import Path
from collections import deque
from datetime import datetime
import carla

PIDFILE_TMPL = "./carla_{port}.pid"
LOGFILE_TMPL = "./carla_{port}.log"

def stop_carla(port=None):
    """Stop a CARLA server we launched earlier (prefer pidfile; else restricted pkill)."""
    # 1) Try pidfile-based group kill (fast, no global scans)
    if port is not None:
        pidfile = PIDFILE_TMPL.format(port=port)
        if os.path.exists(pidfile):
            try:
                with open(pidfile) as f:
                    pid = int(f.read().strip())
                # Kill the whole process group started by Popen(start_new_session=True)
                os.killpg(pid, signal.SIGTERM)
                time.sleep(1.0)
                try:
                    os.killpg(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                print(f"Stopped CARLA pgid {pid} (port {port})")
            except Exception as e:
                print(f"Warning: failed to kill CARLA by pidfile: {e}")
            finally:
                try:
                    os.remove(pidfile)
                except OSError:
                    pass
            return

    # 2) Fallback: restricted pkill (your user only, exact names), time-bounded
    user = os.environ.get("USER", "")
    for name in ("CarlaUE4-Linux-Shipping", "CarlaUE4.sh", "CarlaUE4"):
        for args in (["pkill", "-u", user, "-x", name],
                     ["pkill", "-9", "-u", user, "-x", name]):
            try:
                subprocess.run(args, check=False, timeout=3)
            except subprocess.TimeoutExpired:
                print(f"pkill timed out for {name}; continuing")
    print("Stopped any existing CARLA servers (fallback)")

def _wait_for_carla(port, timeout=120):
    """Poll the RPC port until the server responds or timeout."""
    client = carla.Client("localhost", port)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            client.set_timeout(2.0)
            _ = client.get_server_version()
            return True
        except Exception:
            time.sleep(1.0)
    return False

def start_carla(args):
    carla_path = args.carla_path
    port       = args.port
    gpu        = args.gpu

    if not carla_path:
        raise ValueError("You must pass --carla-path=/path/to/CarlaUE4.sh when --start-carla or --restart-carla is set.")

    cmd = [
        carla_path,
        f"-carla-rpc-port={port}",
        "-RenderOffScreen",
        "-nosound",
    ]

    print(f"cmd: {cmd}")

    if not gpu:
        cmd.append("-opengl")

    # Log to /tmp so we can debug crashes
    log_path = LOGFILE_TMPL.format(port=port)
    logf = open(log_path, "ab", buffering=0)

    # Start in a new session so we can kill the whole group later
    proc = subprocess.Popen(
        cmd,
        stdout=logf,
        stderr=subprocess.STDOUT,
        start_new_session=True
    )

    # Save pidfile for precise shutdowns
    pidfile = PIDFILE_TMPL.format(port=port)
    try:
        with open(pidfile, "w") as f:
            f.write(str(proc.pid))
    except Exception:
        pass

    print(f"Started CARLA on port {port}, PID={proc.pid}. Logs: {log_path}", flush=True)

    # Wait until the server is actually ready
    if not _wait_for_carla(port, timeout=120):
        raise RuntimeError(f"CARLA failed to come up on port {port}. Check log: {log_path}")

    return proc

def restart_carla(args):
    print("Try to restart carla", flush=True)
    stop_carla(args.port)  # fast, pidfile-based; falls back to user-scoped pkill
    return start_carla(args)

def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

def normalize_map_name(m):  # "/Game/Carla/Maps/Town03" -> "Town03"
    return Path(m).name

def disable_red_lights(world):
    # Stop the signal timers so nothing changes while we edit.
    world.freeze_all_traffic_lights(True)

    # Turn every light Off (no red glow in renders) and neuter timers.
    for tl in world.get_actors().filter("traffic.traffic_light"):
        tl.set_state(carla.TrafficLightState.Green)   # or TrafficLightState.Green
        tl.set_red_time(0.0)
        tl.set_yellow_time(0.0)
        tl.set_green_time(1e6)

def spawn_ego(world, bp_lib):
    veh_bp = bp_lib.find("vehicle.tesla.model3")
    veh_bp.set_attribute("role_name", "hero")
    spawn = random.choice(world.get_map().get_spawn_points())
    return world.spawn_actor(veh_bp, spawn)

def attach_front_rgb(world, bp_lib, parent, out_dir, width, height, fov=90):
    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(width))
    cam_bp.set_attribute("image_size_y", str(height))
    cam_bp.set_attribute("fov", str(fov))
    cam_tf = carla.Transform(carla.Location(x=1.6, z=1.7))
    camera = world.spawn_actor(cam_bp, cam_tf, attach_to=parent)

    img_dir = os.path.join(out_dir, "images")
    ensure_dir(img_dir)
    q = deque(maxlen=30)

    state = {"recording": True}  # mutable flag visible to the callback

    def set_recording(flag: bool):
        state["recording"] = bool(flag)

    def _on_img(img):
        # only save/queue when recording
        img.save_to_disk(os.path.join(img_dir, f"{img.frame:06d}.png"))
        q.append(img.frame)

    camera.listen(_on_img)
    return camera, q, set_recording

def spawn_npcs(world, bp_lib, tm, n=40):
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)
    actors, i = [], 0
    for sp in spawn_points:
        if i >= n: break
        veh_bp = random.choice(bp_lib.filter("vehicle.*"))
        try:
            v = world.try_spawn_actor(veh_bp, sp)
            if v:
                v.set_autopilot(True, tm.get_port())
                actors.append(v)
                i += 1
        except RuntimeError:
            pass
    return actors

def collect_once_on_world(client, tm, args, out_dir):
    """Runs one data-collection episode on the CURRENT world."""
    world = client.get_world()

    # sync sim
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / args.fps
    world.apply_settings(settings)

    bp_lib = world.get_blueprint_library()
    
    # ego + camera
    ego = spawn_ego(world, bp_lib)
    camera, img_queue, set_recording = attach_front_rgb(
        world,
        bp_lib,
        ego,
        out_dir,
        args.rgb_width,
        args.rgb_height,
    )

    # optional background traffic
    traffic_actors = spawn_npcs(world, bp_lib, tm, n=args.n_npcs)

    # control provider
    agent = None
    if args.mode == "tm":
        ego.set_autopilot(True, tm.get_port())
    elif args.mode == "basic":
        from agents.navigation.basic_agent import BasicAgent
        agent = BasicAgent(ego)
    elif args.mode == "behavior":
        from agents.navigation.behavior_agent import BehaviorAgent
        agent = BehaviorAgent(ego, behavior="normal")

    meas_path = os.path.join(out_dir, "measurements.jsonl")
    meas_f = open(meas_path, "w", buffering=1)

    try:
        world.tick()  # warm-up for sensors

        stall_timer = 0
        recording = True

        steps_done = 0
        ticks_seen = 0
        TICK_CAP = args.steps * 5

        while steps_done < args.steps and ticks_seen < TICK_CAP:
            if agent is not None:
                control = agent.run_step()
                ego.apply_control(control)

            snapshot = world.tick()
            frame = snapshot
            ticks_seen += 1

            tf = ego.get_transform()
            vel = ego.get_velocity()
            spd_kmh = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2) * 3.6

            # update stall / recording state
            if spd_kmh < args.min_speed_kmh:
                stall_timer += 1
            else:
                stall_timer = 0

            should_record = (stall_timer < args.stall_patience)
            if should_record != recording:
                recording = should_record
                set_recording(recording)  # flip the camera saver on/off

            # only wait for and log frames when recording
            t0 = time.time()
            while frame not in img_queue and time.time() - t0 < 2.0:
                time.sleep(0.001)

            ctrl = ego.get_control()
            rec = {
                "frame": frame,
                "location": {"x": tf.location.x, "y": tf.location.y, "z": tf.location.z},
                "rotation": {"pitch": tf.rotation.pitch, "yaw": tf.rotation.yaw, "roll": tf.rotation.roll},
                "speed_kmh": spd_kmh,
                "control": {
                    "throttle": ctrl.throttle, "steer": ctrl.steer, "brake": ctrl.brake,
                    "reverse": ctrl.reverse, "hand_brake": ctrl.hand_brake,
                    "manual_gear_shift": ctrl.manual_gear_shift, "gear": ctrl.gear,
                },
                "image_path": f"images/{frame:06d}.png",
                "mode": args.mode,
                "town": Path(world.get_map().name).name,
                "recording": recording
            }
            meas_f.write(json.dumps(rec) + "\n")

            if recording:
                steps_done += 1

    finally:
        # 1. Stop sensors and ego autopilot
        camera.stop()
        if agent is None:
            ego.set_autopilot(False)

        # 2. Tell the TM to release control of the NPCs
        for a in traffic_actors:
            if a.is_alive:
                a.set_autopilot(False, tm.get_port())
        
        # 3. Destroy actors
        for a in traffic_actors:
            if a.is_alive: a.destroy()
        if ego.is_alive: ego.destroy()

        # 4. Tick once to process destruction
        world.tick()
        
        meas_f.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["tm","basic", "behavior"], default="tm")
    parser.add_argument("--out", default="out_autopilot")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-npcs", type=int, default=40)
    parser.add_argument("--rgb-width", type=int, default=1280)
    parser.add_argument("--rgb-height", type=int, default=720)
    parser.add_argument("--town", default=None, help="Run only on this town")
    parser.add_argument("--towns", default=None, help="Comma-separated list of towns (e.g., Town01,Town03)")
    parser.add_argument("--all-towns", action="store_true", help="Run on every installed map")
    parser.add_argument("--no-red-light", action="store_true", help="Traffic light always green")
    parser.add_argument("--min-speed-kmh", type=float, default=1.0,
                        help="Only record when ego speed >= this (km/h)")
    # TODO: max_speed_kmh not used yet
    parser.add_argument("--max-speed-kmh", type=float, default=None,
                        help="Enforce speed limit on vehicle control when in traffic manager mode")
    parser.add_argument("--stall-patience", type=int, default=1e6,
                        help="Consecutive slow frames before pausing recording")
    parser.add_argument("--start-carla", action="store_true", help="Start CARLA on run")
    parser.add_argument("--restart-carla", action="store_true", help="Restart CARLA on run")
    parser.add_argument("--gpu", type=bool, default=True, help="Use GPU to run CARLA")
    parser.add_argument("--carla-path", default=None, help="Path to CARLA script (i.e., CarlaUE4.sh)")
    args = parser.parse_args()

    random.seed(args.seed)
    ensure_dir(args.out)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_out = os.path.join(args.out, timestamp_str)
    os.makedirs(root_out, exist_ok=True)

    if args.restart_carla:
        restart_carla(args)
    elif args.start_carla:
        start_carla(args)
    
    client = carla.Client("localhost", args.port)
    client.set_timeout(60.0)

    # Decide which towns to run
    if args.all_towns:
        towns = sorted({normalize_map_name(m) for m in client.get_available_maps()})
    elif args.towns:
        towns = [t.strip() for t in args.towns.split(",") if t.strip()]
    elif args.town:
        towns = [args.town]
    else:
        # use current world only
        towns = [normalize_map_name(client.get_world().get_map().name)]

    tm = client.get_trafficmanager(args.tm_port)
    # Initial setup
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(args.seed)
    tm.set_global_distance_to_leading_vehicle(2.5)
    
    try:
        for town in towns:
            print(f"=== Collecting on {town} ===")
            
            # Force TM back to sync mode before loading the next world
            tm.set_synchronous_mode(True)

            # Load the town and tick once for stability
            world = client.load_world(town)
            time.sleep(10.0)
            world.tick()

            if args.no_red_light:
                disable_red_lights(world)

            # Per-town output folder
            town_out = os.path.join(root_out, town)
            ensure_dir(town_out)

            # Run one episode on this world
            collect_once_on_world(client, tm, args, town_out)

    finally:
        print("Collection finished. Restoring async mode.")
        # Make sure we have a valid world object for cleanup
        world = client.get_world()
        settings = world.get_settings()
        tm.set_synchronous_mode(False)
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None
        world.apply_settings(settings)

    print(f"Saved data to: {root_out}")

if __name__ == "__main__":
    main()
