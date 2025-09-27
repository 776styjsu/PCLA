import os, json, time, math, random, argparse
from pathlib import Path
from collections import deque
from datetime import datetime
import carla

def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

def normalize_map_name(m):  # "/Game/Carla/Maps/Town03" -> "Town03"
    return Path(m).name

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
        if not state["recording"]:
            return
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

    # Touch topology once (forces map/nav data to be resident)
    _ = world.get_map().get_topology()

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
    else:
        from agents.navigation.behavior_agent import BehaviorAgent
        agent = BehaviorAgent(ego, behavior="normal")
        dest = random.choice(world.get_map().get_spawn_points()).location
        agent.set_destination(agent._vehicle.get_location(), dest)

    meas_path = os.path.join(out_dir, "measurements.jsonl")
    meas_f = open(meas_path, "w", buffering=1)

    try:
        world.tick()  # warm-up for sensors

        stall_timer = 0
        recording = True

        steps_done = 0
        ticks_seen = 0
        TICK_CAP = args.steps * 50

        while steps_done < args.steps and ticks_seen < TICK_CAP:
            if agent is not None:
                if agent.done():
                    dest = random.choice(world.get_map().get_spawn_points()).location
                    agent.set_destination(agent._vehicle.get_location(), dest)
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

            if recording:
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
                }
                meas_f.write(json.dumps(rec) + "\n")

                steps_done += 1
            else:
                # when paused, skip waiting and skip logging (and don't count a step)
                pass
    finally:
        camera.stop()
        if agent is None:
            ego.set_autopilot(False)
        for a in traffic_actors:
            if a.is_alive: a.destroy()
        if ego.is_alive: ego.destroy()

        # restore async & close file
        tm.set_synchronous_mode(False)
        settings.synchronous_mode = False
        settings.fixed_delta_seconds = None

        world.apply_settings(settings)
        meas_f.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["tm","behavior"], default="tm")
    parser.add_argument("--out", default="out_autopilot")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=20500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-npcs", type=int, default=40)
    parser.add_argument("--rgb_width", type=int, default=1280)
    parser.add_argument("--rgb_height", type=int, default=720)
    parser.add_argument("--town", default=None, help="Run only on this town")
    parser.add_argument("--towns", default=None, help="Comma-separated list of towns (e.g., Town01,Town03)")
    parser.add_argument("--all-towns", action="store_true", help="Run on every installed map")
    parser.add_argument("--min_speed_kmh", type=float, default=3.0,
                        help="Only record when ego speed >= this (km/h)")
    parser.add_argument("--stall_patience", type=int, default=5,
                        help="# consecutive slow frames before pausing recording")
    args = parser.parse_args()

    random.seed(args.seed)
    ensure_dir(args.out)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_out = os.path.join(args.out, timestamp_str)
    os.makedirs(root_out, exist_ok=True)

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
    tm.set_synchronous_mode(False)
    tm.set_random_device_seed(args.seed)
    tm.set_global_distance_to_leading_vehicle(2.5)
    print(towns)
    for town in towns:
        print(f"=== Collecting on {town} ===")
        # Load the town (reload even if it's already active to reset state)
        world = client.load_world(town)

        # Per-town output folder
        town_out = os.path.join(root_out, town)
        ensure_dir(town_out)

        # Run one episode on this world
        collect_once_on_world(client, tm, args, town_out)

    print(f"Saved data to: {root_out}")

if __name__ == "__main__":
    main()
