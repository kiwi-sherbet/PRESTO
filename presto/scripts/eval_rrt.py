import sys
import numpy as np
import argparse
import random
import yaml
import json
import git
import os

git_repo = git.Repo(__file__, search_parent_directories=True)
git_root = git_repo.git.rev_parse("--show-toplevel")
sys.path.append(git_root)

from external.legato.simulator import SETUPS
from external.legato.simulator.render import CV2Renderer
from external.legato.simulator.envs import *
from external.legato.utils import geom
import external.legato.simulator.robots.panda as panda
import time

def stastics(
         save_dir=os.path.join(git_root, "data", "eval", "bi-rrt"),
         data_label="obj-0-0",
         **kwargs):
    
    log_path = os.path.join(save_dir, data_label, "logs.json")

    original_stdout = sys.stdout # Save a reference to the original standard output
    with open(os.path.join(save_dir, data_label, "statistics.txt"), 'w') as f:
        sys.stdout = f # Change the standard output to the file we created.
        read_json(log_path)
        sys.stdout = original_stdout # Reset the standard output to its original value

def print_statistics(data):

    env_list = [env for key, env in data.items() if not env["error"]]
        
    # Dictionary to store values for each valN
    values_dict = {}

    # Iterate over each environment and collect valN values
    for env in env_list:
        for key, value in env.items():
            if key == "error":
                continue
            if key not in values_dict:
                values_dict[key] = []
            values_dict[key].append(value)

    # Print statistics for each values
    for key, values in values_dict.items():
        values_array = np.array(values, dtype=np.float64)  # Convert to float64 to handle NaNs

        average = np.mean(values_array)
        std_dev = np.std(values_array)
        min_value = np.min(values_array)
        max_value = np.max(values_array)

        print(f"Statistics for {key}:")
        print(f"  Average: {average}")
        print(f"  Standard Deviation: {std_dev}")
        print(f"  Minimum: {min_value}")
        print(f"  Maximum: {max_value}")
        print()

    # Print stastics for error values
    error_values = [env.get("error", False) for env in data.values()]
    print("Statistics for error: {}/{}".format(sum(error_values), len(error_values)))
    print("Number of samples: ", len(env_list))


def update_json(file_path, env_id, new_values):
    # Load existing data from the JSON file if it exists
    data = {}
    if os.path.exists(file_path):
        with open(file_path, mode='r') as jsonfile:
            data = json.load(jsonfile)
    
    # Update the data with the new values for the given ENV_ID
    data[env_id] = new_values
    
    # Save the updated data back to the JSON file
    with open(file_path, mode='w') as jsonfile:
        json.dump(data, jsonfile, indent=4)
    
    # Calculate and print statistics
    print_statistics(data)


def read_json(file_path):
    # Load existing data from the JSON file if it exists
    data = {}
    if os.path.exists(file_path):
        with open(file_path, mode='r') as jsonfile:
            data = json.load(jsonfile)
    print_statistics(data)


def eval(gui, 
         seed, 
         max_runtime,
         cam_name, robot_name,
         vertical_slots, horizontal_slots, 
         env_dir=os.path.join(git_root, "data", "presto_cabinet_eval_rrt","env_info"),
         traj_dir=os.path.join(git_root, "data", "presto_cabinet_eval_rrt", "traj_info"),
         save_dir=os.path.join(git_root, "data", "eval", "bi-rrt"),
         data_label="obj-0-0",
         trj_idx=1,
         **kwargs):

    assert robot_name == "panda"

    if seed is None:
        seed = random.randint(0, 10**3-1)
    random.seed(seed)
    np.random.seed(seed)

    np.set_printoptions(precision=3)

    setup = SETUPS[robot_name]
    robot_type = setup['robot_type']
    env_config = setup['env_config']
    mp_solver = setup['mp_solver']
    mp_config = setup['mp_config']
    mp_config.MAX_RUNTIME = max_runtime

    env_class = PrestoPrimitiveEnv

    primitive_objects = {
        'cuboid': {},
        'cylinder': {},
        'sphere': {},
    }

    if not os.path.exists(os.path.join(traj_dir, data_label, "{:03d}.yml".format(trj_idx))):
        print("File not found: ", os.path.join(traj_dir, data_label, "{:03d}.yml".format(trj_idx)))
        return

    with open(os.path.join(traj_dir, data_label, "{:03d}.yml".format(trj_idx)), "r") as f:
        traj_info = yaml.load(f, Loader=yaml.FullLoader)
        env_path = os.path.join(env_dir, traj_info['env'])
        init_joint = np.array(traj_info['init']).tolist()
        goal_joint = np.array(traj_info['goal']).tolist()
        start = {"joint_{}".format(i): init_joint[i] for i in range(len(init_joint))}
        goal = {"joint_{}".format(i): goal_joint[i] for i in range(len(goal_joint))}

    with open(env_path, "r") as f:
        env_info = yaml.load(f, Loader=yaml.FullLoader)
        joint_ids = env_info['joint_ids']
        init_qpos = np.array(env_info['init_qpos'])
        for obj_name_g0 in env_info['col_geom_ids']['object']:
            if "shelf" in obj_name_g0:
                continue
            if "table" in obj_name_g0:
                continue
            obj_name = obj_name_g0.split("_g0")[0]
            joint_id = joint_ids["{}_joint0".format(obj_name)]
            col_geom_info = env_info['col_geom_infos']["{}_g0".format(obj_name)]
            joint_vals = init_qpos[joint_id: joint_id+7]
            pos = joint_vals[:3].tolist()
            quat = np.array((joint_vals[3:]))
            quat = quat[[[1, 2, 3, 0]]]
            euler = geom.quat_to_euler(quat).tolist()                
            dims = [float(x) for x in col_geom_info['size'].split()]
            
            if col_geom_info['type'] == 'box': 
                prim_type = 'cuboid'
            else:
                prim_type = col_geom_info['type']            
    
            primitive_objects[prim_type][obj_name] = {
                'pose': pos + euler,
                'dims': dims,
            }

    env = env_class(env_config=env_config,
                    vertical_slots=vertical_slots,
                    horizontal_slots=horizontal_slots,
                    primitive_objects=primitive_objects,
                    )
    
    env.reset(mode="forward", initial_qpos=init_qpos)

    mp_solver = mp_solver(config=mp_config, env=env)

    init_time = time.time()
    try:
        path = mp_solver.solve(start=start, goal=goal)
        comp_time = time.time() - init_time
        error_int = 0
        print("Time taken: ", comp_time)
    except Exception as error:
        path = None
        comp_time = np.nan
        error_int = 1
        print(error)

    success = int(path is not None and len(path) > 0)

    os.makedirs(os.path.join(save_dir, data_label), exist_ok=True)
    os.makedirs(os.path.join(save_dir, data_label, "videos"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, data_label, "paths"), exist_ok=True)

    if success:
        renderer = CV2Renderer(device_id=-1, sim=env.sim, cam_name=cam_name,
                            width=480, height=360, 
                            gui=gui,
                            save_path=os.path.join(save_dir, data_label, "videos", "{:03d}.mp4".format(trj_idx))
                            )
        env.set_renderer(renderer)
        mp_solver.test(path, mode="forward")
        renderer.close()

        # Save paths as yaml
        with open(os.path.join(save_dir, data_label, "paths", "{:03d}.yml".format(trj_idx)), "w") as f:
            yaml.dump(np.array(path).tolist(), f)

    log_path = os.path.join(save_dir, data_label, "logs.json")
    update_json(log_path, trj_idx, {"time": comp_time, "success": success, "collision_ratio": 0, "collision_distance": 0, "error": error_int})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=None, type=int,
                        help='The random seed to use.')
    parser.add_argument("--gui", type=int, default=1, help="")
    parser.add_argument("--robot", type=str, default="panda", help="")
    parser.add_argument("--vertical_slots", type=int, default=3, help="")
    parser.add_argument("--horizontal_slots", type=int, default=1, help="")
    parser.add_argument("--label", type=str, default="obj-1-1", help="")
    parser.add_argument("--data_path", type=str, default="data/presto_cabinet_eval_rrt", help="")
    parser.add_argument("--save_path", type=str, default="data/eval/bi-rrt", help="")
    parser.add_argument("--max_runtime", type=int, default=500, help="")
    parser.add_argument(
        "--cam",
        type=str,
        default='diagonalview',
        help="",
    )
    args = parser.parse_args()

    seed = args.seed
    gui = args.gui
    cam_name = args.cam
    robot_name = args.robot
    vertical_slots = args.vertical_slots
    horizontal_slots = args.horizontal_slots
    data_label = args.label
    max_runtime = args.max_runtime

    save_dir = os.path.join(git_root, args.save_path, "bi-rrt_{}".format(max_runtime))
    
    if not os.path.exists(os.path.join(save_dir, data_label)):
        os.makedirs(os.path.join(save_dir, data_label), exist_ok=True)
    traj_indices = [int(f.split(".")[0]) for f in os.listdir(os.path.join(args.data_path, "traj_info", data_label))]

    for trj_idx in traj_indices:
        eval(gui=gui, seed=0, max_runtime=args.max_runtime,
            cam_name=cam_name, robot_name=robot_name,
            vertical_slots=vertical_slots, horizontal_slots=horizontal_slots,
            trj_idx=trj_idx, data_label=data_label,
            env_dir=os.path.join(args.data_path, "env_info"),
            traj_dir=os.path.join(args.data_path, "traj_info"),
            save_dir=save_dir,
            )

    stastics(
        data_label=data_label,
        save_dir=save_dir)