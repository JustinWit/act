import argparse
import os
import pickle

# from constants import TASK_CONFIGS
import IPython
import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import rearrange
from tqdm import tqdm

from policy import ACTPolicy, CNNMLPPolicy
from utils import (  # data functions  # robot functions  # helper functions
    combined_std,
    compute_dict_mean,
    detach_dict,
    load_data,
    preproc_imgs,
    set_seed,
)

e = IPython.embed
import time

import wandb


def main(args):
    set_seed(1)
    # command line parameters
    is_eval = args['eval']
    ckpt_dir = args['ckpt_dir']
    policy_class = args['policy_class']
    # onscreen_render = args['onscreen_render']
    task_name = args['task_name']
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']
    num_epochs = args['num_epochs']
    log_wandb = not args['no_wandb']

    # get task parameters
    # is_sim = task_name[:4] == 'sim_'
    # if is_sim:
    #     from constants import SIM_TASK_CONFIGS
    #     task_config = SIM_TASK_CONFIGS[task_name]
    # else:
    # task_config = TASK_CONFIGS[task_name]
    # episode_len = task_config['episode_len']
    camera_names = ['wrist_camera', 'front_camera']

    # fixed parameters
    state_dim = 7
    lr_backbone = 1e-5
    backbone = 'resnet18'
    if policy_class.upper() == 'ACT':
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {'lr': args['lr'],
                         'num_queries': args['chunk_size'],
                         'kl_weight': args['kl_weight'],
                         'hidden_dim': args['hidden_dim'],
                         'dim_feedforward': args['dim_feedforward'],
                         'lr_backbone': lr_backbone,
                         'backbone': backbone,
                         'enc_layers': enc_layers,
                         'dec_layers': dec_layers,
                         'nheads': nheads,
                         'camera_names': camera_names,
                         }
    elif policy_class == 'CNNMLP':
        policy_config = {'lr': args['lr'], 'lr_backbone': lr_backbone, 'backbone' : backbone, 'num_queries': 1,
                         'camera_names': camera_names,}
    else:
        raise NotImplementedError

    config = {
        'num_epochs': num_epochs,
        'ckpt_dir': ckpt_dir,
        # 'episode_len': episode_len,
        'state_dim': state_dim,
        'lr': args['lr'],
        'policy_class': policy_class,
        # 'onscreen_render': onscreen_render,
        'policy_config': policy_config,
        # 'task_name': task_name,
        'seed': args['seed'],
        'temporal_agg': args['temporal_agg'],
        'camera_names': camera_names,
        # 'real_robot': not is_sim,
        'log_wandb': log_wandb,
        'batch_size': batch_size_train,
        "gripper_proprio": args['gripper_proprio'],
        'absolute_actions': args['absolute_actions'],
        'full_size_img': args['full_size_img'],
        'real_ratio': args['real_ratio'],
    }

    if is_eval:
        ckpt_name = args['ckpt_name']
        eval_bc(
            config,
            ckpt_name,
            not args['no_proprioception'],
            save_episode=True,
        )
        exit()

    if log_wandb:
        wandb.init(
            project='act-training',
            config=config,
            name=args['run_name'],
            entity="jwit3-georgia-institute-of-technology",
        )

    # if is_eval:
    #     ckpt_names = [f'policy_best.ckpt']
    #     results = []
    #     for ckpt_name in ckpt_names:
    #         success_rate, avg_return = eval_bc(config, ckpt_name, save_episode=True)
    #         results.append([ckpt_name, success_rate, avg_return])

    #     for ckpt_name, success_rate, avg_return in results:
    #         print(f'{ckpt_name}: {success_rate=} {avg_return=}')
    #     print()
    #     exit()

    dataset_dir = os.path.join("datasets", task_name)
    num_episodes = len([x for x in os.listdir(dataset_dir) if x.endswith('.h5')])
    train_dataloader, val_dataloader, stats, _ = load_data(
        dataset_dir,
        num_episodes,
        camera_names,
        int(batch_size_train * (1 - args['real_ratio'])),
        batch_size_val,
        proprioception=not args['no_proprioception'],
        chunk_size=args['chunk_size'],
        preload_to_cpu=args['preload_to_cpu'],
        preload_to_gpu=args['preload_to_gpu'],
        gripper_proprio=args['gripper_proprio'],
        absolute_actions=args['absolute_actions'],
        full_size_img=args['full_size_img'],
        )
    # breakpoint()
    # create dataloader for real data
    if args['real_ratio'] > 0:
        real_dataset_dir = os.path.join("datasets", args['real_data_dir'])
        real_num_episodes = len([x for x in os.listdir(real_dataset_dir) if x.endswith('.h5')])
        real_train_dataloader, real_val_dataloader, real_stats, _ = load_data(
            real_dataset_dir,
            real_num_episodes,
            camera_names,
            int(batch_size_train * args['real_ratio']),
            batch_size_val,
            proprioception=not args['no_proprioception'],
            chunk_size=args['chunk_size'],
            preload_to_cpu=args['preload_to_cpu'],
            preload_to_gpu=args['preload_to_gpu'],
            gripper_proprio=args['gripper_proprio'],
            absolute_actions=args['absolute_actions'],
            full_size_img=args['full_size_img'],
            )

        p = stats['total_n'] / (stats['total_n'] + real_stats['total_n'])
        combined_action_mean = p * stats['action_mean'] + (1 - p) * real_stats['action_mean']
        combined_qpos_mean = p * stats['qpos_mean'] + (1 - p) * real_stats['qpos_mean']
        final_stats = {
            "action_mean": combined_action_mean,
            "action_std": combined_std(stats['action_mean'], stats['action_std'], stats['total_n'], real_stats['action_mean'], real_stats['action_std'], real_stats['total_n'], combined_action_mean),
            "qpos_mean": combined_qpos_mean,
            "qpos_std": combined_std(stats['qpos_std'], stats['qpos_std'], stats['total_n'], real_stats['qpos_mean'], real_stats['qpos_std'], real_stats['total_n'], combined_qpos_mean),
            # "example_qpos": stats['example_qpos'],  # maybe delete this ?
            "use_proprioception": stats['use_proprioception'],
            'use_gripper_proprio': stats['use_gripper_proprio'],
            "absolute_actions": stats['absolute_actions'],
            "full_size_img": stats['full_size_img'],
            "total_n": stats['total_n'] + real_stats['total_n'],
        }

    # update norm stats
    if args['real_ratio'] == 0:
        final_stats = stats
    final_stats['action_std'] = torch.clip(final_stats['action_std'], 1e-2, np.inf)
    final_stats['qpos_std'] = torch.clip(final_stats['qpos_std'], 1e-2, np.inf)
    train_dataloader.dataset.set_norm_stats(final_stats)
    if args['real_ratio'] > 0:
        real_train_dataloader.dataset.set_norm_stats(final_stats)

    # save dataset stats
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    stats_path = os.path.join(ckpt_dir, 'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(final_stats, f)

    if args['real_ratio'] > 0:
        best_ckpt_info = train_bc(train_dataloader, real_train_dataloader, val_dataloader, config)
    else:
        best_ckpt_info = train_bc(train_dataloader, None, val_dataloader, config)
    # best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    # # save best checkpoint
    # ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')
    # torch.save(best_state_dict, ckpt_path)
    # print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}')


def make_policy(policy_class, policy_config):
    if policy_class.upper() == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class.upper() == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'CNNMLP':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def get_image(ts, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image


def eval_bc(config, ckpt_name, proprioception, save_episode=True):
    # Imports for controlling Robot
    import cv2
    from deoxys.experimental.motion_utils import reset_joints_to
    from deoxys.franka_interface import FrankaInterface
    from openteach.utils.network import ZMQCameraSubscriber

    # from deoxys.utils.transform_utils import quat2axisangle, mat2quat, euler2mat
    # from scipy.spatial.transform import Rotation as R
    from deoxys_transform_utils import quat2axisangle
    from record_eval import RecordEval

    # ACT variables
    set_seed(1000)
    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    # real_robot = config['real_robot']
    policy_class = config['policy_class']
    # onscreen_render = config['onscreen_render']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    # max_timesteps = config['episode_len']
    temporal_agg = config['temporal_agg']
    # onscreen_cam = 'angle'

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(ckpt_dir, 'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
        assert stats['use_proprioception'] == proprioception
        assert stats['use_gripper_proprio'] == config['gripper_proprio']
        assert stats['absolute_actions'] == config['absolute_actions']
        assert stats['full_size_img'] == config['full_size_img']

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean'].cpu().numpy()) / stats['qpos_std'].cpu().numpy()
    post_process = lambda a: a * stats['action_std'].cpu().numpy() + stats['action_mean'].cpu().numpy()


    # Configure Camera Stream
    overview_subscriber = ZMQCameraSubscriber(
            # host = "143.215.128.151",
            host = "172.16.0.1",
            port = "10007",
            topic_type = 'RGB'
        )

    wrist_subscriber = ZMQCameraSubscriber(
            # host = "143.215.128.151",
            host = "172.16.0.1",
            port = "10006",
            topic_type = 'RGB'
        )

    # Initialize robot
    robot_interface = FrankaInterface(
        os.path.join('/home/ripl/openteach/configs', 'deoxys.yml'), use_visualizer=False,
        control_freq=5,  # TODO: change this to 5 or even higher?
        state_freq=200
    )  # copied from playback_demo.py

    from constants import DEFAULT_CONTROLLER
    DEFAULT_CONTROLLER['is_delta'] = not config['absolute_actions']
    # Golden resetting joints
    reset_joint_positions = [
            0.09162008114028396,
            -0.19826458111314524,
            -0.01990020486871322,
            -2.4732269941140346,
            -0.01307073642274261,
            2.30396583422025,
            0.8480939705504309,
        ]

    while robot_interface.last_gripper_q is None or robot_interface.last_gripper_q < 0.07:
        robot_interface.gripper_control(-1.0)
    time.sleep(1)
    reset_joints_to(robot_interface, reset_joint_positions, timeout=100)  # reset joints to home position

#     # load environment
#     if real_robot:
#         from aloha_scripts.robot_utils import move_grippers # requires aloha
#         from aloha_scripts.real_env import make_real_env # requires aloha
#         env = make_real_env(init_node=True)
#         env_max_reward = 0
#     else:
#         from sim_env import make_sim_env
#         env = make_sim_env(task_name)
#         env_max_reward = env.task.max_reward

    query_frequency = policy_config['num_queries']
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config['num_queries']

    max_timesteps = int(200) # may increase for real-world tasks, TODO

    # num_rollouts = 1
#     episode_returns = []
#     highest_rewards = []
    # for rollout_id in range(num_rollouts):
#         rollout_id += 0
#         ### set task
#         if 'sim_transfer_cube' in task_name:
#             BOX_POSE[0] = sample_box_pose() # used in sim reset
#         elif 'sim_insertion' in task_name:
#             BOX_POSE[0] = np.concatenate(sample_insertion_pose()) # used in sim reset

#         ts = env.reset()

#         ### onscreen render
#         if onscreen_render:
#             ax = plt.subplot()
#             plt_img = ax.imshow(env._physics.render(height=480, width=640, camera_id=onscreen_cam))
#             plt.ion()

        ### evaluation loop
    try:
        if temporal_agg:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, state_dim]).cuda()

        qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
        # image_list = [] # for visualization
        video_recorder_side = RecordEval(ckpt_name, 10005, name_suffix=args['vid_name'])
        video_recorder_wrist = RecordEval(ckpt_name, 10006, name_suffix=args['vid_name'])
        video_recorder_front = RecordEval(ckpt_name, 10007, name_suffix=args['vid_name'])
        video_recorder_side.start()
        video_recorder_wrist.start()
        video_recorder_front.start()
        # qpos_list = []
        # target_qpos_list = []
        # rewards = []
        with torch.inference_mode():
            for t in range(max_timesteps):
#                 ### update onscreen render and wait for DT
#                 if onscreen_render:
#                     image = env._physics.render(height=480, width=640, camera_id=onscreen_cam)
#                     plt_img.set_data(image)
#                     plt.pause(DT)

                ### process previous timestep to get qpos and image_list
                over_frames = overview_subscriber.recv_rgb_image()
                wrist_frames = wrist_subscriber.recv_rgb_image()
                if over_frames[0] is None:
                    continue

                # add dummy camera image to match the training data to use the same image preprocessing function
                color_frames = np.stack([
                    np.zeros_like(wrist_frames[0]),  # dummy
                    wrist_frames[0],
                    over_frames[0]
                    ], axis=0)  # shape: (3, 360, 640, 3)

                curr_image = preproc_imgs(
                    color_frames[np.newaxis, ...], # (1, 3, H, W, 3)
                    full_size_img=config['full_size_img'],
                    ).cuda()  # output is shape (2, 3, H, W)
                curr_image = curr_image.unsqueeze(0)  # (1, 2, 3, H, W) need to add batch dim. During training, this is done in the dataloader

                if proprioception:
                    quat, pos = robot_interface.last_eef_quat_and_pos
                    if quat2axisangle(quat)[0] > 0:
                        axis_angle = quat2axisangle(quat)
                    else:
                        axis_angle = quat2axisangle(-quat)
                    qpos = np.concatenate([
                        pos.flatten(),
                        axis_angle,
                        [robot_interface.last_gripper_q] if config['gripper_proprio'] else [0.0],
                        ])
                    qpos = pre_process(qpos)
                else:
                    qpos = np.zeros(7)
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                qpos_history[:, t] = qpos

                ### query policy
                if config['policy_class'] == "ACT":
                    if t % query_frequency == 0:
                        all_actions = policy(qpos, curr_image)
                    if temporal_agg:
                        all_time_actions[[t], t:t+num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        raw_action = all_actions[:, t % query_frequency]
                elif config['policy_class'] == "CNNMLP":
                    raw_action = policy(qpos, curr_image)
                else:
                    raise NotImplementedError

                ### post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                target_qpos = action

                ### Move the Robot
                # action[3:6] = quat2axisangle(mat2quat(euler2mat(action[3:6])))  # convert euler to axis-angle
                # action = normalize_gripper_action(action, binarize=True)  # normalize gripper action
                action[-1] = 1 if action[-1] > 0 else -1  # binarize gripper action
                print(t, action)

                robot_interface.control(
                    controller_type='OSC_POSE',
                    action=action[:6],
                    controller_cfg=DEFAULT_CONTROLLER,
                )
                robot_interface.gripper_control(action[-1])


                curr_image_np = curr_image[0].cpu().numpy() # shape (2, 3, H, W)
                curr_image_np = np.transpose(curr_image_np, (0, 2, 3, 1))

                # Display the image Press 'q' to exit
                cv2.imshow("Camera", cv2.cvtColor(np.hstack((curr_image_np[0], curr_image_np[1])), cv2.COLOR_RGB2BGR))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break


                ### for visualization
                # qpos_list.append(qpos_numpy)
                # target_qpos_list.append(target_qpos)
                # rewards.append(ts.reward)
    except KeyboardInterrupt:
        print("KeyboardInterrupt, stopping early and saving video...")
    finally:
        video_recorder_side.stop(os.path.join(ckpt_dir, 'videos_side'))
        video_recorder_front.stop(os.path.join(ckpt_dir, 'videos_front'))
        video_recorder_wrist.stop(os.path.join(ckpt_dir, 'videos_wrist'))
        cv2.destroyAllWindows()

#             plt.close()
        # if real_robot:
        #     move_grippers([env.puppet_bot_left, env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)  # open
        #     pass

#         rewards = np.array(rewards)
#         episode_return = np.sum(rewards[rewards!=None])
#         episode_returns.append(episode_return)
#         episode_highest_reward = np.max(rewards)
#         highest_rewards.append(episode_highest_reward)
#         print(f'Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}')

        # if save_episode:
            # save_videos(image_list, DT, video_path=os.path.join(ckpt_dir, f'video{rollout_id}.mp4'))
#     success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
#     avg_return = np.mean(episode_returns)
#     summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
#     for r in range(env_max_reward+1):
#         more_or_equal_r = (np.array(highest_rewards) >= r).sum()
#         more_or_equal_r_rate = more_or_equal_r / num_rollouts
#         summary_str += f'Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n'

#     print(summary_str)

#     # save success rate to txt
#     result_file_name = 'result_' + ckpt_name.split('.')[0] + '.txt'
#     with open(os.path.join(ckpt_dir, result_file_name), 'w') as f:
#         f.write(summary_str)
#         f.write(repr(episode_returns))
#         f.write('\n\n')
#         f.write(repr(highest_rewards))

#     return success_rate, avg_return


def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data
    image_data, qpos_data, action_data, is_pad = image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda()
    return policy(qpos_data, image_data, action_data, is_pad)


def train_bc(train_dataloader, real_train_dataloader, val_dataloader, config):
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']

    set_seed(seed)

    policy = make_policy(policy_class, policy_config)
    policy.cuda()
    optimizer = make_optimizer(policy_class, policy)

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    for epoch in tqdm(range(num_epochs)):
        print(f'\nEpoch {epoch}')
        '''
        # validation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(data, policy)
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary['loss']
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
        print(f'Val loss:   {epoch_val_loss:.5f}')
        summary_string = ''
        if epoch % 10 == 0:
            wandb.log({'epoch': epoch})
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
            if epoch % 10 == 0:
                wandb.log({f'val_{k}': v.item()})
        print(summary_string)
        '''

        # training
        policy.train()
        optimizer.zero_grad()
        if real_train_dataloader is not None:
            real_data_iter = iter(real_train_dataloader)

        for batch_idx, data in enumerate(train_dataloader):
            if real_train_dataloader is not None:
                try:
                    real_data = next(real_data_iter)
                except StopIteration:
                    real_data_iter = iter(real_train_dataloader)
                    real_data = next(real_data_iter)
                data = [torch.cat((data[i], real_data[i])) for i in range(len(data))]

            forward_dict = forward_pass(data, policy)
            # backward
            loss = forward_dict['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
        epoch_summary = compute_dict_mean(train_history[(batch_idx+1)*epoch:(batch_idx+1)*(epoch+1)])
        epoch_train_loss = epoch_summary['loss']
        print(f'Train loss: {epoch_train_loss:.5f}')
        summary_string = ''
        if epoch % 10 == 0 and config['log_wandb']:
            wandb_log_dict = {'epoch': epoch}
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
            if epoch % 10 == 0 and config['log_wandb']:
                wandb_log_dict.update({f'train_{k}': v.item()})
        if epoch % 10 == 0 and config['log_wandb']:
            wandb.log(wandb_log_dict)
        print(summary_string)

        if epoch % 1000 == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
            torch.save(policy.state_dict(), ckpt_path)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

    ckpt_path = os.path.join(ckpt_dir, 'policy_last.ckpt')
    torch.save(policy.state_dict(), ckpt_path)

    # best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    # ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{best_epoch}_seed_{seed}.ckpt')
    # torch.save(best_state_dict, ckpt_path)
    # print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')

    # save training curves
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

    return best_ckpt_info


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(np.linspace(0, num_epochs-1, len(train_history)), train_values, label='train')
        plt.plot(np.linspace(0, num_epochs-1, len(validation_history)), val_values, label='validation')
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    print(f'Saved plots to {ckpt_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    # parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name')
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)
    parser.add_argument('--run_name', action='store', type=str, help='run_name', required=True)
    parser.add_argument('--no_wandb', action='store_true', required=False)
    parser.add_argument('--ckpt_name', action='store', type=str, help='checkpoint name, used only for eval')

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--temporal_agg', action='store_true')
    parser.add_argument('--no_proprioception', action='store_true')
    parser.add_argument('--preload_to_cpu', action='store_true')
    parser.add_argument('--preload_to_gpu', action='store_true')
    parser.add_argument('--gripper_proprio', action='store_true')
    parser.add_argument('--absolute_actions', action='store_true')
    parser.add_argument('--vid_name', type=str, help='video name for eval', required=False, default='')
    parser.add_argument('--full_size_img', action='store_true')
    parser.add_argument('--real_ratio', action='store', type=float, help='proportion of real to sim', required=False, default=0)
    parser.add_argument('--real_data_dir', action='store', type=str, help='real_data_dir', required=False)
    args = vars(parser.parse_args())
    if args['gripper_proprio']:
        assert not args['no_proprioception']
    if args['real_ratio'] > 0 or args['real_data_dir']:
        assert args['real_data_dir'] and args['real_ratio'] > 0, "Must provide both real_data_dir and real_ratio"

    if args['preload_to_cpu'] or args['preload_to_gpu']:
        assert not (args['preload_to_cpu'] and args['preload_to_gpu']), "Cannot set preload to both CPU and GPU"

    main(args)
