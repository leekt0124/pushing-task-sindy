import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib
# from numpngw import write_apng
from IPython.display import Image
from tqdm.notebook import tqdm

import os
import sys

from panda_pushing_env import PandaImageSpacePushingEnv
from visualizers import GIFVisualizer, NotebookVisualizer

from learning_latent_dynamics import *
from utils import *

# for GIF
from PIL import Image

# Load the given data insead of the collected.
collected_data = np.load('pushing_image_data.npy', allow_pickle=True)

# Train the dynamics model
LATENT_DIM = 16
ACTION_DIM = 3
NUM_CHANNELS = 1
NUM_STEPS = 1

single_step_latent_dynamics_model = SINDyDynamicsModel(latent_dim=LATENT_DIM, action_dim=ACTION_DIM, num_channels=NUM_CHANNELS)
# single_step_latent_dynamics_model = LatentDynamicsModel(latent_dim=LATENT_DIM, action_dim=ACTION_DIM, num_channels=NUM_CHANNELS)


# Compute normalization constants
train_loader, val_loader, norm_constants = process_data_multiple_step(collected_data, batch_size=500, num_steps=NUM_STEPS)
norm_tr = NormalizationTransform(norm_constants)

state_loss_fn = nn.MSELoss()
latent_loss_fn = nn.MSELoss()
multistep_loss = SINDyLoss(state_loss_fn, latent_loss_fn)
# multistep_loss = MultiStepLoss(state_loss_fn, latent_loss_fn, alpha=0.1)


TRAIN_MODEL = False
if TRAIN_MODEL:
    # TODO: Train latent_dynamics_model
    # --- Your code here
    NUM_EPOCHS = 250
    # NUM_EPOCHS = 250
    LR = 0.7 * 1e-3


    optimizer = optim.Adam(single_step_latent_dynamics_model.parameters(), lr=LR)
    pbar = tqdm(range(NUM_EPOCHS))
    train_losses = []
    for epoch_i in pbar:
        train_loss_i = 0.
        # --- Your code here
        for batch_idx, sample in enumerate(train_loader):
            # print("batch_idx = ", batch_idx)
            states_batch, actions_batch = sample["states"], sample["actions"]

            optimizer.zero_grad()
            loss = multistep_loss(single_step_latent_dynamics_model, states_batch, actions_batch)
            loss.backward()
            optimizer.step()
            
            # ---
            train_loss_i += loss.item()
            pbar.set_description(f'Latent dim {LATENT_DIM} - Loss: {train_loss_i:.4f}')
            train_losses.append(train_loss_i)

    losses = train_losses


    # plot train loss and test loss:
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 3))
    axes = [axes]
    axes[0].plot(losses, label=f'latent_dim: {LATENT_DIM}')
    axes[0].grid()
    axes[0].legend()
    axes[0].set_title('Train Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Train Loss')
    axes[0].set_yscale('log')


    # ---

    # save model:
    save_path = os.path.join('single_step_latent_dynamics_model.pt')
    torch.save(single_step_latent_dynamics_model.state_dict(), save_path)

    plt.show()

    # Visualize the ability to perform single-step and multi-step state prediction:
    traj = collected_data[0]
    evaluate_model_plot(single_step_latent_dynamics_model, traj, norm_tr)

    plt.show()

# Load the model
latent_dynamics_model = SINDyDynamicsModel(latent_dim=LATENT_DIM, action_dim=ACTION_DIM, num_channels=NUM_CHANNELS)
# latent_dynamics_model = LatentDynamicsModel(latent_dim=LATENT_DIM, action_dim=ACTION_DIM, num_channels=NUM_CHANNELS)
model_path = os.path.join('single_step_latent_dynamics_model.pt')
latent_dynamics_model.load_state_dict(torch.load(model_path))


visualizer = None

target_state = np.array([0.7, 0., 0.])

env = PandaImageSpacePushingEnv(visualizer=visualizer, render_non_push_motions=False,  camera_heigh=800, camera_width=800, render_every_n_steps=5, grayscale=True)
state_0 = env.reset()
env.object_target_pose = env._planar_pose_to_world_pose(target_state)
controller = PushingLatentController(env, latent_dynamics_model, latent_space_pushing_cost_function,norm_constants, num_samples=100, horizon=10)

state = state_0

# num_steps_max = 100
num_steps_max = 20

for i in tqdm(range(num_steps_max)):
    action = controller.control(state)
    state, reward, done, _ = env.step(action)
    end_pose = env.get_object_pos_planar()
    goal_distance = np.linalg.norm(end_pose[:2]-target_state[:2]) # evaluate only position, not orientation
    goal_reached = goal_distance < BOX_SIZE
    if done or goal_reached:
        break

# Generated GIF
frames = [Image.fromarray(frame) for frame in env.frames]
frame_one = frames[1]
frame_one.save("demo.gif", format="GIF", append_images=frames, save_all=True, duration=100, loop=0)


print(f'GOAL REACHED: {goal_reached}')