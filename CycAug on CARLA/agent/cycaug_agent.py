# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

import agent.cycaug_utils

class RandomShiftsAug(nn.Module):
    def __init__(self, pad):
        super().__init__()
        self.pad = pad

    def forward(self, x):
        num_cameras = 3
        n, c, h, w = x.size()
        x = x.reshape(n,c,h,num_cameras,-1)
        x = x.permute(0,1,3,2,4)
        x = x.reshape(n,c*num_cameras,h,-1)
        n,_c,h,w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        sample_x = F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)
        sample_x = sample_x.reshape(n,c,num_cameras,h,-1)
        sample_x = sample_x.permute(0,1,3,2,4)
        sample_x = sample_x.reshape(n,c,h,-1)
        return sample_x

class PadResizePlus(nn.Module):
    def __init__(self, highest_pad_strength):
        super().__init__()
        self.highest_pad_strength = int(highest_pad_strength)

    def crop(self, imgs, pad_x, pad_y):
        n, c, h_pad, w_pad = imgs.size()

        # calculate the crop size
        crop_x = w_pad - pad_x
        crop_y = h_pad - pad_y

        # create a grid for cropping
        eps_x = 1.0 / w_pad
        eps_y = 1.0 / h_pad
        x_range = torch.linspace(-1.0 + eps_x, 1.0 - eps_x, w_pad, device=imgs.device, dtype=imgs.dtype)[:crop_x]
        y_range = torch.linspace(-1.0 + eps_y, 1.0 - eps_y, h_pad, device=imgs.device, dtype=imgs.dtype)[:crop_y]

        grid_y, grid_x = torch.meshgrid(y_range, x_range)

        base_grid = torch.stack([grid_x, grid_y], dim=-1)
        # print('base_grid.shape', base_grid.shape)

        shift_x = torch.randint(0, pad_x + 1, size=(n, 1, 1, 1), device=imgs.device, dtype=imgs.dtype)
        shift_y = torch.randint(0, pad_y + 1, size=(n, 1, 1, 1), device=imgs.device, dtype=imgs.dtype)
        shift_x *= 2.0 / w_pad
        shift_y *= 2.0 / h_pad
        shift = torch.cat([shift_x, shift_y], dim=-1)
        grid = base_grid + shift
        
        # apply the grid to the input tensor to perform cropping
        padded_imgs_after_crop = F.grid_sample(imgs, grid)

        return padded_imgs_after_crop

    def forward(self, imgs):
        strength = torch.randint(0, self.highest_pad_strength+1, (1,)).item()
        
        num_cameras = 3
        n, c, h, w = imgs.size()
        imgs = imgs.reshape(n,c,h,num_cameras,-1)
        imgs = imgs.permute(0,1,3,2,4)
        imgs = imgs.reshape(n,c*num_cameras,h,-1)
        n,_c,h,w = imgs.size()

        # _, _, h, w = imgs.shape
        pad_x = torch.randint(0, strength+1, (1,)).item()
        pad_y = strength - pad_x
        # [x+2*pad_x, y+2*pad_y]
        padded_imgs_before_crop = F.pad(imgs, (pad_x, pad_x, pad_y, pad_y))
        # print('padded_imgs_before_crop', padded_imgs_before_crop.shape)
        # [x+pad_x, y+pad_y]
        padded_imgs_after_crop = self.crop(padded_imgs_before_crop, pad_x, pad_y)
        # print('padded_imgs_after_crop', padded_imgs_after_crop.shape)
        # print('######################')

        resize = T.Resize(size=(h, w))
        resize_imgs = resize(padded_imgs_after_crop)

        resize_imgs = resize_imgs.reshape(n,c,num_cameras,h,-1)
        resize_imgs = resize_imgs.permute(0,1,3,2,4)
        resize_imgs = resize_imgs.reshape(n,c,h,-1)

        return resize_imgs

class NormalizeImg(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, x):
		return x/255.

class CenterCrop(nn.Module):
	def __init__(self, size):
		super().__init__()
		assert size in {84, 100}, f'unexpected size: {size}'
		self.size = size

	def forward(self, x):
		assert x.ndim == 4, 'input must be a 4D tensor'
		if x.size(2) == self.size and x.size(3) == self.size:
			return x
		assert x.size(3) == 100, f'unexpected size: {x.size(3)}'
		if self.size == 84:
			p = 8
		return x[:, :, p:-p, p:-p]

def _get_out_shape(in_shape, layers):
	x = torch.randn(*in_shape).unsqueeze(0)
	return layers(x).squeeze(0).shape[0]

def weight_init(m):
	"""Custom weight init for Conv2D and Linear layers"""
	if isinstance(m, nn.Linear):
		nn.init.orthogonal_(m.weight.data)
		if hasattr(m.bias, 'data'):
			m.bias.data.fill_(0.0)
	elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
		# delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
		assert m.weight.size(2) == m.weight.size(3)
		m.weight.data.fill_(0.0)
		if hasattr(m.bias, 'data'):
			m.bias.data.fill_(0.0)
		mid = m.weight.size(2) // 2
		gain = nn.init.calculate_gain('relu')
		nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)

class Flatten(nn.Module):
	def __init__(self):
		super().__init__()
		
	def forward(self, x):
		return x.view(x.size(0), -1)

class Encoder_orginial(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        #self.repr_dim = 32 * 35 * 35
        #self.repr_dim = 227360
        self.repr_dim = 133280
        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())

        self.apply(agent.cycaug_utils.weight_init)
    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h
    



class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_shape[0]))

        self.apply(agent.cycaug_utils.weight_init)

    def forward(self, obs, std):
        h = self.trunk(obs)

        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = agent.cycaug_utils.TruncatedNormal(mu, std)
        return dist


class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.apply(agent.cycaug_utils.weight_init)

    def forward(self, obs, action):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2
    
class CycAugAgent:
    def __init__(self, obs_shape, action_shape, device, lr, feature_dim, hidden_dim, critic_target_tau, num_expl_steps, update_every_steps, stddev_schedule, stddev_clip, use_tb):
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip

        self.discount = 0.99

        # models
        self.encoder = Encoder_orginial(obs_shape).to(device)
        self.actor = Actor(self.encoder.repr_dim, action_shape, feature_dim,
                           hidden_dim).to(device)

        self.critic = Critic(self.encoder.repr_dim, action_shape, feature_dim,
                             hidden_dim).to(device)
        self.critic_target = Critic(self.encoder.repr_dim, action_shape,
                                    feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # data augmentation
        self.aug_padcrop = RandomShiftsAug(pad=4)
        self.aug_padresize = PadResizePlus(highest_pad_strength=16)

        self.train()
        self.critic_target.train()      
    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def select_action(self, obs,step):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            obs = self.encoder(obs)
            stddev = agent.cycaug_utils.schedule(self.stddev_schedule, step)
            dist = self.actor(obs,stddev)
            action = dist.mean
            return action.cpu().data.numpy().flatten()

    def sample_action(self, obs,step):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            obs = self.encoder(obs)
            stddev = agent.cycaug_utils.schedule(self.stddev_schedule, step)
            dist = self.actor(obs,stddev)

            action = dist.sample(clip=None)
            return action.cpu().data.numpy().flatten()

    def update_critic(self, obs, action, reward, not_done, next_obs, step,L):
        metrics = dict()

        with torch.no_grad():
            stddev = agent.cycaug_utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (not_done * self.discount * target_V)

        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)
        L.log('train_critic/loss', critic_loss, step)
        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        # optimize encoder and critic
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        return metrics

    def update_actor(self, obs, step,L):
        metrics = dict()

        stddev = agent.cycaug_utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()
        L.log('train_actor/loss', actor_loss, step)
        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        return metrics
    
    def update(self, replay_buffer, L, step):
        metrics = dict()
        obs, action, _, reward, next_obs, not_done = replay_buffer.sample()
        L.log('train/batch_reward', reward.mean(), step)

        # augmentation
        if (step // 20000) % 2 == 0:
            augmentation = self.aug_padcrop
        else:
            augmentation = self.aug_padresize      


        obs = augmentation(obs.float())
        next_obs = augmentation(next_obs.float())

        # encode
        obs = self.encoder(obs)
        with torch.no_grad():
            next_obs = self.encoder(next_obs)

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(obs, action, reward, not_done, next_obs, step,L))

        # update actor
        metrics.update(self.update_actor(obs.detach(), step,L))

        # update critic target
        agent.cycaug_utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics
    
    def save(self, model_dir, step):
        torch.save(
            self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.encoder.state_dict(), '%s/encoder_%s.pt' % (model_dir, step)
        )

    def load(self, model_dir, step):
        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, step))
        )
        self.critic.load_state_dict(
            torch.load('%s/critic_%s.pt' % (model_dir, step))
        )
        self.encoder.load_state_dict(
            torch.load('%s/encoder_%s.pt' % (model_dir, step))
        )



  