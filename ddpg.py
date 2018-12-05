# encoding: utf-8
import sys

import torch
import math
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import torch.nn.functional as F
import my_common as mc
from spp_layer import spatial_pyramid_pool


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


spp_num_outputs = [1, 2, 4]


class ActorRnn(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_layer, n_class):
        super(ActorRnn, self).__init__()
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(in_dim=in_dim, hidden_dim=hidden_dim, n_layer=n_layer, batch_first=True)

    def forward(self, C):
        out, _ = self.lstm(C)
        # res_Cbx = states
        # res_Cbx = res_Cbx.squeeze()

        # Decode hidden state of last time step
        out = self.fc(out[:, -1, :])

        return out


class Actor(nn.Module):
    # hidden_size代表全连接中的隐层基本数量
    # action代表b
    # num_outputs代表Actor网络的输出维度
    # num_intputs代表Actor网络的输入维度

    # -------------------- 如果使用resnet 则使用下列参数 -------------------------
    # spp_num_outputs代表数据使用spp时的分割字典数组
    # spp_data_width = data_width代表数据从一位向量被转化为二维向量时的第一个维度数
    # fc_num_inputs代表在spp下计算得到的金字塔输出维度，也是fc的输入维度，取代num_intputs
    def __init__(self, hidden_size, action, num_inputs, num_output, spp_num_outputs=[1, 2, 4], data_width=8):
        super(Actor, self).__init__()
        self.action = action
        self.num_inputs = num_inputs
        self.num_outputs = num_output

        self.spp_data_width = data_width
        self.spp_num_outputs = spp_num_outputs
        self.fc_num_inputs = sum([i * i for i in self.spp_num_outputs])

        '''
        #设计残差模块用于Cb-x的向量输入
        self.Cbx_res_block1 = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=1, padding=0),
            nn.BatchNorm2d(128),
            nn.RReLU(0.66, 0.99),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.RReLU(0.33, 0.66),
            nn.Conv2d(128, 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1)
        )

        self.Cbx_res_block2 = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=1, padding=0),
            nn.BatchNorm2d(128),
            nn.RReLU(0.66, 0.99),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.RReLU(0.33, 0.66),
            nn.Conv2d(128, 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1)
        )
        '''

        # 第一个全连接网络，层上归一化，随机relu
        # self.afc1 = nn.Linear(self.fc_num_inputs, hidden_size)
        self.afc1 = nn.Linear(self.num_inputs, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.relu1 = nn.RReLU(0.01, 0.33)

        # 第二个全连接网络，层上归一化，随机relu
        self.afc2 = nn.Linear(hidden_size, hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.relu2 = nn.RReLU(0.33, 0.66)

        # 第三个全连接网络，层上归一化，随机relu
        self.afc3 = nn.Linear(hidden_size, hidden_size)
        self.ln3 = nn.LayerNorm(hidden_size)
        self.relu3 = nn.RReLU(0.66, 0.99)

        # 第四个全连接网络，层上归一化，随机relu
        self.afc4 = nn.Linear(hidden_size, hidden_size // 2)
        self.ln4 = nn.LayerNorm(hidden_size // 2)
        self.drop = nn.Dropout()

        # 第五层全连接，准备进行激活
        self.mu = nn.Linear(hidden_size // 2, self.num_outputs)
        self.mu.weight.data.mul_(0.1)
        self.mu.bias.data.mul_(0.1)

    def forward(self, states):
        res_Cbx = states
        '''
        res_Cbx = res_Cbx.unsqueeze(2)
        res_Cbx = res_Cbx.unsqueeze(3)
        res_Cbx = res_Cbx.view(res_Cbx.size(0), 1, self.spp_data_width, res_Cbx.size(1) // self.spp_data_width)

        x_Cbx = self.Cbx_res_block1(res_Cbx)
        x_Cbx += res_Cbx
        res_Cbx = x_Cbx
        x_Cbx = self.Cbx_res_block2(res_Cbx)
        x_Cbx += res_Cbx

        x_Cbx = spatial_pyramid_pool(x_Cbx, x_Cbx.size(0), [x_Cbx.size(2), x_Cbx.size(3)], self.spp_num_outputs)
        x = x_Cbx.squeeze(1)
        '''
        res_Cbx = res_Cbx.squeeze()

        x = self.afc1(res_Cbx)
        x = self.ln1(x)
        x = self.relu1(x)

        x = self.afc2(x)
        x = self.ln2(x)
        x = self.relu2(x)

        x = self.afc3(x)
        x = self.ln3(x)
        x = self.relu3(x)

        x = self.afc4(x)
        x = self.ln4(x)
        x = self.drop(x)

        mu = torch.sigmoid(self.mu(x))
        return mu


class Critic(nn.Module):
    # 与Actor网络几乎一致，只是在前两层的网络中
    def __init__(self, hidden_size, action, num_inputs, spp_num_outputs, data_width=8):
        super(Critic, self).__init__()
        self.action = action
        self.num_outputs = self.action.shape[0]
        # self.num_outputs=1
        self.num_inputs = num_inputs

        self.spp_data_width = data_width
        self.spp_num_outputs = spp_num_outputs
        self.fc_num_inputs = sum([i * i for i in self.spp_num_outputs])

        '''
        self.Cbx_res_block1 = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=1, padding=0),
            nn.BatchNorm2d(128),
            nn.RReLU(0.66, 0.99),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.RReLU(0.33, 0.66),
            nn.Conv2d(128, 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1)
        )

        self.Cbx_res_block2 = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=1, padding=0),
            nn.BatchNorm2d(128),
            nn.RReLU(0.66, 0.99),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.RReLU(0.33, 0.66),
            nn.Conv2d(128, 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1)
        )
        '''

        self.cfc11 = nn.Linear(self.num_outputs, hidden_size)
        self.ln11 = nn.LayerNorm(hidden_size)
        self.relu11 = nn.RReLU(0.01, 0.33)

        self.cfc12 = nn.Linear(self.num_inputs, hidden_size)
        self.ln12 = nn.LayerNorm(hidden_size)
        self.relu12 = nn.RReLU(0.01, 0.33)

        self.cfc2 = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.ln2 = nn.LayerNorm(hidden_size * 2)
        self.relu2 = nn.RReLU(0.33, 0.66)

        self.cfc3 = nn.Linear(hidden_size * 2, hidden_size)
        self.ln3 = nn.LayerNorm(hidden_size)
        self.relu3 = nn.RReLU(0.66, 0.99)

        self.cfc4 = nn.Linear(hidden_size, hidden_size // 2)
        self.ln4 = nn.LayerNorm(hidden_size // 2)
        self.drop = nn.Dropout()

        self.V = nn.Linear(hidden_size // 2, 1)
        self.V.weight.data.mul_(0.1)
        self.V.bias.data.mul_(0.1)

    def forward(self, states, actions):
        states = states.squeeze()
        actions = actions.squeeze()

        x_b = actions
        x_b = x_b.reshape(-1, self.num_outputs)

        x_b = self.cfc11(x_b)
        x_b = self.ln11(x_b)
        x_b = self.relu11(x_b)

        x_Cbx = states.squeeze()
        x_Cbx = self.cfc12(x_Cbx)
        x_Cbx = self.ln12(x_Cbx)
        x_Cbx = self.relu12(x_Cbx)

        x = torch.cat([x_b, x_Cbx], -1)
        # x = x.squeeze(1)

        x = self.cfc2(x)
        x = self.ln2(x)
        x = self.relu2(x)

        x = self.cfc3(x)
        x = self.ln3(x)
        x = self.relu3(x)

        x = self.cfc4(x)
        x = self.ln4(x)
        x = self.drop(x)

        V = self.V(x)
        return V


class DDPG(object):
    def __init__(self, gamma, tau, hidden_size, action, num_inputs, actor_size):

        self.action = action
        self.hd = action.shape[0]

        if torch.cuda.is_available():
            self.actor = Actor(hidden_size, self.action, num_inputs, actor_size).cuda()
            self.actor_target = Actor(hidden_size, self.action, num_inputs, actor_size).cuda()
            self.actor_perturbed = Actor(hidden_size, self.action, num_inputs, actor_size).cuda()  # 探索阶段
        else:
            self.actor = Actor(hidden_size, self.action, num_inputs, actor_size)
            self.actor_target = Actor(hidden_size, self.action, num_inputs, actor_size)
            self.actor_perturbed = Actor(hidden_size, self.action, num_inputs, actor_size)  # 探索阶段

        self.actor_optim = Adam(self.actor.parameters(), lr=0.05)  # actor的学习阶段，随机梯度下降，注意学习率

        if torch.cuda.is_available():
            self.critic = Critic(hidden_size, self.action, num_inputs, spp_num_outputs).cuda()
            self.critic_target = Critic(hidden_size, self.action, num_inputs, spp_num_outputs).cuda()
        else:
            self.critic = Critic(hidden_size, self.action, num_inputs, spp_num_outputs)
            self.critic_target = Critic(hidden_size, self.action, num_inputs, spp_num_outputs)

        self.critic_optim = Adam(self.critic.parameters(), lr=0.1)  # critic的学习阶段，随机梯度下降，注意学习率

        self.gamma = gamma
        self.tau = tau

        hard_update(self.actor_target, self.actor)  # Make sure target is with the same weight
        hard_update(self.critic_target, self.critic)

    # 探索阶段的行为确定，对mu的扰动
    def select_action(self, state, action_noise=None, param_noise=None):
        self.actor.eval()
        if param_noise is not None:
            if (torch.cuda.is_available()):
                mu = self.actor_perturbed((Variable(state).cuda()))
            else:
                mu = self.actor_perturbed((Variable(state)))
        else:
            if (torch.cuda.is_available()):
                mu = self.actor((Variable(state).cuda()))
            else:
                mu = self.actor((Variable(state)))

        self.actor.train()
        mu = mu.data

        # 扰动的出发条件
        if action_noise is not None:
            '''
            if torch.cuda.is_available():
                mu += torch.Tensor(action_noise.noise()).cuda()
            else:
                mu += torch.Tensor(action_noise.noise())
            '''
            # 根据随机的效果取一半的数据更改符号
            mu = mc.random_change_sgn(0, self.hd - 1, self.hd // 22, mu)

        return mu

    def update_parameters(self, batch):
        if torch.cuda.is_available():
            state_batch = Variable(torch.cat(batch.state)).cuda()
            action_batch = Variable(torch.cat(batch.action)).cuda()
            reward_batch = Variable(torch.cat(batch.reward)).cuda()
            mask_batch = Variable(torch.cat(batch.mask)).cuda()
            next_state_batch = Variable(torch.cat(batch.next_state)).cuda()
        else:
            state_batch = Variable(torch.cat(batch.state))
            action_batch = Variable(torch.cat(batch.action))
            reward_batch = Variable(torch.cat(batch.reward))
            mask_batch = Variable(torch.cat(batch.mask))
            next_state_batch = Variable(torch.cat(batch.next_state))

        next_action_batch = self.actor_target(next_state_batch)
        next_state_action_values = self.critic_target(next_state_batch, next_action_batch)

        reward_batch = reward_batch.unsqueeze(1)  # 纵向排列
        mask_batch = mask_batch.unsqueeze(1)
        expected_state_action_batch = reward_batch + (
                self.gamma * mask_batch * next_state_action_values)  # mask_batch中是true和false，python会自动将其转换为1和0

        # critic网络重置梯度为0
        self.critic_optim.zero_grad()

        state_action_batch = self.critic((state_batch), (action_batch))

        # 以均方差loss计算value值并反馈
        value_loss = F.mse_loss(state_action_batch, expected_state_action_batch)
        value_loss.backward()
        self.critic_optim.step()  # 完成参数更新

        # critic网络重置梯度为0
        self.actor_optim.zero_grad()

        # 对于Actor网络，使用critic的计算值作为反馈的loss
        policy_loss = -self.critic((state_batch), self.actor((state_batch)))
        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()

        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return value_loss.item(), policy_loss.item()

    def perturb_actor_parameters(self, param_noise):
        """Apply parameter noise to actor model, for exploration"""
        hard_update(self.actor_perturbed, self.actor)
        params = self.actor_perturbed.state_dict()  # 取出所有的params，并且把给的noise加入到param中
        for name in params:
            if 'ln' in name:
                pass
            param = params[name]
            param += torch.randn(param.shape) * param_noise.current_stddev

    def save_model(self, env_name, suffix="", actor_path=None, critic_path=None):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        if actor_path is None:
            actor_path = "models/ddpg_actor_{}_{}".format(env_name, suffix)
        if critic_path is None:
            critic_path = "models/ddpg_critic_{}_{}".format(env_name, suffix)
        print('Saving models to {} and {}'.format(actor_path, critic_path))
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    def load_model(self, actor_path, critic_path):
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.actor.load_state_dict(torch.load(actor_path))
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))
