# -*- coding: utf-8 -*-
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import DataGenerator
from model.actor import Actor
from model.critic import Critic
from model.encoder import Encoder
from model.glimpse import Glimpse
from model.config import get_config, print_config
from itertools import chain
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def initital_network_weights(element):
    for k,v in element.named_parameters():
        if "weight" in k:
            torch.nn.init.xavier_normal_(v.data, gain=torch.nn.init.calculate_gain('linear'))
        if "bias" in k:
            torch.nn.init.constant_(v.data, 0)
        if k=="v" or k=="v_g":
            torch.nn.init.normal_(v.data,0,.002)


def main():
    # Get running configuration
    config, _ = get_config()
    print_config()
    
    # Build tensorflow graph from config
    print("Building graph...")
    actor = Actor(config).to(device)
    critic = Critic(config).to(device)
    actor.apply(initital_network_weights)
    critic.apply(initital_network_weights)
    # Training config (actor)
    lr1_start = config.lr1_start  # initial learning rate
    lr1_decay_rate = config.lr1_decay_rate  # learning rate decay rate
    lr1_decay_step = config.lr1_decay_step  # learning rate decay step

    # Training config (critic)
    lr2_start = config.lr1_start  # initial learning rate
    lr2_decay_rate = config.lr1_decay_rate  # learning rate decay rate
    lr2_decay_step = config.lr1_decay_step  # learning rate decay step

    opt1 = torch.optim.Adam(actor.parameters(),lr=lr1_start)
    opt2 = torch.optim.Adam(critic.parameters(),lr=lr2_start)
    scheduler1 = torch.optim.lr_scheduler.StepLR(opt1, step_size=lr1_decay_step, gamma=lr1_decay_rate)
    scheduler2 = torch.optim.lr_scheduler.StepLR(opt2, step_size=lr2_decay_step, gamma=lr2_decay_rate)
    rewards = []
    print("Build Data...")
    result_pos_list = []
    training_set = DataGenerator(config)
    
    print("Starting Train...")
    if config.training_mode:
        if config.restore_model is True:
            actor.load_state_dict(torch.load(config.restore_from))
            print("Model restored.")
        print("Starting training...")
        for i in tqdm(range(config.iteration)):
            input_batch =  torch.FloatTensor(training_set.train_batch()).to(device)
            opt1.zero_grad()
            positions, log_softmax = actor(input_batch)
            predictions = critic(input_batch)
            reward = actor.get_reward(input_batch,positions)
            loss1 = actor.loss1(reward,predictions,log_softmax)
            loss1.backward()
            opt1.step()
            opt2.zero_grad()
            predictions = critic(input_batch)
            loss2 = actor.loss2(reward,predictions)
            loss2.backward()
            opt2.step()
            scheduler1.step()
            scheduler2.step()
            rewards.append(torch.mean(reward).item())
            if i % 100 == 0 and i!=0:
                print("after " + str(i) + " rounds training, Travel Distance is: " + str(rewards[-1]))
            # Save the variables to disk
            if i % 1000 == 0 and i != 0:
                torch.save(actor.state_dict(),config.save_to)
                print("Model saved in file: %s" % config.save_to)
        print("Training COMPLETED !")
        torch.save(actor.state_dict(),config.save_to)
        print("Model saved in file: %s" % config.save_to)
    else:
       T1 = time.time()
       # Get test data
       input_batch = training_set.train_batch()
       np.save(open("city.pth","wb"),input_batch)
       input_batch = torch.FloatTensor(input_batch).to(device)
    #    el = []
    #    for i in range(input_batch.size(1)):
    #        el.append([])
    #        for j in range(input_batch.size(1)):
    #         a = input_batch[0][i]
    #         b = input_batch[0][j]
    #         delta_x2 = torch.square(a[0] - b[0])
    #         delta_y2 = torch.square(a[0] - b[0])
    #         inter_city_distances = torch.sqrt(delta_x2 + delta_y2)  # sqrt(delta_x**2 + delta_y**2)
    #         el[i].append(round(float(inter_city_distances.item()),2))
    #    print(np.array(el))
       actor.load_state_dict(torch.load(config.restore_from))
       with torch.no_grad():
            positions, _ = actor(input_batch)
       reward = actor.get_reward(input_batch,positions)
       print(reward)
       length = actor.get_everylong(input_batch,positions)
       length = length[0]
       positions = positions.detach().cpu().numpy()
       city = input_batch[0].detach().cpu().numpy()
       position = positions[0]
       result_pos_list = city[position, :]
       T2 = time.time()
       print(T2-T1)
    # 解决中文显示问题
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

    if config.training_mode:
        plt.plot(list(range(len(rewards))), rewards, c='blue')
        plt.title("效果曲线")
        plt.xlabel('轮数')
        plt.ylabel('路径长度')
        plt.legend()
        plt.show()
    else:
        plt.plot(result_pos_list[:, 0], result_pos_list[:, 1], 'o-b')
        rplx = (result_pos_list[1:, 0] + result_pos_list[:-1, 0])/2
        rply = (result_pos_list[1:, 1] + result_pos_list[:-1, 1])/2
        for x, y, s in zip(rplx,rply,length):
            plt.text(x,y,int(s.item()))
        plt.title(f"路线{int(reward[0].item())}")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()
