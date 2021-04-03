import argparse
import os
import shutil
from random import random, randint, sample
import pygame
import button
import matplotlib
import matplotlib.backends.backend_agg as agg
import pylab
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from deep_q_learning import DeepQNetwork
from tetris_cheater import Tetris as Cheater
from tetris_fair import Tetris as Fair
from collections import deque
from train_el_tetris import Tetris as El_Tetris
matplotlib.use("Agg")


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Tetris""")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--initial_epsilon", type=float, default=1)
    parser.add_argument("--final_epsilon", type=float, default=5e-4)
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--log_path", type=str, default="tensorboard")
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--num_decay_epochs", type=float, default=2000)
    parser.add_argument("--num_epochs", type=int, default=3000)
    parser.add_argument("--replay_memory_size", type=int, default=30000)

    args = parser.parse_args()
    return args


def train(opt, training_type, number_of_features):
    # Checks if the device has a supported gpu otherwise use cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)

    # Creates a folder for the given log path
    os.makedirs(opt.log_path)
    writer = SummaryWriter(opt.log_path)

    # Set and refresh screen
    screen = pygame.display.set_mode((1400, 700))
    screen.fill((0, 0, 0))

    # Modes
    if training_type == "fair":
        print("Here")
        env = Fair(screen, "train", True)
    elif training_type == "cheater":
        print("Here 2")
        env = Cheater(screen, "train", True)
    elif training_type == "el tetris":
        print("here 3")
        env = El_Tetris(screen, "train", True)
        print("env", env)
        # model = DeepQNetwork(number_of_features).to(device)

        # return

    # model is the neural network
    model = DeepQNetwork(number_of_features).to(device)
    # Optimises the model using the learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    # measures the mean squared error between elements
    criterion = nn.MSELoss()
    # Gets the default states of the environment
    state = env.reset().to(device)

    # Limits amount of moves made
    steps = 0
    max_step = 2000

    font_small = pygame.font.SysFont('Arial', 20)
    clock = pygame.time.Clock()

    # Setups abound queue to the length of the reply memory size
    replay_memory = deque(maxlen=opt.replay_memory_size)

    epoch = 0
    score = []
    return_button = button.Button((61, 97, 128), 575, 625, 200, 50, 'Return')

    pygame.display.flip()
    while epoch < opt.num_epochs:
        next_steps = env.get_next_states()
        # Decides to do exploration or exploitation
        epsilon = opt.final_epsilon + (max(opt.num_decay_epochs - epoch, 0) * (
                opt.initial_epsilon - opt.final_epsilon) / opt.num_decay_epochs)
        u = random()
        random_action = u <= epsilon
        next_actions, next_states = zip(*next_steps.items())
        next_states = torch.stack(next_states).to(device)

        # Evaluates model
        model.eval()

        with torch.no_grad():
            predictions = model(next_states)[:, 0]
        # Trains model
        model.train()
        if random_action:
            index = randint(0, len(next_steps) - 1)
        else:
            index = torch.argmax(predictions).item()

        next_state = next_states[index, :].to(device)
        action = next_actions[index]

        # Gets next steps from environment
        reward, done = env.step(action)
        steps = steps + 1

        replay_memory.append([state, reward, next_state, done])
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.display.quit()
                quit()
            pos = pygame.mouse.get_pos()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if return_button.is_over(pos):
                    writer.close()
                    return True
            if event.type == pygame.MOUSEMOTION:
                if return_button.is_over(pos):
                    return_button.color = (61, 97, 128)
                else:
                    return_button.color = (147, 150, 153)

        area = pygame.Rect(0, 75, 900, 625)
        return_button.draw(screen)
        fps = font_small.render("fps:" + str(int(clock.get_fps())), True, pygame.Color('white'))
        screen.blit(fps, (10, 75))
        clock.tick(200)
        pygame.display.update(area)
        if done or (max_step <= steps):
            final_score = env.score
            final_pieces_placed = env.total_pieces_placed
            final_cleared_lines = env.total_lines_cleared
            state = env.reset().to(device)
            steps = 0
        else:
            state = next_state
            continue
        if len(replay_memory) < opt.replay_memory_size / 10:
            continue
        score.append(final_score)
        epoch += 1
        batch = sample(replay_memory, min(len(replay_memory), opt.batch_size))
        state_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        state_batch = torch.stack(tuple(state for state in state_batch)).to(device)
        reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None]).to(device)
        next_state_batch = torch.stack(tuple(state for state in next_state_batch)).to(device)

        q_values = model(state_batch)
        model.eval()
        with torch.no_grad():
            next_prediction_batch = model(next_state_batch)
        model.train()

        y_batch = torch.cat(
            tuple(reward if done else reward + opt.gamma * prediction for reward, done, prediction in
                  zip(reward_batch, done_batch, next_prediction_batch)))[:, None]

        optimizer.zero_grad()
        loss = criterion(q_values, y_batch)
        loss.backward()
        optimizer.step()
        graph_results(score, opt.num_epochs)

        writer.add_scalar('Train/Score', final_score, epoch - 1)
        writer.add_scalar('Train/Tetrominoes', final_pieces_placed, epoch - 1)
        writer.add_scalar('Train/Cleared lines', final_cleared_lines, epoch - 1)

        if epoch > 0 and epoch % opt.save_interval == 0:
            torch.save(model, "{}/{}_tetris_{}".format(opt.saved_path, training_type, epoch))

    # torch.save(model, "{}/{}_tetris".format(opt.saved_path, training_type))
    writer.close()
    display(screen)


# Draws graph
def graph_results(score, length):
    fig = pylab.figure(figsize=[4, 4], dpi=90)
    ax = fig.gca()
    ax.plot(score)
    ax.set_title("Agents score over {}/{} Iteration".format(length, len(score)))
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Score')

    canvas = agg.FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    raw_data = renderer.tostring_rgb()

    pygame.init()

    screen = pygame.display.get_surface()

    size = canvas.get_width_height()

    surf = pygame.image.fromstring(raw_data, size, "RGB")
    screen.blit(surf, (800, 200))
    area = pygame.Rect(800, 0, 600, 700)
    pygame.display.update(area)
    pylab.close('all')


# Draws notice at the end of training
def display(screen):
    pygame.draw.rect(screen, (71, 73, 74), (1400 / 2 - 200, 200, 400, 300), 0)
    selection_menu_button = button.Button((61, 97, 128), 525, 400, 350, 50, 'Selection Menu')
    draw_text_middle("Training Complete", 40, (255, 255, 255), screen)
    pygame.display.update()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.display.quit()
                quit()
            pos = pygame.mouse.get_pos()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if selection_menu_button.is_over(pos):
                    return False
            if event.type == pygame.MOUSEMOTION:
                if selection_menu_button.is_over(pos):
                    selection_menu_button.color = (61, 97, 128)
                else:
                    selection_menu_button.color = (147, 150, 153)
        selection_menu_button.draw(screen)
        pygame.display.update()


# Draws centred text
def draw_text_middle(text, size, color, screen):
    font = pygame.font.SysFont('Arial', size, bold=True)
    label = font.render(text, 1, color)

    screen.blit(label, (1400 / 2 - (label.get_width() / 2), 250 - label.get_height() / 2))


def main(training_type, number_of_features):
    opt = get_args()
    train(opt, training_type, number_of_features)
    return True


def elTetris():
    print("in new function")
    opt = get_args()
    number_of_features = 4
    screen = pygame.display.set_mode((1400, 700))
    screen.fill((0, 0, 0))
    env = El_Tetris(screen, "train", True)
    print("env", env)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return_button = button.Button((61, 97, 128), 575, 625, 200, 50, 'Return')

    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)

    # Creates a folder for the given log path
    os.makedirs(opt.log_path)
    writer = SummaryWriter(opt.log_path)

    # model is the neural network
    # model = DeepQNetwork(number_of_features).to(device)
    # Optimises the model using the learning rate
    # optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    # measures the mean squared error between elements
    criterion = nn.MSELoss()
    # Gets the default states of the environment
    state = env.reset().to(device)

    # Limits amount of moves made
    steps = 0
    max_step = 2000

    font_small = pygame.font.SysFont('Arial', 20)
    clock = pygame.time.Clock()

    # Setups abound queue to the length of the reply memory size
    replay_memory = deque(maxlen=opt.replay_memory_size)

    epoch = 0
    score = []
    # return_button = button.Button((61, 97, 128), 575, 625, 200, 50, 'Return')

    pygame.display.flip()
    while epoch < opt.num_epochs:
        next_steps = env.get_next_states()
        # movesAndWeights = env.get_next_states()
        # print("movesAndWeights", movesAndWeights)
        # Decides to do exploration or exploitation
        # epsilon = opt.final_epsilon + (max(opt.num_decay_epochs - epoch, 0) * (
        #         opt.initial_epsilon - opt.final_epsilon) / opt.num_decay_epochs)
        # u = random()
        # random_action = u <= epsilon
        next_actions, next_states = zip(*next_steps.items())
        next_states = torch.stack(next_states).to(device)

        # print("next_actions", next_actions)
        # print("next_states", next_states)

        weightTotals = []
        previousGrid = []
        for weight in next_states:
            # feature0 = -4.500158825082766 * weight[0]
            # feature1 = 3.4181268101392694 * weight[1]
            # feature2 = -3.2178882868487753 * weight[2]
            # feature3 = -9.348695305445199 * weight[3]
            # feature4 = -7.899265427351652 * weight[4]
            # feature5 = -3.3855972247263626 * weight[5]
            feature0 = -45* weight[0]
            feature1 = 34 * weight[1]
            feature2 = -32 * weight[2]
            feature3 = -98 * weight[3]
            feature4 = -79 * weight[4]
            feature5 = -34 * weight[5]
            # previousGrid.append(print(weight[6]))
            weightTotals.append(feature0 + feature1 + feature2 + feature3 + feature4 + feature5)

        maxIndex = -9999999
        maxValue = -9999999

        for i in range(0, len(weightTotals)):
            # print("weightTotals", weightTotals[i].numpy())
            # print("maxValue", maxValue)
            # print("if", weightTotals[i].numpy() > maxValue)
            if weightTotals[i].numpy() > maxValue:
                # print("HERERERERRE")
                maxIndex = i
                maxValue = weightTotals[i].numpy()


        # Evaluates model
        # model.eval()

        # with torch.no_grad():
        #     predictions = model(next_states)[:, 0]
        # Trains model
        # model.train()
        # if random_action:
        #     index = randint(0, len(next_steps) - 1)
        # else:
        #     index = torch.argmax(predictions).item()

        # next_state = next_states[index, :].to(device)
        # action = next_actions[index]
        next_state = next_states[maxIndex, :].to(device)
        action = next_actions[maxIndex]
        # print("action", action)


        # Gets next steps from environment
        reward, done = env.step(action)
        # print("reward", reward)
        # print("done", done)
        steps = steps + 1

        replay_memory.append([state, reward, next_state, done])
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.display.quit()
                quit()
            pos = pygame.mouse.get_pos()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if return_button.is_over(pos):
                    writer.close()
                    return True
            if event.type == pygame.MOUSEMOTION:
                if return_button.is_over(pos):
                    return_button.color = (61, 97, 128)
                else:
                    return_button.color = (147, 150, 153)

        area = pygame.Rect(0, 75, 900, 625)
        return_button.draw(screen)
        fps = font_small.render("fps:" + str(int(clock.get_fps())), True, pygame.Color('white'))
        screen.blit(fps, (10, 75))
        clock.tick(100)
        pygame.display.update(area)
        if done or (max_step <= steps):
            final_score = env.score
            final_pieces_placed = env.total_pieces_placed
            final_cleared_lines = env.total_lines_cleared
            state = env.reset().to(device)
            steps = 0
        else:
            state = next_state
            continue
        if len(replay_memory) < opt.replay_memory_size / 10:
            continue
        score.append(final_score)
        epoch += 1
        batch = sample(replay_memory, min(len(replay_memory), opt.batch_size))
        state_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        state_batch = torch.stack(tuple(state for state in state_batch)).to(device)
        reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None]).to(device)
        next_state_batch = torch.stack(tuple(state for state in next_state_batch)).to(device)

        # q_values = model(state_batch)
        # model.eval()
        # with torch.no_grad():
        #     next_prediction_batch = model(next_state_batch)
        # model.train()

        y_batch = torch.cat(
            tuple(reward if done else reward + opt.gamma * prediction for reward, done, prediction in
                  zip(reward_batch, done_batch, next_prediction_batch)))[:, None]

        optimizer.zero_grad()
        loss = criterion(q_values, y_batch)
        loss.backward()
        optimizer.step()
        graph_results(score, opt.num_epochs)

        writer.add_scalar('Train/Score', final_score, epoch - 1)
        writer.add_scalar('Train/Tetrominoes', final_pieces_placed, epoch - 1)
        writer.add_scalar('Train/Cleared lines', final_cleared_lines, epoch - 1)

        # if epoch > 0 and epoch % opt.save_interval == 0:
            # torch.save(model, "{}/{}_tetris_{}".format(opt.saved_path, training_type, epoch))

    # torch.save(model, "{}/{}_tetris".format(opt.saved_path, training_type))
    writer.close()
    display(screen)

    # Set and refresh screen
   
    