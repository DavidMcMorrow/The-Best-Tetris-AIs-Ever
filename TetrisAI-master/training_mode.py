import argparse
import os
import shutil
from random import random, randint, sample, shuffle
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
from settings import Setting
matplotlib.use("Agg")

settings = Setting()
screen_width = settings.screen_width
screen_height = settings.screen_height
dark_grey = settings.screen_colour
screen_centre = screen_width / 2
button_colour_off = settings.button_colour_off
button_colour_on = settings.button_colour_on
button_width = settings.button_width
button_height = settings.button_height
button_centred = screen_centre - button_width / 2

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
        env = Fair(screen, "train", True)
    else:
        env = Cheater(screen, "train", True)

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

    torch.save(model, "{}/{}_tetris".format(opt.saved_path, training_type))
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
    # opt = get_args()
    # train(opt, training_type, number_of_features)
    runGeneticAlgorithm()
    return True

def choosingNext(stack, chrom):
    outputArray = []
    for line in stack:
        outputArray.append(line[0].numpy()*chrom[0] + line[1].numpy()*chrom[1] + line[2].numpy()*chrom[2] + line[3].numpy()*chrom[3] + line[4].numpy()*chrom[4])
    outputArray = torch.from_numpy(np.array(outputArray))
    return outputArray

def crossBreeding(population, scores):
    # first random selection
    k = randint(0, len(population)-1)
    # selection_ix = int(random()*len(population))
    selection_ix = randint(0, len(population)-1)
    for val in range(len(population)-k):
        if scores[val+k] > scores[selection_ix]:
            selection_ix = val + k
        # for ix in range(len(population)):
            # check if better (e.g. perform a tournament)
    #         if scores[ix] < scores[selection_ix]:
    #             selection_ix = ix
    # else:
    #     for ix in range(randint(k, len(population)-1)):
    #     # for ix in range(len(population)):
    #         # check if better (e.g. perform a tournament)
    #         if scores[ix] < scores[selection_ix]:
    #             selection_ix = ix
    print(selection_ix)
    return population[selection_ix]
# first random selection
	# selection_ix = randint(len(pop))
	# for ix in randint(0, len(pop), k-1):
		# check if better (e.g. perform a tournament)
	# 	if scores[ix] < scores[selection_ix]:
	# 		selection_ix = ix
	# return pop[selection_ix]

def crossFunction(population, scores):
    total = []
    currscores = np.array(scores)
    for times in range(int(len(population)/2)):
        highest = 0
        end = 0
        for item in range(len(currscores)):
            # print(scores[item])
            if currscores[item] > highest:
                end = item
                highest = currscores[item]
        # print(scores)
        # print("Scores", scores[end])
        # print("Pop", population[end])
        total.append(population[end])
        total.append(population[end])
        currscores = np.delete(currscores, end)
    
    shuffle(total)
    # print("Selected", total)
    return total

def crossover(p1, p2, r_cross):
	# children are copies of parents by default
    c1, c2 = p1.copy(), p2.copy()
    # check for recombination
    if random() < r_cross:
        c1, c2 = [], []
        pt = randint(1, len(p1)-2)
        for val in p1[:pt]:
            c1.append(val)
        for val in p2[pt:]:
            c1.append(val)
        for val in p2[:pt]:
            c2.append(val)
        for val in p1[pt:]:
            c2.append(val)
        c1 = np.array(c1)
        c2 = np.array(c2)
        # c1 = np.concatenate(p1[:pt],p2[pt:])
        # c2 = np.concatenate(p2[:pt],p1[pt:])
    return [c1, c2]

def mutation(bitstring, r_mut):
	for i in range(len(bitstring)):
		# check for a mutation
		if random() < r_mut:
			# flip the bit
			bitstring[i] = random() - bitstring[i]

def runGeneticAlgorithm():
    # Checks if the device has a supported gpu otherwise use cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    # if os.path.isdir(opt.log_path):
    #     shutil.rmtree(opt.log_path)

    # # Creates a folder for the given log path
    # os.makedirs(opt.log_path)
    # writer = SummaryWriter(opt.log_path)

    # Set and refresh screen
    # screen = pygame.display.set_mode((1400, 700))
    # screen.fill((0, 0, 0))
    

    screen = pygame.display.set_mode((screen_width, screen_height))
    screen.fill((0, 0, 0))
    env = Fair(screen, "train", True)
    # Modes
    n_pop = 30
    arraySize = 7
    generations = 100
    r_cross = 0.9
    r_mut = 0.01
    population = []
    for pop in range(n_pop):
        population.append(np.random.uniform(low=-3.0, high=3.0, size=(arraySize,)))
    print(population)
    # population = [[-2.84753725,  0.58446161, -0.58790005,  0.43321749, -0.71288113,
    #     0.07884711, -2.43557128]]
    population = np.array(population)
    seed = 0
    for gens in range(generations):

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        else:
            torch.manual_seed(seed)
        seed = seed + 1
        scores = []

        for pop in population:
            clock = pygame.time.Clock()
            area = pygame.Rect(0, 75, 900, 625)
            return_button = button.Button(button_colour_off, 625, 625, 150, 50, 'Return')
            font_small = pygame.font.SysFont('Arial', 20)
            holder = 0
            keepGoing = True
            while keepGoing:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.display.quit()
                        quit()
                    pos = pygame.mouse.get_pos()
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        if return_button.is_over(pos):
                            return True
                    if event.type == pygame.MOUSEMOTION:
                        if return_button.is_over(pos):
                            return_button.color = (61, 97, 128)
                        else:
                            return_button.color = (147, 150, 153)

                
                next_steps = env.get_next_states()
                next_actions, next_states = zip(*next_steps.items())
                next_states = torch.stack(next_states).to(device)

                holder = 0
                
                # saved_path="fair_tetris"
                # model = torch.load(("trained_models/{}".format(saved_path)), map_location=lambda storage, loc: storage).to(device)
                # model.eval()

                # predictions = model(next_states)[:, 0]
                # print(predictions)
                predictions = choosingNext(next_states, pop)
                
                
                index = torch.argmax(predictions).item()
                action = next_actions[index]
                reward, won = env.step(action, holder)

                return_button.draw(screen)
                fps = font_small.render("fps:" + str(int(clock.get_fps())), True, pygame.Color('white'))
                screen.blit(fps, (10, 75))
                clock.tick(200)
                pygame.display.update(area)
                if won or env.total_pieces_placed >= 250000:
                    print(population)
                    print("New Generation: ", gens)
                    print("Final Scores: ", env.score)
                    print("Pieces placed: ", env.total_pieces_placed)
                    print("Lines cleared: ", env.total_lines_cleared)
                    scores.append(env.last_score)
                    
                    env.reset()
                    keepGoing = False
        
        # selected = [crossBreeding(population, scores) for _ in range(n_pop)]
        # selected = crossFunction(population, scores)

        # elite = []
        # time = 0
        # for times in range(2):
        #     highest = 0
        #     end = 0
        #     for item in range(len(scores)):
        #         # print(scores[item])
        #         if scores[item] > highest:
        #             end = item
        #             highest = scores[item]
        #     # print(scores)
        #     # print("Scores", scores[end])
        #     # print("Pop", population[end])
        #     elite.append(population[end-time])
        #     print(scores[end])
        #     print(population[end-time])
        #     scores = np.delete(scores, end)
        #     time += 1
    
        
        # children = list()
        # for i in range(0, n_pop, 2):
        #     # get selected parents in pairs
        #     p1, p2 = selected[i], selected[i+1]
        #     # crossover and mutation
        #     for c in crossover(p1, p2, r_cross):
        #         # mutation
        #         mutation(c, r_mut)
        #         # store for next generation
        #         children.append(c)
        # # children = np.delete(children, len(children)-1)
        # # children = np.delete(children, len(children)-1)
        # np.append(children, elite[0])
        # np.append(children, elite[1])
        # children[len(children)-2] = elite[0]
        # children[len(children)-1] = elite[1]
        # population = children
        print(population)
        print("New Generation: ", gens)
        print("Final Scores: ", scores)
        print("Pieces placed: ", env.total_pieces_placed)
        print("Lines cleared: ", env.total_lines_cleared)
        # print("Final Population", population)
    # print(env.get_state_properties([[(0, 0, 0) for x in range(10)] for x in range(20)]))

    # # model is the neural network
    # model = DeepQNetwork(number_of_features).to(device)
    # # Optimises the model using the learning rate
    # optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    # # measures the mean squared error between elements
    # criterion = nn.MSELoss()
    # # Gets the default states of the environment
    # state = env.reset().to(device)

    # # Limits amount of moves made
    # steps = 0
    # max_step = 2000

    # font_small = pygame.font.SysFont('Arial', 20)
    # clock = pygame.time.Clock()

    # # Setups abound queue to the length of the reply memory size
    # replay_memory = deque(maxlen=opt.replay_memory_size)

    # epoch = 0
    # score = []
    # return_button = button.Button((61, 97, 128), 575, 625, 200, 50, 'Return')

    # pygame.display.flip()
    # while epoch < opt.num_epochs:
    #     next_steps = env.get_next_states()
    #     # Decides to do exploration or exploitation
    #     epsilon = opt.final_epsilon + (max(opt.num_decay_epochs - epoch, 0) * (
    #             opt.initial_epsilon - opt.final_epsilon) / opt.num_decay_epochs)
    #     u = random()
    #     random_action = u <= epsilon
    #     next_actions, next_states = zip(*next_steps.items())
    #     next_states = torch.stack(next_states).to(device)

    #     # Evaluates model
    #     model.eval()

    #     with torch.no_grad():
    #         predictions = model(next_states)[:, 0]
    #     # Trains model
    #     model.train()
    #     if random_action:
    #         index = randint(0, len(next_steps) - 1)
    #     else:
    #         index = torch.argmax(predictions).item()

    #     next_state = next_states[index, :].to(device)
    #     action = next_actions[index]

    #     # Gets next steps from environment
    #     reward, done = env.step(action)
    #     steps = steps + 1

    #     replay_memory.append([state, reward, next_state, done])
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             pygame.display.quit()
    #             quit()
    #         pos = pygame.mouse.get_pos()
    #         if event.type == pygame.MOUSEBUTTONDOWN:
    #             if return_button.is_over(pos):
    #                 writer.close()
    #                 return True
    #         if event.type == pygame.MOUSEMOTION:
    #             if return_button.is_over(pos):
    #                 return_button.color = (61, 97, 128)
    #             else:
    #                 return_button.color = (147, 150, 153)

    #     area = pygame.Rect(0, 75, 900, 625)
    #     return_button.draw(screen)
    #     fps = font_small.render("fps:" + str(int(clock.get_fps())), True, pygame.Color('white'))
    #     screen.blit(fps, (10, 75))
    #     clock.tick(200)
    #     pygame.display.update(area)
    #     if done or (max_step <= steps):
    #         final_score = env.score
    #         final_pieces_placed = env.total_pieces_placed
    #         final_cleared_lines = env.total_lines_cleared
    #         state = env.reset().to(device)
    #         steps = 0
    #     else:
    #         state = next_state
    #         continue
    #     if len(replay_memory) < opt.replay_memory_size / 10:
    #         continue
    #     score.append(final_score)
    #     epoch += 1
    #     batch = sample(replay_memory, min(len(replay_memory), opt.batch_size))
    #     state_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
    #     state_batch = torch.stack(tuple(state for state in state_batch)).to(device)
    #     reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None]).to(device)
    #     next_state_batch = torch.stack(tuple(state for state in next_state_batch)).to(device)

    #     q_values = model(state_batch)
    #     model.eval()
    #     with torch.no_grad():
    #         next_prediction_batch = model(next_state_batch)
    #     model.train()

    #     y_batch = torch.cat(
    #         tuple(reward if done else reward + opt.gamma * prediction for reward, done, prediction in
    #               zip(reward_batch, done_batch, next_prediction_batch)))[:, None]

    #     optimizer.zero_grad()
    #     loss = criterion(q_values, y_batch)
    #     loss.backward()
    #     optimizer.step()
    #     graph_results(score, opt.num_epochs)

    #     writer.add_scalar('Train/Score', final_score, epoch - 1)
    #     writer.add_scalar('Train/Tetrominoes', final_pieces_placed, epoch - 1)
    #     writer.add_scalar('Train/Cleared lines', final_cleared_lines, epoch - 1)

    #     if epoch > 0 and epoch % opt.save_interval == 0:
    #         torch.save(model, "{}/{}_tetris_{}".format(opt.saved_path, training_type, epoch))

    # torch.save(model, "{}/{}_tetris".format(opt.saved_path, training_type))
    # writer.close()
    display(screen)