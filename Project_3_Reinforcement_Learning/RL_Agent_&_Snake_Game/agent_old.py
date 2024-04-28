import torch
import random
import numpy as np
from collections import deque # to record memory
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot

#check gpu is vaialble or not

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device:', device)




MAX_MEMORY = 100_000 #we can store 1000000 memory in deque
BATCH_SIZE = 1000 # we will take 1000 memory from deque to train our model
LR = 0.001 # learning rate
BLOCK_SIZE  = 20

class Agent:
    def __init__(self):
        #self.epsilon = 0.9  # Initial epsilon value
        #self.epsilon_decay = 0.001  # Amount to decay epsilon in each game
        self.n_games = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate, must be smaller than 1
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(11, 256, 3).to(device)
        #model, trainer
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

       
    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - BLOCK_SIZE, head.y)
        point_r = Point(head.x + BLOCK_SIZE, head.y)
        point_u = Point(head.x, head.y - BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            #Danger straight
            (dir_r and game.is_collision(point_r)) or
            (dir_l and game.is_collision(point_l)) or
            (dir_u and game.is_collision(point_u)) or
            (dir_d and game.is_collision(point_d)),
            #Danger right
            (dir_u and game.is_collision(point_r)) or
            (dir_d and game.is_collision(point_l)) or
            (dir_l and game.is_collision(point_u)) or
            (dir_r and game.is_collision(point_d)),
            #Danger left
            (dir_d and game.is_collision(point_r)) or
            (dir_u and game.is_collision(point_l)) or
            (dir_r and game.is_collision(point_u)) or
            (dir_l and game.is_collision(point_d)),
            #Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            #Food location
            game.food.x < game.head.x, # food left
            game.food.x > game.head.x, # food right
            game.food.y < game.head.y, # food up
            game.food.y > game.head.y # food down
        ]
        return np.array(state, dtype=int)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft is MAX_MEMORY
    def train_long_memory(self):
        if len(self.memory) < BATCH_SIZE:
            #mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
            mini_sample = self.memory  # Use all available transitions for training

        else:
            #mini_sample = self.memory
            mini_sample = random.sample(self.memory, BATCH_SIZE)

            # states, actions, rewards, next_states, dones = zip(*mini_sample)
            # # Convert each element to the desired device (GPU or CPU)
            # states = torch.tensor(states, dtype=torch.float).to(device)
            # next_states = torch.tensor(next_states, dtype=torch.float).to(device)
            # actions = torch.tensor(actions, dtype=torch.long).to(device)
            # rewards = torch.tensor(rewards, dtype=torch.float).to(device)
            # dones = torch.tensor(dones, dtype=torch.bool).to(device)

            states, actions, rewards, next_states, dones = zip(*mini_sample)
            states = np.array(states)  # Convert states to numpy array
            next_states = np.array(next_states)  # Convert next_states to numpy array
            actions = np.array(actions)  # Convert actions to numpy array
            rewards = np.array(rewards, dtype=np.float32)  # Convert rewards to numpy array
            dones = np.array(dones, dtype=np.uint8)  # Convert dones to numpy array
            # Print the device of the model's parameters
            # print("Model parameters device:", next(self.model.parameters()).device)
            # print("Trainer device:", self.trainer.device)
            # Call trainer.train_step with the converted tensors
            self.trainer.train_step(states, actions, rewards, next_states, dones)
    def train_short_memory(self,  state, action, reward, next_state, done):
        #self.trainer.train_step(state, action, reward, next_state, done)
        state = np.array(state)  # Convert state to numpy array
        action = np.array(action)  # Convert action to numpy array
        reward = np.array([reward], dtype=np.float32)  # Convert reward to a numpy array

        self.trainer.train_step(state, action, reward, next_state, done)
    def get_action(self, state):
        # ramdom moves : tradeoff exploration and /exploitation - in deep learning
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else: # get action from Q table
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        return final_move

        # self.epsilon = max(0.1, 100 - self.n_games)  # Ensure epsilon doesn't go below 0.1
        # # Decay epsilon over time
        # # self.epsilon = max(0.1, self.epsilon - self.epsilon_decay)
        
        # move = 0
        # if random.randint(0, 200) < self.epsilon:
        #     # Explore
        #     move = random.randint(0, 2)
        # else:
        #     # Exploit
        #     state0 = torch.tensor(state, dtype=torch.float).to(device)
        #     prediction = self.model(state0)
        #     action = torch.argmax(prediction).item()
        #     move = action

        # # Create a one-hot encoded action vector
        # final_move = [0, 0, 0]
        # final_move[move] = 1

        # return final_move


    def train(self):
        plot_score = []
        plot_mean_score = []
        total_score = 0
        record = 0
        #agent = Agent()
        game = SnakeGameAI()

        while True:
            # get old state
            state_old = agent.get_state(game)

            # get move
            final_move = agent.get_action(state_old)

            # perform move and get new state
            reward, done, score = game.play_step(final_move)
            state_new = agent.get_state(game)

            # train short memory
            agent.train_short_memory(state_old, final_move, reward, state_new, done)

            # remember
            agent.remember(state_old, final_move, reward, state_new, done)

            if done:
                # train long memory, plot result
                game.reset()
                agent.n_games += 1
                agent.train_long_memory()

                if score > record:
                    record = score
                    agent.model.save()

                print('Game', agent.n_games, 'Score', score, 'Record:', record)

                plot_score.append(score)
                total_score += score
                mean_score = total_score / agent.n_games
                plot_mean_score.append(mean_score)
                plot(plot_score, plot_mean_score)

  

    def play_game_with_saved_model(self):
        # Create a new instance of Linear_QNet
        model = Linear_QNet(11, 256, 3)
        # Load the saved model state dictionary into the new model
        model.load_state_dict(torch.load("path_to_saved_model.pth"))
        model.eval()  # Set the model to evaluation mode

        game = SnakeGameAI()
        while True:
            state = agent.get_state(game)
            # Move state to CPU or GPU if needed
            state = torch.tensor(state, dtype=torch.float).to(device)

            # Use the model to get the action
            with torch.no_grad():
                prediction = model(state)
            move = torch.argmax(prediction).item()
            final_move = [0, 0, 0]
            final_move[move] = 1

            reward, game_over, score = game.play_step(final_move)

            if game_over:
                print("Game Over")
                break

    # ... (Other code)

        

    # def save_model(self):
    #     pass
    # def load_model(self):
    #     pass


if __name__ == '__main__':
    agent = Agent()
    agent.train()


