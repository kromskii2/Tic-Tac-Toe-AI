import pygame
import numpy as np
import torch
import random
import torch.nn as nn
import torch.optim as optim

# Параметры игры и обучения
BOARD_SIZE = 10  # Изменили размер доски на 10x10
WIN_LENGTH = 5  # Нужно собрать 5 в ряд для победы
WIN_REWARD = 1
DRAW_REWARD = 0.5
LOSE_REWARD = -1
ALPHA = 0.1
GAMMA = 0.9
CELL_SIZE = 60  # Размер клетки на экране (уменьшен для 10x10)
SCREEN_SIZE = BOARD_SIZE * CELL_SIZE

# Инициализация Pygame
pygame.init()
screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
pygame.display.set_caption("Tic-Tac-Toe AI")

# Цвета
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# Нейронная сеть
class TicTacToeNet(nn.Module):
    def __init__(self):
        super(TicTacToeNet, self).__init__()
        self.fc1 = nn.Linear(BOARD_SIZE * BOARD_SIZE, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, BOARD_SIZE * BOARD_SIZE)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

model = TicTacToeNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

def get_empty_cells(board):
    return [(i, j) for i in range(BOARD_SIZE) for j in range(BOARD_SIZE) if board[i][j] == 0]

def check_winner(board, player):
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if board[row][col] == player:
                if check_direction(board, player, row, col, 1, 0) or \
                   check_direction(board, player, row, col, 0, 1) or \
                   check_direction(board, player, row, col, 1, 1) or \
                   check_direction(board, player, row, col, 1, -1):
                    return True
    return False

def check_direction(board, player, row, col, dr, dc):
    count = 0
    for _ in range(WIN_LENGTH):
        if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE and board[row][col] == player:
            count += 1
            row += dr
            col += dc
        else:
            break
    return count == WIN_LENGTH

def train_step(state, action, reward, next_state, done):
    state_tensor = torch.FloatTensor(state).view(1, -1)
    next_state_tensor = torch.FloatTensor(next_state).view(1, -1)
    q_values = model(state_tensor)
    q_next = model(next_state_tensor).detach()
    target = q_values.clone()
    target[0][action] = reward + GAMMA * torch.max(q_next) if not done else reward
    optimizer.zero_grad()
    loss = loss_fn(q_values, target)
    loss.backward()
    optimizer.step()

# Функция для отрисовки игрового поля
def draw_board(board):
    screen.fill(WHITE)
    for i in range(1, BOARD_SIZE):
        pygame.draw.line(screen, BLACK, (0, i * CELL_SIZE), (SCREEN_SIZE, i * CELL_SIZE), 2)
        pygame.draw.line(screen, BLACK, (i * CELL_SIZE, 0), (i * CELL_SIZE, SCREEN_SIZE), 2)
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if board[row][col] == 1:
                pygame.draw.circle(screen, BLUE, (col * CELL_SIZE + CELL_SIZE // 2, row * CELL_SIZE + CELL_SIZE // 2), CELL_SIZE // 3, 2)
            elif board[row][col] == 2:
                pygame.draw.line(screen, RED, (col * CELL_SIZE + 20, row * CELL_SIZE + 20),
                                 (col * CELL_SIZE + CELL_SIZE - 20, row * CELL_SIZE + CELL_SIZE - 20), 2)
                pygame.draw.line(screen, RED, (col * CELL_SIZE + CELL_SIZE - 20, row * CELL_SIZE + 20),
                                 (col * CELL_SIZE + 20, row * CELL_SIZE + CELL_SIZE - 20), 2)
    pygame.display.flip()

# Игра против компьютера с визуализацией
def play_game():
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=int)
    done = False
    player_turn = 1

    while not done:
        draw_board(board)
        pygame.time.delay(500)

        if player_turn == 1:
            state = board.flatten()
            state_tensor = torch.FloatTensor(state).view(1, -1)
            q_values = model(state_tensor)
            action = torch.argmax(q_values).item()
            row, col = divmod(action, BOARD_SIZE)
            if board[row][col] == 0:
                board[row][col] = 1
                reward = WIN_REWARD if check_winner(board, 1) else 0
                done = reward != 0 or not get_empty_cells(board)
                player_turn = 2
            else:
                reward = LOSE_REWARD
                done = True
        else:
            empty_cells = get_empty_cells(board)
            row, col = random.choice(empty_cells)
            board[row][col] = 2
            reward = LOSE_REWARD if check_winner(board, 2) else 0
            done = reward != 0 or not get_empty_cells(board)
            player_turn = 1

        next_state = board.flatten()
        train_step(state, row * BOARD_SIZE + col, reward, next_state, done)
    draw_board(board)
    pygame.time.delay(1000)

# Цикл обучения и игры
for episode in range(1000):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
    play_game()
    if episode % 100 == 0:
        print(f"Episode {episode} complete")
print("Модель обучена.")
