import pygame
import chess
import os
from pathlib import Path
import random
from collections import defaultdict
import numpy as np

# Pygame başlatma
pygame.init()

# Ekran boyutları
WINDOW_SIZE = 600
SQUARE_SIZE = WINDOW_SIZE // 8

# Renkler
WHITE = (255, 255, 255)
BLACK = (128, 128, 128)
HIGHLIGHT = (255, 255, 0, 50)

# Ekranı oluştur
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption("Pekiştirmeli Öğrenme Satranç")

class ChessEnvironment:
    def __init__(self):
        self.board = chess.Board()
        # Merkez kareler için değer matrisi
        self.center_weights = np.array([
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            [0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.1],
            [0.1, 0.2, 0.3, 0.3, 0.3, 0.3, 0.2, 0.1],
            [0.1, 0.2, 0.3, 0.4, 0.4, 0.3, 0.2, 0.1],
            [0.1, 0.2, 0.3, 0.4, 0.4, 0.3, 0.2, 0.1],
            [0.1, 0.2, 0.3, 0.3, 0.3, 0.3, 0.2, 0.1],
            [0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.1],
            [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        ])
        self.reset()
    
    def reset(self):
        self.board.reset()
        return self._get_state()
    
    def _get_state(self):
        return self.board.fen()
    
    def _calculate_piece_position_value(self, square, piece_type, color):
        file_idx = chess.square_file(square)
        rank_idx = chess.square_rank(square)
        if color == chess.BLACK:
            rank_idx = 7 - rank_idx
        return self.center_weights[rank_idx][file_idx]
    
    def _calculate_control_value(self):
        control_value = 0
        for square in chess.SQUARES:
            if bool(self.board.attackers(chess.WHITE, square)):
                control_value += self.center_weights[chess.square_rank(square)][chess.square_file(square)]
            if bool(self.board.attackers(chess.BLACK, square)):
                control_value -= self.center_weights[chess.square_rank(square)][chess.square_file(square)]
        return control_value
    
    def step(self, action):
        self.board.push(action)
        done = self.board.is_game_over()
        reward = self._calculate_reward()
        return self._get_state(), reward, done
    
    def _calculate_reward(self):
        # Oyun sonu kontrolü (checkmate/stalemate)
        if self.board.is_game_over():
            if self.board.is_checkmate():
                return 100 if self.board.turn == chess.BLACK else -100
            elif self.board.is_stalemate():
                return 0
    
        # Materyal değerleri
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0
        }
    
        # Temel ödül bileşenleri
        material_value = 0
        position_value = 0
        control_value = self._calculate_control_value()
        
        # Taş pozisyon ve materyal değerleri
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                value = piece_values[piece.piece_type]
                position_bonus = self._calculate_piece_position_value(square, piece.piece_type, piece.color)
                
                if piece.color == chess.WHITE:
                    material_value += value
                    position_value += position_bonus
                else:
                    material_value -= value
                    position_value -= position_bonus
    
        # Hamle kalitesi bonusları
        move_quality_bonus = 0
        if self.board.move_stack:  # Son hamleyi kontrol et
            last_move = self.board.peek()
            
            # 1. Şah çekme bonusu
            if self.board.is_check():
                move_quality_bonus += 0.8 if self.board.turn == chess.BLACK else -0.8
            
            # 2. Taş alma bonusu (capture)
            if self.board.is_capture(last_move):
                captured_piece = self.board.piece_at(last_move.to_square)
                if captured_piece:
                    bonus = piece_values[captured_piece.piece_type] * 0.5
                    move_quality_bonus += bonus if self.board.turn == chess.BLACK else -bonus
            
            # 3. Merkeze hamle bonusu (e4/d4/e5/d5)
            to_square = last_move.to_square
            if to_square in [chess.E4, chess.D4, chess.E5, chess.D5]:
                move_quality_bonus += 0.3 if self.board.turn == chess.WHITE else -0.3
    
        # Toplam ödül (mevcut değerler + hamle kalitesi)
        total_reward = (
            material_value * 1.0 +
            position_value * 0.3 +
            control_value * 0.2 +
            move_quality_bonus  # Yeni eklenen kalite metriği
        )
    
        return total_reward

class QLearningAgent:
    def __init__(self, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.9999):  # 0.995'ten 0.999'a değişti
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.alpha = 0.1
        self.gamma = 0.9
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def get_action(self, state, legal_moves):
        legal_moves_list = list(legal_moves)
        if not legal_moves_list:
            return None
        
        if random.random() < self.epsilon:
            return random.choice(legal_moves_list)
        else:
            return self._get_best_action(state, legal_moves_list)
    
    def _get_best_action(self, state, legal_moves):
        best_value = float('-inf')
        best_actions = []
        
        for move in legal_moves:
            value = self.q_table[state][str(move)]
            if value > best_value:
                best_value = value
                best_actions = [move]
            elif value == best_value:
                best_actions.append(move)
        
        return random.choice(best_actions) if best_actions else legal_moves[0]
    
    def learn(self, state, action, reward, next_state):
        if action is None:
            return
        
        old_value = self.q_table[state][str(action)]
        next_board = chess.Board(next_state)
        next_values = [self.q_table[next_state][str(move)] for move in next_board.legal_moves]
        next_max = max(next_values) if next_values else 0
        
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_table[state][str(action)] = new_value

class SimpleBlackPlayer:
    def __init__(self):
        self.piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0
        }
    
    def evaluate_move(self, board, move):
        score = 0
        
        # Taş alma hamlelerine büyük bonus (eskiden 10 katıydı, 20 katı yapın)
        captured_piece = board.piece_at(move.to_square)
        if captured_piece:
            score += self.piece_values[captured_piece.piece_type] * 20  # 10 → 20
        
        # Şah çekme bonusunu artır (5 → 8)
        board.push(move)
        if board.is_check():
            score += 8
        board.pop()
        
        # Merkeze hamle bonusu (0.1 → 0.5)
        to_file = chess.square_file(move.to_square)
        to_rank = chess.square_rank(move.to_square)
        center_distance = abs(3.5 - to_file) + abs(3.5 - to_rank)
        score += (8 - center_distance) * 0.5  # Daha agresif merkez kontrolü
        
        return score
    
    def get_move(self, board):
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None
        
        move_scores = [(move, self.evaluate_move(board, move)) for move in legal_moves]
        best_score = max(score for _, score in move_scores)
        best_moves = [move for move, score in move_scores if score == best_score]
        
        return best_moves[0]  # Rastgele seçim yapma, ilk en iyi hamleyi seç

def train_agent(episodes=1000):
    env = ChessEnvironment()
    agent = QLearningAgent()
    black_player = SimpleBlackPlayer()
    
    print("Eğitim başlıyor...")
    for episode in range(episodes):
        state = env.reset()
        done = False
        moves_count = 0
        
        while not done and moves_count < 100:
            if env.board.turn == chess.WHITE:
                action = agent.get_action(state, env.board.legal_moves)
                if action is None:
                    break
                    
                next_state, reward, done = env.step(action)
                agent.learn(state, action, reward, next_state)
                state = next_state
            
            else:
                black_move = black_player.get_move(env.board)
                if black_move is None:
                    break
                    
                state, reward, done = env.step(black_move)
            
            moves_count += 1
        
        # Epsilon decay
        agent.decay_epsilon()
        
        if episode % 100 == 0:
            print(f"Episode {episode} tamamlandı (epsilon: {agent.epsilon:.3f})")
    
    print("Eğitim tamamlandı!")
    return agent

def load_pieces():
    pieces = {}
    pieces_path = Path("chess_pieces")
    if not pieces_path.exists():
        pieces_path.mkdir()
        print("Lütfen satranç taşı resimlerini 'chess_pieces' klasörüne ekleyin!")
        return None
    
    piece_files = {
        'P': 'white_pawn.png',
        'R': 'white_rook.png',
        'N': 'white_knight.png',
        'B': 'white_bishop.png',
        'Q': 'white_queen.png',
        'K': 'white_king.png',
        'p': 'black_pawn.png',
        'r': 'black_rook.png',
        'n': 'black_knight.png',
        'b': 'black_bishop.png',
        'q': 'black_queen.png',
        'k': 'black_king.png'
    }
    
    missing_files = []
    for piece, file_name in piece_files.items():
        file_path = pieces_path / file_name
        if file_path.exists():
            try:
                image = pygame.image.load(str(file_path))
                pieces[piece] = pygame.transform.scale(image, (SQUARE_SIZE, SQUARE_SIZE))
                print(f"Yüklendi: {file_name}")
            except pygame.error as e:
                print(f"Hata - {file_name} yüklenemedi: {e}")
                missing_files.append(file_name)
        else:
            print(f"Dosya bulunamadı: {file_name}")
            missing_files.append(file_name)
    
    if missing_files:
        print("\nEksik dosyalar:")
        for file in missing_files:
            print(f"- {file}")
    
    return pieces

class ChessGUI:
    def __init__(self, agent):
        self.board = chess.Board()
        self.pieces = load_pieces()
        self.selected_square = None
        self.agent = agent
        self.game_status = ""
    
    def draw_board(self):
        for row in range(8):
            for col in range(8):
                color = WHITE if (row + col) % 2 == 0 else BLACK
                pygame.draw.rect(screen, color, 
                               (col * SQUARE_SIZE, row * SQUARE_SIZE, 
                                SQUARE_SIZE, SQUARE_SIZE))
                
                if self.selected_square == (row, col):
                    highlight_surface = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
                    pygame.draw.rect(highlight_surface, HIGHLIGHT, 
                                   (0, 0, SQUARE_SIZE, SQUARE_SIZE))
                    screen.blit(highlight_surface, (col * SQUARE_SIZE, row * SQUARE_SIZE))
    
    def draw_pieces(self):
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                x = chess.square_file(square) * SQUARE_SIZE
                y = (7 - chess.square_rank(square)) * SQUARE_SIZE
                if self.pieces and piece.symbol() in self.pieces:
                    screen.blit(self.pieces[piece.symbol()], (x, y))
    
    def square_to_coords(self, square):
        file_idx = chess.square_file(square)
        rank_idx = 7 - chess.square_rank(square)
        return (rank_idx, file_idx)
    
    def coords_to_square(self, row, col):
        return chess.square(col, 7 - row)
    
    def handle_click(self, pos):
        if self.board.is_game_over():
            return

        x, y = pos
        col = x // SQUARE_SIZE
        row = y // SQUARE_SIZE
        
        if self.selected_square is None:
            self.selected_square = (row, col)
        else:
            from_square = self.coords_to_square(self.selected_square[0], 
                                              self.selected_square[1])
            to_square = self.coords_to_square(row, col)
            
            move = chess.Move(from_square, to_square)
            if move in self.board.legal_moves:
                self.board.push(move)
                
                if self.board.is_game_over():
                    self.update_game_status()
                else:
                    # AI'ın hamlesi
                    ai_move = self.agent.get_action(self.board.fen(), 
                                                  self.board.legal_moves)
                    if ai_move:
                        self.board.push(ai_move)
                        if self.board.is_game_over():
                            self.update_game_status()
            
            self.selected_square = None

    def update_game_status(self):
        if self.board.is_checkmate():
            winner = "Siyah" if self.board.turn == chess.WHITE else "Beyaz"
            self.game_status = f"Mat! {winner} kazandı!"
        elif self.board.is_stalemate():
            self.game_status = "Pat! Berabere!"
        elif self.board.is_insufficient_material():
            self.game_status = "Yetersiz materyal! Berabere!"
    
    def draw_status(self):
        if self.game_status:
            font = pygame.font.Font(None, 36)
            text = font.render(self.game_status, True, (255, 0, 0))
            text_rect = text.get_rect(center=(WINDOW_SIZE/2, WINDOW_SIZE/2))
            background = pygame.Surface((text_rect.width + 20, text_rect.height + 20))
            background.fill(WHITE)
            background_rect = background.get_rect(center=(WINDOW_SIZE/2, WINDOW_SIZE/2))
            screen.blit(background, background_rect)
            screen.blit(text, text_rect)
    
    def run(self):
        running = True
        clock = pygame.time.Clock()
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Sol tık
                        self.handle_click(event.pos)
            
            self.draw_board()
            self.draw_pieces()
            self.draw_status()
            pygame.display.flip()
            clock.tick(60)
        
        pygame.quit()

if __name__ == "__main__":
    trained_agent = train_agent(episodes=1000)
    chess_gui = ChessGUI(trained_agent)
    chess_gui.run()