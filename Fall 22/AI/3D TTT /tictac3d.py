"""
Basic 3D Tic Tac Toe with Minimax and Alpha-Beta pruning, using a simple
heuristic to check for possible winning moves or blocking moves if no better
alternative exists.
"""


from shutil import move
from colorama import Back, Style, Fore
import numpy as np
from random import choice

class TicTacToe3D(object):
    """3D TTT logic and underlying game state object.

    Attributes:
        board (np.ndarray)3D array for board state.
        difficulty (int): Ply; number of moves to look ahead.
        depth_count (int): Used in conjunction with ply to control depth.

    Args:
        player (str): Player that makes the first move.
        player_1 (Optional[str]): player_1's character.
        player_2 (Optional[str]): player_2's character.
        ply (Optional[int]): Number of moves to look ahead.
    """

   

    def __init__(self, board = None, player=-1, player_1=-1, player_2=1, ply=3):
        if board is not None:
            assert type(board) == np.ndarray, "Board must be a numpy array"
            assert board.shape == (3,3,3), "Board must be 3x3x3"
            self.np_board = board
        else:
            self.np_board = self.create_board()
        self.map_seq_to_ind, self.map_ind_to_seq = self.create_map()
        self.allowed_moves = list(range(pow(3, 3)))
        self.difficulty = ply
        self.depth_count = 0
        if player == player_1:
            self.player_1_turn = True
        else:
            self.player_1_turn = False
        self.player_1 = player_1
        self.player_2 = player_2
        self.players = (self.player_1, self.player_2)

    def create_map(self):
        """Create a mapping between index of 3D array and list of sequence, and vice-versa.

        Args: None

        Returns:
            map_seq_to_ind (dict): Mapping between sequence and index.
            map_ind_to_seq (dict): Mapping between index and sequence.
        """
        a = np.hstack((np.zeros(9),np.ones(9),np.ones(9)*2))
        a = a.astype(int)
        b = np.hstack((np.zeros(3),np.ones(3),np.ones(3)*2))
        b = np.hstack((b,b,b))
        b = b.astype(int)
        c = np.array([0,1,2],dtype=int)
        c = np.tile(c,9)
        mat = np.transpose(np.vstack((a,b,c)))
        ind = np.linspace(0,26,27).astype(int)
        map_seq_to_ind = {}
        map_ind_to_seq = {}
        for i in ind:
            map_seq_to_ind[i] = (mat[i][0],mat[i][1],mat[i][2])
            map_ind_to_seq[(mat[i][0],mat[i][1],mat[i][2])] = i
        return map_seq_to_ind, map_ind_to_seq
    
    def reset(self):
        """Reset the game state."""
        self.allowed_moves = list(range(pow(3, 3)))
        self.np_board = self.create_board()
        self.depth_count = 0


    @staticmethod
    def create_board():
        """Create the board with appropriate positions and the like

        Returns:
            np_board (numpy.ndarray):3D array with zeros for each position.
        """
        np_board = np.zeros((3,3,3), dtype=int)
        return np_board

    
    def winningPlayer(self, brd, player):
        winningStates = [[brd[0][0][0], brd[0][0][1], brd[0][0][2]],
                        [brd[0][1][0], brd[0][1][1], brd[0][1][2]],
                        [brd[0][2][0], brd[0][2][1], brd[0][2][2]],
                        [brd[0][0][0], brd[0][1][0], brd[0][2][0]],
                        [brd[0][0][1], brd[0][1][1], brd[0][2][1]],
                        [brd[0][0][2], brd[0][1][2], brd[0][2][2]],
                        [brd[0][0][0], brd[0][1][1], brd[0][2][2]],
                        [brd[0][0][2], brd[0][1][1], brd[0][2][0]],
                        
                        [brd[1][0][0], brd[1][0][1], brd[1][0][2]],
                        [brd[1][1][0], brd[1][1][1], brd[1][1][2]],
                        [brd[1][2][0], brd[1][2][1], brd[1][2][2]],
                        [brd[1][0][0], brd[1][1][0], brd[1][2][0]],
                        [brd[1][0][1], brd[1][1][1], brd[1][2][1]],
                        [brd[1][0][2], brd[1][1][2], brd[1][2][2]],
                        [brd[1][0][0], brd[1][1][1], brd[1][2][2]],
                        [brd[1][0][2], brd[1][1][1], brd[1][2][0]],

                        [brd[2][0][0], brd[2][0][1], brd[2][0][2]],
                        [brd[2][1][0], brd[2][1][1], brd[2][1][2]],
                        [brd[2][2][0], brd[2][2][1], brd[2][2][2]],
                        [brd[2][0][0], brd[2][1][0], brd[2][2][0]],
                        [brd[2][0][1], brd[2][1][1], brd[2][2][1]],
                        [brd[2][0][2], brd[2][1][2], brd[2][2][2]],
                        [brd[2][0][0], brd[2][1][1], brd[2][2][2]],
                        [brd[2][0][2], brd[2][1][1], brd[2][2][0]],

                        [brd[0][0][0], brd[1][0][0], brd[2][0][0]],
                        [brd[0][0][1], brd[1][0][1], brd[2][0][1]],
                        [brd[0][0][2], brd[1][0][2], brd[2][0][2]],
                        [brd[0][1][0], brd[1][1][0], brd[2][1][0]],
                        [brd[0][1][1], brd[1][1][1], brd[2][1][1]],
                        [brd[0][1][2], brd[1][1][2], brd[2][1][2]],
                        [brd[0][2][0], brd[1][2][0], brd[2][2][0]],
                        [brd[0][2][1], brd[1][2][1], brd[2][2][1]],
                        [brd[0][2][2], brd[1][2][2], brd[2][2][2]],

                        [brd[0][0][0], brd[1][0][1], brd[2][0][2]],
                        [brd[0][1][0], brd[1][1][1], brd[2][1][2]],
                        [brd[0][2][0], brd[1][2][1], brd[2][2][2]],

                        [brd[0][0][0], brd[1][1][0], brd[2][2][0]],
                        [brd[0][0][1], brd[1][1][1], brd[2][2][1]],
                        [brd[0][0][2], brd[1][1][2], brd[2][2][2]],

                        [brd[0][0][0], brd[1][1][1], brd[2][2][2]],
                        [brd[0][0][2], brd[1][1][1], brd[2][2][0]],
                        [brd[0][2][0], brd[1][1][1], brd[2][0][2]],
                        [brd[0][2][2], brd[1][1][1], brd[2][0][0]],
                        ]
        isWinner = [player, player, player] in winningStates
        return isWinner
    
    def getScore(self, brd):
        if self.winningPlayer(brd, +1):
            return 10

        elif self.winningPlayer(brd, -1):
            return -10

        else:
            return 0

    def gameWon(self, board):
        return self.winningPlayer(board, +1) or self.winningPlayer(board, -1)

    def emptyCells(self, board):
        emptyC = []
        for x in range(3):
            for y in range(3):
                for z in range(3):
                    if board[x][y][z] == 0:
                        emptyC.append([x, y, z])

        return emptyC

    def boardFull(self, board):
        if len(self.emptyCells(board)) == 0:
            return True
        return False

    def MiniMaxAB(self, board, depth, alpha, beta, player):
        sqr = -1
        row = -1
        col = -1
    
        if depth == 0 or self.gameWon(board):
            return [sqr, row, col, self.getScore(board)]

        else:
            for cell in self.emptyCells(board):
                self.setMove(board, cell[0], cell[1], cell[2], player)
                score = self.MiniMaxAB(board, depth - 1, alpha, beta, -player)
                if player == +1:
                    if score[3] > alpha:
                        alpha = score[3]
                        sqr = cell[0]
                        row = cell[1]
                        col = cell[2]

                else:
                    if score[3] < beta:
                        beta = score[3]
                        sqr = cell[0]
                        row = cell[1]
                        col = cell[2]

                self.setMove(board, cell[0], cell[1], cell[2], 0)
                if alpha >= beta:
                    break

            if player == +1:
                return [sqr, row, col, alpha]

            else:
                return [sqr, row, col, beta]

    def setMove(self, board, x, y, z, currentPlayer):
        board[x][y][z] = currentPlayer

    def AI_MOVE(self, board, currentPlayer):
        if len(self.emptyCells(board)) == 27:
            x = choice([0, 1, 2])
            y = choice([0, 1, 2])
            z = choice([0, 1, 2])
            self.setMove(board, x, y, z, currentPlayer)
            return board

        else:
            result = self.MiniMaxAB(board, 5, -1, 1, currentPlayer)
            self.setMove(board, result[0], result[1], result[2], currentPlayer)
            return board
    
    def AI_MOVE2(self, board):
        if len(self.emptyCells(board)) == 27:
            x = choice([0, 1, 2])
            y = choice([0, 1, 2])
            z = choice([0, 1, 2])
            self.setMove(board, x, y, z, -1)
            return board

        else:
            result = self.MiniMaxAB(board, 5, -1, 1, -1)
            self.setMove(board, result[0], result[1], result[2], -1)
            return board

    def play_game(self):
        """Primary game loop.

        Until the game is complete we will alternate between computer and
        player turns while printing the current game state.
        """
        try:
            board = self.np_board
            print(board)
            if(self.player_1_turn):
                currentPlayer = self.player_1
            else:
                 currentPlayer = self.player_2
            while not (self.boardFull(board) or self.gameWon(board)):
                    finalBoard = self.AI_MOVE(board, currentPlayer)
                    currentPlayer *= -1

            return finalBoard, -currentPlayer
        except KeyboardInterrupt:
            print('\n ctrl-c detected, exiting')



if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        '--player', dest='player', help='Player that plays first, 1 or -1',\
                type=int, default=-1, choices=[1,-1]
    )
    parser.add_argument(
        '--ply', dest='ply', help='Number of moves to look ahead', \
                type=int, default=6
    )
    args = parser.parse_args()
    brd,winner = TicTacToe3D(player=args.player, ply=args.ply).play_game()
    print("final board: \n{}".format(brd))
    print("winner: player {}".format(winner))
