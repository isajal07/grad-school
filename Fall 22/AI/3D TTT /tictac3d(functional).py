from random import choice
from math import inf
import numpy as np

XPLAYER = +1
OPLAYER = -1
EMPTY = 0

board = np.array([[[ 0, 0, 0],
                              [ 0, 1, 0],
                              [ 0, 0,-1]],
                             
                             [[ 0, 0, 0],
                              [ 0, -1, 0],
                              [ 0, 0, 0]],
                             
                             [[ 0, 1,-1],
                              [ 0,-1, 0],
                              [ 0, 1, 0]]])


def printBoard(brd):
    print(np.array(brd))



def clearBoard(brd):
    for x, sqr in enumerate(brd):
        for y, row in enumerate(sqr):
            for z, col in enumerate(row):
                brd[x][y][z] = EMPTY
    

def winningPlayer(brd, player):
    # print([brd[0][0][0], brd[0][0][1], brd[0][0][2]])
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

                     [brd[0][0][2], brd[1][1][1], brd[2][2][1]],
                     [brd[2][0][2], brd[2][1][1], brd[2][2][0]],
                     [brd[2][1][0], brd[2][1][1], brd[2][1][2]],
                     [brd[0][0][2], brd[1][2][1], brd[2][1][1]],
                     ]
    # print(brd)
    isWinner = [player, player, player] in winningStates
    return isWinner
   


def gameWon(brd):
    # print('winning game',winningPlayer(brd, XPLAYER) or winningPlayer(brd, OPLAYER))
    return winningPlayer(brd, XPLAYER) or winningPlayer(brd, OPLAYER)


def printResult(brd):
    if winningPlayer(brd, XPLAYER):
        print('+1 won! ' + '\n')
    elif winningPlayer(brd, OPLAYER):
        print('-1 won! ' + '\n')
    else:
        print('Draw' + '\n')

def emptyCells(brd):
    # print(brd)
    emptyC = []
    for x, sqr in enumerate(brd):
        for y, row in enumerate(sqr):
            for z, col in enumerate(row):
                if brd[x][y][z] == EMPTY:
                    emptyC.append([x, y, z])

    return emptyC


def boardFull(brd):
    if len(emptyCells(brd)) == 0:
        return True
    return False


def setMove(brd, x, y, z, player):
    # print(brd[x][y][z], player)
    brd[x][y][z] = player
    # print(brd)

def getScore(brd):
    if winningPlayer(brd, XPLAYER):
        return 10

    elif winningPlayer(brd, OPLAYER):
        return -10

    else:
        return 0


def MiniMaxAB(brd, depth, alpha, beta, player):
    sqr = -1
    row = -1
    col = -1
    # print( gameWon(brd))
    if depth == 0 or gameWon(brd):
        return [sqr, row, col, getScore(brd)]

    else:
        for cell in emptyCells(brd):
            # print('cells:', cell)
            setMove(brd, cell[0], cell[1], cell[2], player)
            score = MiniMaxAB(brd, depth - 1, alpha, beta, -player)
            # print('score:', score)
            if player == XPLAYER:
                # X is always the max player
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

            setMove(brd, cell[0], cell[1], cell[2], EMPTY)
            if alpha >= beta:
                break

        if player == XPLAYER:
            return [sqr, row, col, alpha]

        else:
            return [sqr, row, col, beta]


def AIMove(brd, player):
    # print('here')
    # print('here in 1')
    if len(emptyCells(brd)) == 2:
        x = choice([0, 1, 2])
        y = choice([0, 1, 2])
        z = choice([0, 1, 2])
        # print(brd, x, y, z, OPLAYER)
        setMove(brd, x, y, z, XPLAYER)
        printBoard(brd)

    else:
        # print('calledDown')
        result = MiniMaxAB(brd, len(emptyCells(brd)), -1, 1, XPLAYER)
        # print('result:',result)
        setMove(brd, result[0], result[1], result[2], XPLAYER)
        printBoard(brd)


def AI2Move(brd):
    if len(emptyCells(brd)) == 2:
        x = choice([0, 1, 2])
        y = choice([0, 1, 2])
        z = choice([0, 1, 2])
        setMove(brd, x, y, z, OPLAYER)
        printBoard(brd)

    else:
        # print(brd, len(emptyCells(brd)), -inf, inf, OPLAYER)
        # ERROR IS HERE! IN THE THIS BELOW LINE.
        result = MiniMaxAB(brd, len(emptyCells(brd)), -1, 1, OPLAYER)
        setMove(brd, result[0], result[1], result[2], OPLAYER)
        printBoard(brd)


def AIvsAI():
    currentPlayer = XPLAYER
    # count = 0
    # board = np.zeros((3, 3, 3), dtype=int)
    # print(board)
    # print(board)
    # for x in range(2):
        # clearBoard(board)
        
    while not (boardFull(board) or gameWon(board)):
            makeMove(board, currentPlayer)
            currentPlayer *= -1
            # print("currentPlayer: " , currentPlayer)
    printResult(board)
        # if gameWon(board):
        #     count += 1

    # print('Number of AI vs AI wins =', count)

 
def makeMove(brd, player):

        print('Loading final result...')
        if player == XPLAYER:
            AIMove(brd, player)
        else:
            AI2Move(brd)

# def main():
AIvsAI()

# if __name__ == '__main__':
#     main()
