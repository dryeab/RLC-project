from RLC.move_chess.environment import Board
from RLC.move_chess.agent import Piece
from RLC.move_chess.learn import Reinforce

env = Board()
p = Piece(piece='rook')
r = Reinforce(p,env)

r.policy_iteration(k=1,gamma=1,synchronous=True)