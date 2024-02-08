"""
A12:
why does the illustration not have a well?
it was *explicitly* stated that there yhould be an even number of plays
schnick schnack schnuck with:
- (sc)Schere/scissors
- (st)Stein/stone
- (pa)Papier/paper
- (sp)Spock
- (br)Brunnen/well
- (ec)Echse/lizard
"""
from random import randint
class playable_type():
    def __init__(self, id, winn) -> None:
        self.id = id
        self.winn = winn
    
    def get_id(self) -> int:
        return self.id
    def get_winn(self) -> list:
        return self.winn

type_list = []

type_list.append(playable_type("sc", ["pa", "ec"]))
type_list.append(playable_type("st", ["sc", "ec"]))
type_list.append(playable_type("pa", ["st", "sp", "br"]))
type_list.append(playable_type("sp", ["st", "st", "br"]))
type_list.append(playable_type("br", ["st", "sc"]))
type_list.append(playable_type("ec", ["sp", "pa"]))

computer_points = 0
player_points = 0
for i in range(10):
    # taking players move:
    player_play = input("play one of:\n- (sc)Schere/scissors\n- (st)Stein/stone\n- (pa)Papier/paper\n- (sp)Spock\n- (br)Brunnen/well\n- (ec)Echse/lizard\n")
    
    #computer move:
    computer_play = type_list[randint(0,5)]
    print("\nPc plays:", computer_play.get_id())

    if(computer_play.get_id() == player_play):
        print("draw!\n")
    elif(player_play in computer_play.get_winn()):
        print("computer winns!\n")
        computer_points += 1
    else:
        print("you win!\n")
        player_points += 1
    
    print("Score:\nPlayer: ", player_points, "\nComputer: ", computer_points, "\n")
