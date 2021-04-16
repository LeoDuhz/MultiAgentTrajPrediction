#  referee state in proto file
 
 # All robots should completely stop moving.
HALT = 0 
 # Robots must keep 50 cm from the ball.
STOP = 1 
 # A prepared kickoff or penalty may now be taken.
NORMAL_START = 2 
 # The ball is dropped and free for either team.
FORCE_START = 3 
 # The yellow team may move into kickoff position.
PREPARE_KICKOFF_YELLOW = 4 
 # The blue team may move into kickoff position.
PREPARE_KICKOFF_BLUE = 5 
 # The yellow team may move into penalty position.
PREPARE_PENALTY_YELLOW = 6 
 # The blue team may move into penalty position.
PREPARE_PENALTY_BLUE = 7 
 # The yellow team may take a direct free kick.
DIRECT_FREE_YELLOW = 8 
 # The blue team may take a direct free kick.
DIRECT_FREE_BLUE = 9 
 # The yellow team may take an indirect free kick.
INDIRECT_FREE_YELLOW = 10 
 # The blue team may take an indirect free kick.
INDIRECT_FREE_BLUE = 11 
 # The yellow team is currently in a timeout.
TIMEOUT_YELLOW = 12 
 # The blue team is currently in a timeout.
TIMEOUT_BLUE = 13 
 # The yellow team just scored a goal.
 # For information only.
 # For rules compliance, teams must treat as STOP.
GOAL_YELLOW = 14 
 # The blue team just scored a goal.
GOAL_BLUE = 15 