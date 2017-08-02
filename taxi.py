
import gym
from collections import defaultdict
from maxqGraph import *
import sys, operator
import numpy as np 


task2Action = {
    'South': 0,
    'North': 1,
    'East': 2, 
    'West': 3,
    'Pickup': 4, 
    'Putdown': 5,
}
action2Task = dict((v,k) for k,v in task2Action.iteritems())

task_bound = { "Navigate" : [0,1,2,3] }

locs = [(0,0), (0,4), (4,0), (4,3)]
def decode( i):
    out = []
    out.append(i % 4)
    i = i // 4
    out.append(i % 5)
    i = i // 5
    out.append(i % 5)
    i = i // 5
    out.append(i)
    assert 0 <= i < 5
    return reversed(out)


# manually 
def ImmediateReward( task, state ):
    task , _ = task 

    bInvalid = False 

    if task == 'Pickup' :
        taxirow, taxicol, passidx, destidx = decode( state )   
        # passenger is in taxi,  or not reach passenger
        if passidx  >= 4 or (taxirow, taxicol) != locs[ passidx ] :
            bInvalid = True 
    elif task == 'Putdown' :
        # not in taxi , or not arrive destination 
        if passidx  < 4  or (taxirow, taxicol) != locs[ destidx  ] :
            bInvalid = True 

    return getImmediateReward( task, bInvalid   )

def IsActiveState(  i , state   ) :
    task , param = i 
    if isPrimitiveAction(task) :
        return True  #  can be performed in any state
        
    taxirow, taxicol, passidx, destidx = decode( state )   
    if task == 'Root' : 
        return not state_terminated 
    elif task == 'Get' :
        return not passidx >= 4 
    elif task == 'Put' :
        return not passidx < 4 
    elif task == 'Navigate' :
        return not (taxirow, taxicol) == locs[param ]
    else :
        assert False 

def IsTerminalState(  i , state  ) :
    task , param = i
    if state_terminated:
        return True 

    if isPrimitiveAction(task) :
        return True  # terminate in any state
        
    taxirow, taxicol, passidx, destidx = decode( state )   
    if task == 'Root' : 
        return state_terminated 
    elif task == 'Get' :
        return passidx >= 4 
    elif task == 'Put' :
        return passidx < 4 
    elif task == 'Navigate' :
        return (taxirow, taxicol) == locs[param ]
    else :
        assert False 



# Pseudo-code for Greedy Execution of the MAXQ Graph.
# MAXQ graph do not store V , so it need 
# recursively go through graph to evaluate V 
# In addition to returning Vt(i, s) , also returns the action at the leaf node that achieves this value. 
def EVALUATEMAXNODE(i,s) :
    if isPrimitiveAction(i):
        # the exit condition is primitive action
        # so EVALUATEMAXNODE will always return (v,primitive_a ) 
        return ( ImmediateReward( i,s ) , i ) 
    else:
        v_opt = - sys.maxint 
        a_opt = None 
        for j in getAvailableActions( i ,task_bound ) :
            v , a =  EVALUATEMAXNODE(j,s) 
            v +=  Cvalues[(i,s,j)]  
            if v > v_opt :
                v_opt, a_opt = v , a  
        return (  v_opt, a_opt   ) 


# true value 
def V( i, s ) :
    if isPrimitiveAction(i) :
        return Vvalues[ (i,s) ]
    else:
        return Q(i,s, argmaxQ(i,s) )


def argmaxQ( i, s , esp_exploaton = False , pseudo_CR = False  ) :
    actions = []
    for j in getAvailableActions( i , task_bound ) :
        if IsActiveState( j,s ):
            actions.append( j ) 
        else :    
            # print "in active state for " , j , i 
            pass

    if esp_exploaton :
        eps = 0.1
        if np.random.random() < eps :
            return actions [ np.random.choice( len( actions) ) ]

    qs = [ Q_tilde( i,s,a ) if pseudo_CR else Q( i,s,a )   for a in actions   ]
    i,v = max(enumerate( qs  ), key=operator.itemgetter(1))
    return actions[i]

def Q( i,s , a ):
    return V(a, s) + Cvalues[ (i,s,a) ]

def Q_tilde( i,s,a ) :
    return V(a,s) + CTildevalues[ (i,s,a) ] 

def R_tilde( i,s ) :
    if IsTerminalState(i, s) :
        return 1.0
    else:
        return 0.0


from collections import deque 
alpha = 0.25
s_prime = None 
state_terminated = False 
def MAXQ_Q( i  , s  ) :
    global s_prime , state_terminated 
    seq = deque()  # be the sequence of states visited while executing i
    # The list of states will be ordered most-recent-first
    
    if isPrimitiveAction( i) :  # primitive MaxNode
        s_prime  , r , state_terminated , _ = env.step( task2Action[ i[0] ] ) 
        Vvalues[(i,s)] = (1.0 - alpha) * Vvalues[(i,s)] + alpha * r
        seq.appendleft( s ) # push 5 onto the beginning of seq
    else:
        # count = 0
        while not IsTerminalState( i,s ) :
            # choose an action a according to the current exploration policy pi_x(i,s)
            # Q tilde , with exploation 
            a = argmaxQ ( i,s , True , True  )

            childSeq = MAXQ_Q(a,s)  # childSeq is the sequence of states visited , in reverse order
            
            # here we needobserve result state s' , saved in global variable s_prime
            # Q tilde, without exploation 
            a_opt = argmaxQ( i, s_prime , False , True   )
            N = 1
            for _s in childSeq:
                # first update CTildevalues using a_opt 
                # this update include R_tilde 
                # in paper , the last term is V(a*,s) , not V(a*,s'))
                gamma = 1.0 
                d = gamma**N 
                # CTildevalues[(i,s_prime,a_opt)] + V(a_opt, s_prime  => Q_tilde
                CTildevalues[(i,_s,a)] = (1-alpha)*CTildevalues[(i,_s,a)] + alpha*d*( R_tilde(i,s_prime) + Q_tilde( i, s_prime , a_opt  )  ) 
                # update Cvalues using a_opt
                # Cvalues      (i,s_prime,a_opt) + V(s_opt, s_prime  => Q
                Cvalues     [(i,_s,a)] = (1-alpha)*Cvalues     [(i,_s,a)] + alpha*d*(                      Q( i, s_prime , a_opt )  ) 
                N = N+1 

            seq.extend( childSeq )
            s = s_prime 
    return seq 

         

if __name__ == '__main__' :
    parseMaxqGraph() 

    env = gym.make('Taxi-v2') 
    s = env.reset()

    Cvalues = defaultdict( float )
    Vvalues = defaultdict( float ) 
    CTildevalues = defaultdict( float ) 
    
    MAXQ_Q( ( 'Root' ,None ) , s  )

    print 'done' , state_terminated 
