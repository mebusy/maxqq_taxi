
import gym
from collections import defaultdict
from maxqGraph import *
import sys, operator
import numpy as np 

def encode( taxirow, taxicol, passloc, destidx):
    # (5) 5, 5, 4
    i = taxirow
    i *= 5
    i += taxicol
    i *= 5
    i += passloc
    i *= 4
    i += destidx
    return i

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

    taxirow, taxicol, passidx, destidx = decode( state )   
    if task == Task_Pickup :
        # passenger is in taxi,  or not reach passenger
        if passidx  >= 4 or (taxirow, taxicol) != locs[ passidx ] :
            bInvalid = True 
    elif task == Task_Putdown :
        # not in taxi , or not arrive destination 
        if passidx  < 4  or (taxirow, taxicol) != locs[ destidx  ] :
            bInvalid = True 

    return getImmediateReward( task, bInvalid   )

# in argmaxQ , use IsActiveState filter inavailable actions
def IsActiveState(  i , state   ) :
    task , param = i 
    
    if state_terminated:
        return False 
        
    taxirow, taxicol, passidx, destidx = decode( state )   
    if task == Task_Root : 
        return not state_terminated 
    elif task == Task_Get :
        return not passidx >= 4   # not in taxi
    elif task == Task_Put:
        return not passidx < 4    # in taxi
    elif task == Task_Navigate :
        return not (taxirow, taxicol) == locs[param ]
    else :
        return True 

# only used for composite task 
def IsTerminalState(  i , state  ) :
    task , param = i
    if state_terminated:
        return True 

    if isPrimitiveAction(task) :
        return True  # terminate in any state , because its terminal nodes
        
    taxirow, taxicol, passidx, destidx = decode( state )   
    if task == Task_Root : 
        return state_terminated 
    elif task == Task_Get :
        return passidx >= 4 
    elif task == Task_Put :
        return passidx < 4 
    elif task == Task_Navigate :
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
            if not IsActiveState( j,s ):
                continue 
            v , a =  EVALUATEMAXNODE(j,s) 
            v +=  Cvalues[(i,s,j)]  
            if v > v_opt :
                v_opt, a_opt = v , a  
        return (  v_opt, a_opt   ) 

def EXECUTEHGPOLICY( s ) :
    while True :
        v, a = EVALUATEMAXNODE( (Task_Root,None) ,s  )
        s , r, done , _ =  env.step(  task2Action[ a[0] ]    )
        env.render()
        if done:
            break 


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
            # eg. when passenger is in taxi , Root->Get will never be active for any state 
            pass

    if len(actions) == 0:
        return None

    if esp_exploaton :
        eps = 0.1
        if np.random.random() < eps :
            return actions [ np.random.choice( len( actions) ) ]

    qs = [ Q_tilde( i,s,a ) if pseudo_CR else Q( i,s,a )   for a in actions   ]
    i,v = max(enumerate( qs  ), key=operator.itemgetter(1))
    return actions[i]

def Q( i,s , a ):
    return V(a, s) + Cvalues[ (i,s,a) ] if a else 0.0

def Q_tilde( i,s,a ) :
    return V(a,s) + CTildevalues[ (i,s,a) ] if a else 0.0

def R_tilde( i,s ) :
    # if IsTerminalState(i, s) :
    #     return 1.0
    # else:
    #     return 0.0
    return 0.0

from collections import deque 
gamma = 1.0 
alpha = 0.25
s_prime = None 
state_terminated = False 

bExplore = True 
            
def MAXQ_Q( i  , s  ) :
    global s_prime , state_terminated 
    global bExplore 


    seq = deque()  # be the sequence of states visited while executing i
    # The list of states will be ordered most-recent-first
    
    if isPrimitiveAction( i) :  # primitive MaxNode
        s_prime  , r , state_terminated , _ = env.step( task2Action[ i[0] ] ) 
        
        meanCumulativeReward = debug_reward[-1] if len(debug_reward) > 0 else 0.0 
        newMean = meanCumulativeReward + ( r - meanCumulativeReward  ) / ( len(debug_reward)+1 ) 
        debug_reward.append( newMean  )


        if bRender :
            env.render() 
        Vvalues[(i,s)] = (1.0 - alpha) * Vvalues[(i,s)] + alpha * r
        # print Vvalues[(i,s)]
        seq.appendleft( s ) # push 5 onto the beginning of seq
    else:
        # count = 0
        while not IsTerminalState( i,s ) :
            # choose an action a according to the current exploration policy pi_x(i,s)
            # Q tilde , with exploation 
            a = argmaxQ ( i,s , bExplore , True  )
            if not a:
                break 

            childSeq = MAXQ_Q(a,s)  # childSeq is the sequence of states visited , in reverse order
            
            # here we needobserve result state s' , saved in global variable s_prime
            # Q tilde, without exploation 
            a_opt = argmaxQ( i, s_prime , False , True   )
            if not a_opt:
                break

            N = 1
            for _s in childSeq:
                # first update CTildevalues using a_opt 
                # this update include R_tilde 
                # in paper , the last term is V(a*,s) , not V(a*,s'))
                d = gamma**N 
                # CTildevalues[(i,s_prime,a_opt)] + V(a_opt, s_prime  => Q_tilde
                CTildevalues[(i,_s,a)] = (1-alpha)*CTildevalues[(i,_s,a)] + alpha*d*( R_tilde(i,s_prime) + Q_tilde( i, s_prime , a_opt  )  ) 
                # update Cvalues using a_opt
                # Cvalues      (i,s_prime,a_opt) + V(s_opt, s_prime  => Q
                Cvalues     [(i,_s,a)] = (1-alpha)*Cvalues     [(i,_s,a)] + alpha*d*(                      Q( i, s_prime , a_opt )  ) 
                N = N+1 

            seq.extend( childSeq )
            s = s_prime 
            # end while
        # end else 
    return seq 

         

class AbstractVvalues( defaultdict  ) :
    def state_abstract(self , i , s) :
        # print s 
        task , param = i 
        # abs1: North, South, East, and West. These terminal nodes require one quantity each
        if task in [ Task_South,Task_North , Task_East , Task_West ] :
            s = -1

        # abs2: Pickup and Putdown each require 2 values (legal and illegal states), 
        elif task == Task_Pickup :
            taxirow, taxicol, passidx, destidx = decode( s )
            s = -2 if passidx  >= 4 or (taxirow, taxicol) != locs[ passidx ] else -1

        elif task == Task_Putdown :
            taxirow, taxicol, passidx, destidx = decode( s )
            s = -2 if passidx  < 4  or (taxirow, taxicol) != locs[ destidx  ]  else -1 

        return (i,s ) 
        
    def __setitem__(self, key, value):
        key = self.state_abstract( *key ) 
        return super( AbstractVvalues , self  ).__setitem__( key, value)

    def __getitem__( self, key  ) :
        key = self.state_abstract( *key ) 
        # prevent calling __setitem__ again
        if key not in self :
            return self.default_factory() 
        return super( AbstractVvalues , self  ).__getitem__( key)


class AbstractCvalues( defaultdict  ) :
    def state_abstract(self , i , s , a ) :
        task , param = i 
        subtask , subparam = a 
        taxirow, taxicol, passidx, destidx = decode( s ) 
        # abs3 : QNorth(t), QSouth(t), QEast(t), and QWest(t) each require 100 values 
        # (four values for t and 25 locations). (Max Node Irrelevance.)
        # p
        if task == Task_Navigate:
            # t is bounded in i
            s = encode( taxirow, taxicol , 0,0  ) 

        # abs4: QNavigateForGet requires 4 values (for the four possible source locations).
        # The passenger destination is Max Node Irrelevant for MaxGet
        # and the taxi starting location is Result Distribution Irrelevant for the Navigate action
        elif task == Task_Get and subtask == Task_Navigate :
            s = encode( 0,0, passidx , 0  )
            
        # abs5: QPickup requires 100 possible values, 4 possible source locations and 25 possible taxi locations.
        # p
        elif task == Task_Get and subtask == Task_Pickup :    
            s = encode( taxirow, taxicol, passidx , 0  )

        # abs6 : QGet requires 16 possible values(4 source locations, 4 destination locations). (Result Distribution Irrelevance.)
        # p
        elif subtask == Task_Get:  # C(Root,s,Get)
            s = encode( 0,0, passidx , destidx  )

        # abs7: QNavigateForPut requires only 4 values (for the four possible destination locations).
        elif task == Task_Put and subtask == Task_Navigate :
            s = encode( 0,0, 0 , destidx  ) 

        # QPutdown requires 100 possible values (25 taxi locations, 4 possible destination locations). 
        # (Passengersourceis Max Node Irrelevant for MaxPut.)
        # p
        elif task == Task_Put and subtask == Task_Putdown :
            s = encode( taxirow, taxicol, 0, destidx  ) 

        # QPut requires 0 values. (Termination and Shielding.)
        # p
        elif subtask == Task_Put:  # C(Root, s, Put)
            s = -1 
        return (i,s ,a  )

    def __setitem__(self, key, value):
        key = self.state_abstract( *key ) 
        return super(AbstractCvalues , self  ).__setitem__( key, value)

    def __getitem__( self, key  ) :
        key = self.state_abstract( *key ) 
        # prevent calling __setitem__ again
        if key not in self :
            return self.default_factory() 
        return super(AbstractCvalues , self  ).__getitem__( key)


if __name__ == '__main__' :
    parseMaxqGraph() 
    for task in getMaxNodes():
        exec( "Task_{0} = '{1}'".format(  task, task   ) )


    task2Action = {
        Task_South: 0,
        Task_North: 1,
        Task_East: 2, 
        Task_West: 3,
        Task_Pickup: 4, 
        Task_Putdown: 5,
    }
    action2Task = dict((v,k) for k,v in task2Action.iteritems())


    task_bound = { Task_Navigate : [0,1,2,3] }

    locs = [(0,0), (0,4), (4,0), (4,3)]

    env = gym.make('Taxi-v2') 
    s = env.reset()
    
    Cvalues = AbstractCvalues ( float )
    Vvalues = AbstractVvalues( float  ) 
    CTildevalues = AbstractCvalues( float   ) 

    debug_reward = []

    bRender = False 
    for i in (xrange(300)) : 
        MAXQ_Q( ( Task_Root ,None ) , s  )

        state_terminated = False  
        print 'episode '  , i  , V(  ( Task_Root ,None ) , s  )
        s = env.reset() 
        s_prime = None          

    bRender = True 
    bExplore = False 
    s = env.reset()
    MAXQ_Q( ( Task_Root ,None ) , s  )

    # import matplotlib.pyplot as pp
    # pp.plot( debug_reward  )
    # pp.show()

    print 'done' 

