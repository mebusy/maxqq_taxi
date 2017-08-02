import ConfigParser

__config = None 


def getMaxNodes(  ) :
    global __config 
    return __config.sections()

def isPrimitiveAction( action ):
    if isinstance (action ,tuple) :
        action , _ = action 
    global __config
    return not __config.has_option( action , "actions"  )

def getPrimitiveActions() :
    max_nodes = getMaxNodes(  ) 
    prim_actions = [ a for a in max_nodes if isPrimitiveAction(a) ]
    return prim_actions

def getCompositeActions():
    max_nodes = getMaxNodes(  ) 
    comp_actions = [ a for a in max_nodes if not isPrimitiveAction(a) ]
    return comp_actions
    

# dict_param_bound :
# { "Nav" : [0,1,2,3] , "Shoot" : [5,6,7] }
from itertools import repeat , chain
def getAvailableActions( action  , dict_param_bound ) :
    if isinstance (action ,tuple) :
        action , param  = action 

    if isPrimitiveAction(action):
        return [(action , param )]
    else:
        actions = __config.get( action , "actions").translate( None, '[]' ).split( "," )
        actions = map(str.strip , actions ) 
        # if dict_param_bound and not set( dict_param_bound.keys() ).isdisjoint( actions )  :
        # print 'extending'
        actions = map( lambda x: [(x,None)] if not dict_param_bound or x not in dict_param_bound else zip(repeat( x ) , dict_param_bound[x] )  , actions )
        actions = list(chain.from_iterable( actions ))
        return actions 

def getImmediateReward( action , bReturnInvalidActionReward = False ) :
    if bReturnInvalidActionReward:
        return __config.getfloat( action , "r_invalid" )
    else:
        return __config.getfloat( action , "r" )

def visualizeGraph():
    # checking
    max_nodes = getMaxNodes(  )
    print "max nodes" ,  max_nodes

    prim_actions = getPrimitiveActions()
    print "prim actions" ,  prim_actions
    for a in prim_actions:
        assert __config.has_section( a  )


    for i in getCompositeActions():
        available_actions = getAvailableActions( i , None)
        print i, available_actions 

        for a in available_actions:
            assert __config.has_section( a[0]  )


    for i in getCompositeActions():
        available_actions = getAvailableActions( i , { "Navigate" : [0,1,2,3] , "Get" : [5,6,7] }  )
        print "bounded task: " ,i , available_actions



def parseMaxqGraph():
    global __config 
    __config = ConfigParser.RawConfigParser()
    __config.read('config.txt')

    visualizeGraph()



if __name__ == '__main__':
    parseMaxqGraph()
     



    pass
