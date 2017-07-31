import ConfigParser

config = None 


def getMaxNodes(  ) :
    global config 
    return config.sections()

def isPrimitiveAction( action ):
    global config
    return not config.has_option( action , "actions"  )

def getPrimitiveActions() :
    max_nodes = getMaxNodes(  ) 
    prim_actions = [ a for a in max_nodes if isPrimitiveAction(a) ]
    return prim_actions

def getCompositeActions():
    max_nodes = getMaxNodes(  ) 
    comp_actions = [ a for a in max_nodes if not isPrimitiveAction(a) ]
    return comp_actions
    

def getAvailableActions( action ) :
    if isPrimitiveAction(action):
        return action 
    else:
        actions = config.get( action , "actions").translate( None, '[]' ).split( "," )
        actions = map(str.strip , actions ) 
        return actions 

def visualizeGraph():
    # checking
    max_nodes = getMaxNodes(  )
    print max_nodes

    prim_actions = getPrimitiveActions()
    print prim_actions
    for a in prim_actions:
        assert config.has_section( a  )


    for i in getCompositeActions():
        available_actions = getAvailableActions( i )
        print i, available_actions 

        for a in available_actions:
            assert config.has_section( a  )





def parseMaxqGraph():
    global config 
    config = ConfigParser.RawConfigParser()
    config.read('config.txt')

    visualizeGraph()



if __name__ == '__main__':
    parseMaxqGraph()
     



    pass
