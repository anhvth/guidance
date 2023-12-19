from guidance import models
OPEN_BLOCKS = {}
id_open_block = id(OPEN_BLOCKS)

class ContextBlock:
    def __init__(self, opener, closer, name=None):
        self.opener = opener
        self.closer = closer
        self.name = name
        self.id_open_block = id_open_block

    def __enter__(self):
        OPEN_BLOCKS[self] = None
    
    def __exit__(self, exc_type, exc_value, traceback):
        del OPEN_BLOCKS[self]
        
def block(name=None, opener="", closer=""):
    return ContextBlock(opener, closer, name=name)