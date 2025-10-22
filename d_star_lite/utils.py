

def stateNameToCoords(name):
    return [int(name.split('x')[1].split('y')[0]), int(name.split('x')[1].split('y')[1])]

def coordsToStateName(col, row):
    """Convert column and row coordinates to state name format 'x{col}y{row}'"""
    return f"x{col}y{row}"
