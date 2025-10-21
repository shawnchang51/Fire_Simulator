import heapq
from .utils import stateNameToCoords

OBS_VAL = -1

def set_OBS_VAL(val):
    global OBS_VAL
    OBS_VAL = val

def topKey(queue):
    # Heap property guarantees queue[0] is minimum - no need to sort!
    # Sorting was breaking the heap structure and causing O(n log n) overhead
    if len(queue) > 0:
        return queue[0][:2]
    else:
        return (float('inf'), float('inf'))


def heuristic_from_s(graph, id, s):
    x_distance = abs(int(id.split('x')[1][0]) - int(s.split('x')[1][0]))
    y_distance = abs(int(id.split('y')[1][0]) - int(s.split('y')[1][0]))
    return max(x_distance, y_distance)


def calculateKey(graph, id, s_current, k_m):
    return (min(graph.graph[id].g, graph.graph[id].rhs) + heuristic_from_s(graph, id, s_current) + k_m, min(graph.graph[id].g, graph.graph[id].rhs))


def updateVertex(graph, queue, queue_set, id, s_current, k_m):
    s_goal = graph.goal
    if id != s_goal:
        min_rhs = float('inf')
        for i in graph.graph[id].children:
            min_rhs = min(
                min_rhs, graph.graph[i].g + graph.graph[id].children[i])
        graph.graph[id].rhs = min_rhs

    # Use set for O(1) membership check instead of O(n) list comprehension
    # We use lazy deletion: just remove from set, stale entries in heap are skipped during pop
    if id in queue_set:
        queue_set.discard(id)

    if graph.graph[id].rhs != graph.graph[id].g:
        heapq.heappush(queue, calculateKey(graph, id, s_current, k_m) + (id,))
        queue_set.add(id)


def computeShortestPath(graph, queue, queue_set, s_start, k_m):
    while (graph.graph[s_start].rhs != graph.graph[s_start].g) or (topKey(queue) < calculateKey(graph, s_start, s_start, k_m)):
        k_old = topKey(queue)
        u = heapq.heappop(queue)[2]

        # Lazy deletion: skip stale entries that were already removed from queue_set
        if u not in queue_set:
            continue
        queue_set.discard(u)

        if k_old < calculateKey(graph, u, s_start, k_m):
            heapq.heappush(queue, calculateKey(graph, u, s_start, k_m) + (u,))
            queue_set.add(u)
        elif graph.graph[u].g > graph.graph[u].rhs:
            graph.graph[u].g = graph.graph[u].rhs
            for i in graph.graph[u].parents:
                updateVertex(graph, queue, queue_set, i, s_start, k_m)
        else:
            graph.graph[u].g = float('inf')
            updateVertex(graph, queue, queue_set, u, s_start, k_m)
            for i in graph.graph[u].parents:
                updateVertex(graph, queue, queue_set, i, s_start, k_m)


def nextInShortestPath(graph, s_current):
    min_rhs = float('inf')
    s_next = None

    # Try to find the next best move regardless of current rhs value
    # The rhs might be infinity during replanning, which is normal
    for i in graph.graph[s_current].children:
        # Skip infinite cost edges (blocked by obstacles)
        if graph.graph[s_current].children[i] == float('inf'):
            continue

        child_cost = graph.graph[i].g + graph.graph[s_current].children[i]
        if child_cost < min_rhs:
            min_rhs = child_cost
            s_next = i

    if s_next and min_rhs < float('inf'):
        return s_next
    else:
        # Only return "stuck" if we truly cannot find any valid move
        print('You are done stuck')
        return "stuck"


def scanForObstacles(graph, queue, queue_set, s_current, scan_range, k_m):
    states_to_update = {}
    range_checked = 0
    if scan_range >= 1:
        for neighbor in graph.graph[s_current].children:
            neighbor_coords = stateNameToCoords(neighbor)
            states_to_update[neighbor] = graph.cells[neighbor_coords[1]
                                                     ][neighbor_coords[0]]
        range_checked = 1

    while range_checked < scan_range:
        new_set = {}
        for state in states_to_update:
            new_set[state] = states_to_update[state]
            for neighbor in graph.graph[state].children:
                if neighbor not in new_set:
                    neighbor_coords = stateNameToCoords(neighbor)
                    new_set[neighbor] = graph.cells[neighbor_coords[1]
                                                    ][neighbor_coords[0]]
        range_checked += 1
        states_to_update = new_set

    new_obstacle = False
    for state in states_to_update:
        if states_to_update[state] < 0:  # found cell with obstacle
            for neighbor in graph.graph[state].children:
                # first time to observe this obstacle where one wasn't before
                if(graph.graph[state].children[neighbor] != float('inf')):
                    neighbor_coords = stateNameToCoords(state)
                    graph.cells[neighbor_coords[1]][neighbor_coords[0]] = -2
                    graph.graph[neighbor].children[state] = float('inf')
                    graph.graph[state].children[neighbor] = float('inf')
                    updateVertex(graph, queue, queue_set, state, s_current, k_m)
                    new_obstacle = True

    return new_obstacle


def moveAndRescan(graph, queue, queue_set, s_current, scan_range, k_m, occupancy, max_occupancy):
    if(s_current == graph.goal):
        return 'goal', k_m
    else:
        s_last = s_current

        # First, scan for obstacles and replan - this is the core of D* Lite
        results = scanForObstacles(graph, queue, queue_set, s_current, scan_range, k_m)
        k_m += heuristic_from_s(graph, s_last, s_current)
        computeShortestPath(graph, queue, queue_set, s_current, k_m)

        # Now try to find the next move after replanning
        s_new = nextInShortestPath(graph, s_current)
        if s_new == "stuck":
            return 'stuck', k_m

        new_coords = stateNameToCoords(s_new)

        # Check if the chosen move runs into a newly discovered obstacle
        if graph.cells[new_coords[1]][new_coords[0]] == OBS_VAL:
            s_new = s_current  # stay put if we would hit an obstacle
            print(f"Obstacle detected at {s_new}, staying at {s_current}")
        elif occupancy[new_coords[1]][new_coords[0]] > max_occupancy:
            s_new = s_current  # stay put if we would hit an obstacle
            print(f"Too many agnets at {s_new}, staying at {s_current}")

        return s_new, k_m


def initDStarLite(graph, queue, s_start, s_goal, k_m):
    queue_set = set()  # Initialize the queue membership tracking set
    graph.graph[s_goal].rhs = 0
    heapq.heappush(queue, calculateKey(
        graph, s_goal, s_start, k_m) + (s_goal,))
    queue_set.add(s_goal)
    computeShortestPath(graph, queue, queue_set, s_start, k_m)

    return (graph, queue, k_m, queue_set)
