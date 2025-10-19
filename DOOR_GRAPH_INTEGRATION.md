# Door Graph Integration Guide

## Overview

The door graph system provides **hierarchical pathfinding** for evacuation:
- **High-level:** Door-to-door navigation (strategic planning between rooms)
- **Low-level:** D* Lite on grid (tactical movement within rooms)

---

## Per-Agent vs Shared Graph

### Per-Agent Graphs (Recommended for Lazy Updates)

**Use Case:** Each agent discovers and updates their own knowledge of the environment.

```python
class EvacuationSimulation:
    def __init__(self, config):
        # Build base graph once (template)
        self.base_door_graph = build_door_graph(grid, door_configs)

class EvacuationAgent:
    def __init__(self, start_pos, targets, base_door_graph):
        import copy
        # Each agent gets their own copy
        self.door_graph = copy.deepcopy(base_door_graph)
```

**Pros:**
- True lazy updates (agents discover rooms independently)
- Realistic simulation (different agents have different knowledge)
- Simple logic (no coordination needed)

**Cons:**
- Memory overhead (small - ~10KB per agent)

---

## Integration Points

### 1. Initialization (Once)

```python
# In EvacuationSimulation.__init__()
from door_graph import build_door_graph

self.base_door_graph = build_door_graph(
    grid=self.initial_fire_map,
    door_configs=self.config.door_configs
)
```

---

### 2. Agent Initialization

```python
# In EvacuationAgent.__init__()
import copy
from door_graph import replan_path

self.door_graph = copy.deepcopy(base_door_graph)
self.door_path = None  # Will store [door_id1, door_id2, exit_id]
self.current_door_target = None

# Compute initial path
self.replan_door_path(grid)
```

---

### 3. High-Level Replanning

```python
# In EvacuationAgent
def replan_door_path(self, grid):
    """Find path through doors to any exit"""
    self.door_path = replan_path(
        graph=self.door_graph,
        start_pos=self.current_position,
        grid=grid
    )

    if self.door_path:
        # Set first door as D* Lite target
        next_door_id = self.door_path[0]
        self.current_door_target = self.door_graph.nodes[next_door_id].position
        self.reset_for_new_planning(self.current_door_target)
    else:
        self.status = "trapped"
```

**When to call:**
- Agent initialization
- When D* Lite returns "stuck"
- When reaching a door and fire blocks the next connection
- Optionally: Every N timesteps for optimization

---

### 4. Lazy Graph Updates (Per-Agent)

```python
# In EvacuationAgent
def on_reach_door(self, door_id):
    """Update graph when agent enters a new room"""
    # Get all doors in current room
    connected_doors = self.door_graph.get_connected_nodes_cached(grid, self.current_position)

    # Update edge weights based on current observations
    for i, (door_a_pos, _) in enumerate(connected_doors):
        for door_b_pos, _ in connected_doors[i+1:]:
            door_a_id = find_door_id_by_position(self.door_graph, door_a_pos)
            door_b_id = find_door_id_by_position(self.door_graph, door_b_pos)

            # Estimate weight based on current fire/smoke
            fire_penalty = self.estimate_room_danger()
            base_weight = self.door_graph.edges[(door_a_id, door_b_id)].base_distance
            new_weight = base_weight + fire_penalty

            self.door_graph.update_edge_weight(door_a_id, door_b_id, new_weight)

    # Invalidate cache for this room
    self.door_graph.clear_cache()
```

**When to call:**
- Agent reaches a door position
- Agent enters a room for the first time

---

### 5. Door-to-Door Progression

```python
# In EvacuationAgent.move()
def check_door_progress(self):
    """Check if agent reached current door target"""
    if self.current_position == self.current_door_target:
        current_door_id = self.door_path[0]

        # Update graph based on observations in this room
        self.on_reach_door(current_door_id)

        # Remove reached door from path
        self.door_path.pop(0)

        if not self.door_path:
            # Reached final exit
            self.status = "evacuated"
            return

        # Check if next connection is still valid
        next_door_id = self.door_path[0]
        edge_weight = self.door_graph.get_weight(current_door_id, next_door_id)

        if edge_weight == float('inf'):
            # Connection blocked, replan
            self.replan_door_path(grid)
        else:
            # Move to next door
            self.current_door_target = self.door_graph.nodes[next_door_id].position
            self.reset_for_new_planning(self.current_door_target)
```

---

## Execution Timeline

```
Simulation Start:
  └─ build_door_graph() → Create base template

Each Agent Init:
  ├─ Copy base graph
  └─ replan_path() → Get initial door sequence

Every Agent Timestep:
  ├─ D* Lite: Move toward current_door_target
  ├─ If reached door:
  │   ├─ Update graph edges (lazy observation)
  │   ├─ Pop door from path
  │   └─ Set next door as target
  └─ If stuck:
      └─ replan_path() → Find new door sequence

Agent Reaches Exit:
  └─ Mark evacuated
```

---

## Key Functions

| Function | Purpose | When to Use |
|----------|---------|-------------|
| `build_door_graph()` | Create door graph from config | Once at simulation start |
| `replan_path()` | Find path to any exit | Agent init, when stuck, periodically |
| `get_connected_nodes_cached()` | Find doors in current room | When updating graph, checking connectivity |
| `invalidate_cache_region()` | Clear stale cache | After fire spreads (global) |
| `clear_cache()` | Clear agent's cache | After updating edge weights (per-agent) |
| `update_edge_weight()` | Modify edge cost | When agent observes fire/smoke |
| `find_door_id_by_position()` | Convert position to door ID | When processing connected_nodes results |

---

## Configuration Format

Add to `SimulationConfig`:

```json
{
  "door_configs": [
    {"id": "d1", "position": "x15y3", "type": "door"},
    {"id": "d2", "position": "x25y3", "type": "door"},
    {"id": "exit1", "position": "x50y50", "type": "exit"}
  ]
}
```

---

## Example Flow

```python
# Agent starts at x10y5
agent.door_graph = deepcopy(base_graph)  # Initial: all edges have base_distance

# Plan path: x10y5 → [d1, d2, exit1]
agent.replan_door_path(grid)
agent.current_door_target = graph.nodes["d1"].position  # x15y3

# Navigate to d1 using D* Lite
# ... agent reaches x15y3 ...

# Update graph: "I'm in room 1, I see fire level 0.3"
agent.on_reach_door("d1")
# → Updates edges: d1↔d2 weight increases due to fire

# Move to next door
agent.current_door_target = graph.nodes["d2"].position  # x25y3

# ... continue until reaching exit1 ...
```

---

## Notes

- **Cache invalidation:** Call `clear_cache()` after updating edge weights
- **Memory:** Each agent's graph is ~10KB (negligible for 100s of agents)
- **Communication:** Can be added later by sharing observed edge weights between agents
- **Fire spread:** Global fire updates don't automatically update per-agent graphs (agents discover lazily)
