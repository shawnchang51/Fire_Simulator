from .graph import Node, Graph


class GridWorld(Graph):
    def __init__(self, x_dim, y_dim, connect8=True, fire_fearness=1.0):
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.fire_fearness = fire_fearness
        # First make an element for each row (height of grid)
        self.cells = [0] * y_dim
        # Go through each element and replace with row (width of grid)
        for i in range(y_dim):
            self.cells[i] = [0] * x_dim

        self.occupancy = [0] * y_dim
        # Go through each element and replace with row (width of grid)
        for i in range(y_dim):
            self.occupancy[i] = [0] * x_dim

        # will this be an 8-connected graph or 4-connected?
        self.connect8 = connect8
        self.graph = {}

        self.generateGraphFromGrid()
        # self.printGrid()

    def __str__(self):
        msg = 'Graph:'
        for i in self.graph:
            msg += '\n  node: ' + i + ' g: ' + \
                str(self.graph[i].g) + ' rhs: ' + str(self.graph[i].rhs) + \
                ' neighbors: ' + str(self.graph[i].children)
        return msg

    def __repr__(self):
        return self.__str__()

    def printGrid(self):
        print('** GridWorld **')
        for row in self.cells:
            print(row)

    def printGValues(self):
        for j in range(self.y_dim):
            str_msg = ""
            for i in range(self.x_dim):
                node_id = 'x' + str(i) + 'y' + str(j)
                node = self.graph[node_id]
                if node.g == float('inf'):
                    str_msg += ' - '
                else:
                    str_msg += ' ' + str(node.g) + ' '
            print(str_msg)

    def getTerrainCost(self, cell_value):
        """Calculate terrain cost based on cell value and fire fearness"""
        if cell_value == -5:  # impassable obstacle
            return float('inf')
        elif cell_value < 0:
            return abs(cell_value) * 2  # other negative values as difficult terrain
        else:
            base_cost = max(1, cell_value + 1)  # positive values: 0->1, 1->2, 2->3, etc.
            # Apply fearness multiplier to fire (cell_value > 0)
            return base_cost * self.fire_fearness if cell_value > 0 else base_cost

    def generateGraphFromGrid(self):
        import math

        for i in range(len(self.cells)):
            row = self.cells[i]
            for j in range(len(row)):
                node = Node('x' + str(j) + 'y' + str(i))
                current_cost = self.getTerrainCost(self.cells[i][j])

                # 4-connected neighbors (horizontal/vertical)
                directions = []
                if i > 0:  # top
                    directions.append((i-1, j, 1.0))
                if i + 1 < self.y_dim:  # bottom
                    directions.append((i+1, j, 1.0))
                if j > 0:  # left
                    directions.append((i, j-1, 1.0))
                if j + 1 < self.x_dim:  # right
                    directions.append((i, j+1, 1.0))

                # 8-connected neighbors (add diagonals)
                if self.connect8:
                    diagonal_cost = math.sqrt(2)  # √2 ≈ 1.414 for diagonal movement
                    if i > 0 and j > 0:  # top-left
                        directions.append((i-1, j-1, diagonal_cost))
                    if i > 0 and j + 1 < self.x_dim:  # top-right
                        directions.append((i-1, j+1, diagonal_cost))
                    if i + 1 < self.y_dim and j > 0:  # bottom-left
                        directions.append((i+1, j-1, diagonal_cost))
                    if i + 1 < self.y_dim and j + 1 < self.x_dim:  # bottom-right
                        directions.append((i+1, j+1, diagonal_cost))

                # Add all valid neighbors
                for ni, nj, base_multiplier in directions:
                    neighbor_cost = self.getTerrainCost(self.cells[ni][nj])
                    edge_cost = max(current_cost, neighbor_cost) * base_multiplier
                    neighbor_id = 'x' + str(nj) + 'y' + str(ni)
                    node.parents[neighbor_id] = edge_cost
                    node.children[neighbor_id] = edge_cost

                self.graph['x' + str(j) + 'y' + str(i)] = node

    def updateGraphFromTerrain(self):
        """Update graph edge costs after terrain changes"""
        self.generateGraphFromGrid()

    def reset_for_new_planning(self):
        """Reset D* Lite algorithm state for new start/goal planning

        Returns:
            tuple: (queue, k_m) - Fresh queue and k_m value for new planning
        """
        # Reset all nodes' g and rhs values to infinity
        for node_id in self.graph:
            self.graph[node_id].g = float('inf')
            self.graph[node_id].rhs = float('inf')

        # Return fresh queue and k_m value
        return [], 0
