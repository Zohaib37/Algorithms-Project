class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __str__(self):
        return f"({self.x}, {self.y})"
    
    def __repr__(self):
        return self.__str__()

class Polygon:
    def __init__(self, vertices):
        """
        Initialize a polygon with a list of vertices.
        
        Parameters:
        vertices (list): List of Point objects representing the vertices of the polygon
                        in clockwise or counter-clockwise order.
        """
        self.vertices = vertices
        self.n = len(vertices)
        
    def get_vertex(self, index):
        """Get vertex by index (1-indexed as in the paper)"""
        # Convert from 1-indexed (as in paper) to 0-indexed (for Python)
        return self.vertices[(index - 1) % self.n]
    
    def get_edge(self, i, j):
        """Get edge from vertex i to vertex j (1-indexed)"""
        return (self.get_vertex(i), self.get_vertex(j))
    
    def __str__(self):
        return f"Polygon with {self.n} vertices: {self.vertices}"

class VisibilityGraph:
    def __init__(self, polygon):
        """
        Initialize a visibility graph for a polygon.
        
        Parameters:
        polygon (Polygon): The polygon to build the visibility graph for
        """
        self.polygon = polygon
        self.n = polygon.n
        
        # Initialize adjacency lists for outgoing and incoming edges
        # We use 1-indexed arrays to match the paper's notation
        self.outgoing = [[] for _ in range(self.n + 1)]  # N⁺(v)
        self.incoming = [[] for _ in range(self.n + 1)]  # N⁻(v)
        
    def add_edge(self, i, j):
        """
        Add an edge to the visibility graph from vertex i to vertex j.
        Ensure i < j to maintain topological order.
        
        Parameters:
        i, j (int): 1-indexed vertex indices
        """
        if i < j:
            self.outgoing[i].append(j)
            self.incoming[j].append(i)
        else:
            self.outgoing[j].append(i)
            self.incoming[i].append(j)
    
    def sort_adjacency_lists(self):
        """Sort all adjacency lists in descending order"""
        for i in range(1, self.n + 1):
            self.outgoing[i].sort(reverse=True)
            self.incoming[i].sort(reverse=True)
    
    def is_visible(self, i, j):
        """
        Check if vertex j is in the outgoing list of vertex i or vice versa.
        
        Parameters:
        i, j (int): 1-indexed vertex indices
        
        Returns:
        bool: True if vertices i and j are visible to each other
        """
        if i < j:
            return j in self.outgoing[i]
        else:
            return i in self.outgoing[j]

class EdgeCrossingDetector:
    def __init__(self, polygon, candidate_edges):
        """
        Initialize the edge crossing detector.
        
        Parameters:
        polygon (Polygon): The polygon
        candidate_edges (list): List of (i,j) tuples representing edges in the visibility graph
                               where i,j are 1-indexed vertex indices
        """
        self.polygon = polygon
        self.n = polygon.n
        
        # Convert candidate edges to ensure i < j (topological order)
        self.candidate_edges = []
        for i, j in candidate_edges:
            if i < j:
                self.candidate_edges.append((i, j))
            else:
                self.candidate_edges.append((j, i))
        
        self.m = len(self.candidate_edges)
        
        # Create adjacency lists for outgoing and incoming edges
        self.outgoing = [[] for _ in range(self.n + 1)]  # N⁺(v)
        self.incoming = [[] for _ in range(self.n + 1)]  # N⁻(v)
        
        # Fill adjacency lists
        for i, j in self.candidate_edges:
            self.outgoing[i].append(j)
            self.incoming[j].append(i)
        
        # Sort adjacency lists in descending order
        for i in range(1, self.n + 1):
            self.outgoing[i].sort(reverse=True)
            self.incoming[i].sort(reverse=True)
        
        # Create doubly-linked list for non-empty adjacency lists
        self.next = [-1] * (self.n + 1)  # next[v] contains the next index u with N⁺(u) ≠ ∅
        self.prev = [-1] * (self.n + 1)  # prev[v] contains the previous index u with N⁺(u) ≠ ∅
        
        # Initialize the doubly-linked list
        self._init_linked_list()
    
    def _init_linked_list(self):
        """Initialize the doubly-linked list for non-empty outgoing adjacency lists"""
        non_empty = [i for i in range(1, self.n + 1) if self.outgoing[i]]
        
        if not non_empty:
            return
            
        # Set up the linked list
        for i in range(len(non_empty) - 1):
            self.next[non_empty[i]] = non_empty[i + 1]
            self.prev[non_empty[i + 1]] = non_empty[i]
    
    def compute_all_crossings(self):
        """
        Compute all crossings among the candidate edges.
        
        Returns:
        list: List of pairs ((a,b), (u,v)) representing crossing edges
        """
        crossings = []
        
        # Initialize an array to mark visited edges
        visited = set()
        
        # Iterate through vertices from left to right (increasing order)
        for u in range(1, self.n + 1):
            # Skip if no incoming edges
            if not self.incoming[u]:
                continue
            
            # Process each incoming edge (u,v)
            for v in self.incoming[u]:
                edge = (v, u) 
                
                if edge in visited:
                    continue
                    
                visited.add(edge)
                
                # Find all edges that cross with (v,u)
                # According to Observation 1: edges (a,b) with v < a < u < b will cross
                
                # Start from the first vertex after v
                a = v + 1
                while a < u:
                    # Skip if no outgoing edges
                    if not self.outgoing[a]:
                        a += 1
                        continue
                    
                    # Check outgoing edges from a
                    for b in self.outgoing[a]:
                        if b > u:  # This means v < a < u < b
                            # We've found a crossing between (v,u) and (a,b)
                            crossings.append(((v, u), (a, b)))
                    
                    a += 1
        
        return crossings
    
    def compute_all_crossings_optimized(self):
        """
        Compute all crossings using the optimized column-wise sweep algorithm.
        
        Returns:
        list: List of pairs ((a,b), (u,v)) representing crossing edges
        """
        crossings = []
        
        # For each column (vertex) v
        for v in range(1, self.n + 1):
            # Skip if no incoming edges
            if not self.incoming[v]:
                continue
            
            # For each row (vertex) u < v with an edge (u,v)
            for u in self.incoming[v]:
                # Get the rectangle [u+1, v-1] × [v+1, n]
                # All edges with source in [u+1, v-1] and target > v will cross (u,v)
                
                # Start with the first vertex after u
                curr = u + 1
                
                # Traverse through vertices between u and v
                while curr < v:
                    # For each outgoing edge (curr, t) where t > v
                    i = 0
                    # Use the outgoing list like a stack
                    while i < len(self.outgoing[curr]) and self.outgoing[curr][i] > v:
                        t = self.outgoing[curr][i]
                        # We found a crossing between (u,v) and (curr,t)
                        crossings.append(((u, v), (curr, t)))
                        i += 1
                    
                    curr += 1
        
        return crossings
    
    def compute_crossings_with_linked_list(self):
        """
        Compute all crossings using the doubly-linked list optimization.
        This is the algorithm described in the paper.
        
        Returns:
        list: List of pairs ((a,b), (u,v)) representing crossing edges
        """
        crossings = []
        
        # Simulate the column-wise sweep from left to right
        for v in range(1, self.n + 1):
            # Skip if no incoming edges
            if not self.incoming[v]:
                continue
            
            # For each row (vertex) u with an edge (u,v)
            for u in self.incoming[v]:
                # We need to find all edges (a,b) where u < a < v < b
                
                # Start with the first vertex after u that has outgoing edges
                a = u + 1
                while a < v:
                    if self.outgoing[a]:
                        # For each b where (a,b) is an edge and b > v
                        i = 0
                        while i < len(self.outgoing[a]) and self.outgoing[a][i] > v:
                            b = self.outgoing[a][i]
                            # We found a crossing between (u,v) and (a,b)
                            crossings.append(((u, v), (a, b)))
                            i += 1
                    
                    # Move to the next vertex with outgoing edges
                    a += 1
        
        return crossings

class CrossingComponent:
    def __init__(self, polygon, edge_crossings):
        """
        Initialize the crossing component computation.
        
        Parameters:
        polygon (Polygon): The input polygon
        edge_crossings (list): List of crossing edge pairs ((a,b), (u,v))
        """
        self.polygon = polygon
        self.n = polygon.n
        self.edge_crossings = edge_crossings
        
        # Initialize the pseudo-intersection graph GP
        # GP will be represented as an adjacency list
        self.GP = [[] for _ in range(self.n + 1)]  # 1-indexed
        
        # Initialize arrays for the doubly-linked list
        self.next = [-1] * (self.n + 1)
        self.prev = [-1] * (self.n + 1)
        
        # Initialize arrays for tracking visited edges
        self.visited = [False] * (self.n + 1)
        self.visited_edges = set()
        
        # Store the crossing components
        self.components = []
    
    def build_pseudo_intersection_graph(self):
        """
        Build the pseudo-intersection graph GP from the edge crossings.
        """
        # Sort edge crossings by column (v) in descending order
        sorted_crossings = sorted(self.edge_crossings, 
                                key=lambda x: x[1][1],  # Sort by v in (u,v)
                                reverse=True)
        
        # Process each crossing
        for (a, b), (u, v) in sorted_crossings:
            # Add edge to GP if it hasn't been visited
            if (a, b) not in self.visited_edges:
                self.GP[a].append(b)
                self.visited_edges.add((a, b))
            
            if (u, v) not in self.visited_edges:
                self.GP[u].append(v)
                self.visited_edges.add((u, v))
    
    def find_connected_components(self):
        """
        Find connected components in GP using BFS.
        """
        visited = set()
        
        for v in range(1, self.n + 1):
            if v not in visited and self.GP[v]:
                # Start a new component
                component = []
                queue = [v]
                visited.add(v)
                
                while queue:
                    current = queue.pop(0)
                    component.append(current)
                    
                    # Visit all neighbors
                    for neighbor in self.GP[current]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                
                # Sort component vertices in ascending order
                component.sort()
                self.components.append(component)
    
    def compute_crossing_components(self):
        """
        Compute all crossing components.
        
        Returns:
        list: List of crossing components, where each component is a list of vertices
        """
        # Build the pseudo-intersection graph
        self.build_pseudo_intersection_graph()
        
        # Find connected components
        self.find_connected_components()
        
        return self.components

# Helper function to test edge crossings
def detect_geometric_crossing(p1, p2, p3, p4):
    """
    Check if line segments (p1,p2) and (p3,p4) cross geometrically.
    
    Parameters:
    p1, p2, p3, p4 (Point): Endpoints of the two segments
    
    Returns:
    bool: True if the segments cross, False otherwise
    """
    # Compute the orientation of triplets
    def orientation(p, q, r):
        val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)
        if val == 0:
            return 0  # Collinear
        return 1 if val > 0 else 2  # Clockwise or Counterclockwise
    
    # Check if point q is on segment pr
    def on_segment(p, q, r):
        return (q.x <= max(p.x, r.x) and q.x >= min(p.x, r.x) and
                q.y <= max(p.y, r.y) and q.y >= min(p.y, r.y))
    
    # Find the four orientations
    o1 = orientation(p1, p2, p3)
    o2 = orientation(p1, p2, p4)
    o3 = orientation(p3, p4, p1)
    o4 = orientation(p3, p4, p2)
    
    # General case
    if o1 != o2 and o3 != o4:
        return True
    
    # Special cases
    if o1 == 0 and on_segment(p1, p3, p2):
        return True
    if o2 == 0 and on_segment(p1, p4, p2):
        return True
    if o3 == 0 and on_segment(p3, p1, p4):
        return True
    if o4 == 0 and on_segment(p3, p2, p4):
        return True
    
    return False

def test_edge_crossing_algorithm():
    # Create a simple polygon
    vertices = [
        Point(0, 0),
        Point(4, 0),
        Point(4, 4),
        Point(0, 4)
    ]
    polygon = Polygon(vertices)
    
    # Create some candidate edges that have crossings
    candidate_edges = [
        (1, 3),  # Edge from vertex 1 to vertex 3
        (2, 4)   # Edge from vertex 2 to vertex 4
    ]
    
    # These edges should cross
    
    # Create the edge crossing detector
    detector = EdgeCrossingDetector(polygon, candidate_edges)
    
    # Compute crossings
    crossings = detector.compute_all_crossings()
    print("Crossings using basic algorithm:", crossings)
    
    # Compute crossings using the optimized algorithm
    crossings_opt = detector.compute_all_crossings_optimized()
    print("Crossings using optimized algorithm:", crossings_opt)
    
    # Compute crossings using the linked list algorithm
    crossings_ll = detector.compute_crossings_with_linked_list()
    print("Crossings using linked list algorithm:", crossings_ll)
    
    # Verify the crossings geometrically
    for (a, b), (c, d) in crossings:
        p1 = polygon.get_vertex(a)
        p2 = polygon.get_vertex(b)
        p3 = polygon.get_vertex(c)
        p4 = polygon.get_vertex(d)
        
        cross = detect_geometric_crossing(p1, p2, p3, p4)
        print(f"Edges ({a},{b}) and ({c},{d}) cross geometrically: {cross}")

def test_crossing_component_algorithm():
    # Create a simple polygon
    vertices = [
        Point(0, 0),
        Point(4, 0),
        Point(4, 4),
        Point(0, 4)
    ]
    polygon = Polygon(vertices)
    
    # Create some candidate edges that have crossings
    candidate_edges = [
        (1, 3),  # Edge from vertex 1 to vertex 3
        (2, 4)   # Edge from vertex 2 to vertex 4
    ]
    
    # Create the edge crossing detector
    detector = EdgeCrossingDetector(polygon, candidate_edges)
    
    # Compute crossings
    crossings = detector.compute_all_crossings()
    print("Edge crossings:", crossings)
    
    # Create the crossing component computer
    component_computer = CrossingComponent(polygon, crossings)
    
    # Compute crossing components
    components = component_computer.compute_crossing_components()
    print("Crossing components:", components)
    
    # Verify the components
    for i, component in enumerate(components):
        print(f"Component {i+1}: {component}")
        # Verify that all vertices in the component are connected in GP
        for j in range(len(component)-1):
            v1, v2 = component[j], component[j+1]
            assert v2 in component_computer.GP[v1] or v1 in component_computer.GP[v2], \
                f"Vertices {v1} and {v2} should be connected in GP"

# Example usage
def create_example_polygon():
    # Create a simple polygon, like a pentagon
    vertices = [
        Point(0, 0),
        Point(2, 0),
        Point(3, 2),
        Point(1, 3),
        Point(-1, 1)
    ]
    return Polygon(vertices)

def create_example_visibility_graph(polygon):
    # Create a visibility graph for the polygon
    vis_graph = VisibilityGraph(polygon)
    
    # Add some example edges
    vis_graph.add_edge(1, 3)  # Edge from vertex 1 to vertex 3
    vis_graph.add_edge(1, 4)  # Edge from vertex 1 to vertex 4
    vis_graph.add_edge(2, 4)  # Edge from vertex 2 to vertex 4
    vis_graph.add_edge(2, 5)  # Edge from vertex 2 to vertex 5
    vis_graph.add_edge(3, 5)  # Edge from vertex 3 to vertex 5
    
    # Sort adjacency lists as required by the algorithm
    vis_graph.sort_adjacency_lists()
    
    return vis_graph

def test_polygon_representation():
    # Create and test polygon representation
    polygon = create_example_polygon()
    print(f"Created polygon with {polygon.n} vertices")
    
    # Test vertex access
    for i in range(1, polygon.n + 1):
        print(f"Vertex {i}: {polygon.get_vertex(i)}")
    
    # Create and test visibility graph
    vis_graph = create_example_visibility_graph(polygon)
    
    # Print outgoing edges for each vertex
    for i in range(1, polygon.n + 1):
        print(f"Outgoing edges from vertex {i}: {vis_graph.outgoing[i]}")
    
    # Print incoming edges for each vertex
    for i in range(1, polygon.n + 1):
        print(f"Incoming edges to vertex {i}: {vis_graph.incoming[i]}")
    
    # Test visibility check
    print(f"Vertex 2 is visible from vertex 3: {vis_graph.is_visible(2, 3)}")
    print(f"Vertex 2 is visible from vertex 4: {vis_graph.is_visible(2, 4)}")

def orientation(p, q, r):
    """
    Find orientation of ordered triplet (p, q, r).
    Returns:
    0: Collinear
    1: Clockwise
    2: Counterclockwise
    """
    val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)
    if val == 0:
        return 0
    return 1 if val > 0 else 2

def convex_hull(points):
    """
    Compute the convex hull of a set of points using Andrew's monotone chain algorithm.
    
    Parameters:
    points (list): List of Point objects
    
    Returns:
    list: List of Point objects forming the convex hull in counterclockwise order
    """
    n = len(points)
    if n < 3:
        return points
    
    # Sort points lexicographically
    points = sorted(points, key=lambda p: (p.x, p.y))
    
    # Build lower hull
    lower = []
    for p in points:
        while len(lower) >= 2 and orientation(lower[-2], lower[-1], p) != 2:
            lower.pop()
        lower.append(p)
    
    # Build upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and orientation(upper[-2], upper[-1], p) != 2:
            upper.pop()
        upper.append(p)
    
    # Concatenate and remove duplicates
    return lower[:-1] + upper[:-1]

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

class ShortcutHull:
    def __init__(self, polygon, lambda_param=0.5):
        """
        Initialize the Shortcut Hull computation.
        
        Parameters:
        polygon (Polygon): The input polygon
        lambda_param (float): Parameter for the cost function (0 ≤ λ ≤ 1)
        """
        self.polygon = polygon
        self.n = polygon.n
        self.vertices = polygon.vertices
        self.lambda_param = lambda_param
        
        # Compute convex hull
        self.convex_hull = convex_hull(self.vertices)
        
        # Initialize pockets
        self.pockets = []
        
        # Initialize shortcut hull
        self.shortcut_hull = None
        
        # Initialize crossing components
        self.crossing_components = None
        
        # Initialize DP tables
        self.dp = {}  # For memoization
        self.optimal_paths = {}  # For storing optimal paths
    
    def visualize(self, show_convex_hull=True, show_pockets=True, show_shortcut_hull=True):
        """
        Visualize the polygon, convex hull, pockets, and shortcut hull.
        
        Parameters:
        show_convex_hull (bool): Whether to show the convex hull
        show_pockets (bool): Whether to show the pockets
        show_shortcut_hull (bool): Whether to show the shortcut hull
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot the original polygon
        polygon_points = [(p.x, p.y) for p in self.vertices]
        polygon_patch = MplPolygon(polygon_points, fill=False, color='blue', label='Original Polygon')
        ax.add_patch(polygon_patch)
        
        # Plot the convex hull
        if show_convex_hull and self.convex_hull:
            hull_points = [(p.x, p.y) for p in self.convex_hull]
            hull_patch = MplPolygon(hull_points, fill=False, color='red', linestyle='--', label='Convex Hull')
            ax.add_patch(hull_patch)
        
        # Plot the pockets
        if show_pockets and self.pockets:
            for i, pocket in enumerate(self.pockets):
                pocket_points = [(self.polygon.get_vertex(j).x, self.polygon.get_vertex(j).y) for j in pocket]
                pocket_patch = MplPolygon(pocket_points, fill=True, color='green', alpha=0.2, label=f'Pocket {i+1}')
                ax.add_patch(pocket_patch)
        
        # Plot the shortcut hull
        if show_shortcut_hull and self.shortcut_hull:
            shortcut_points = [(p.x, p.y) for p in self.shortcut_hull]
            shortcut_patch = MplPolygon(shortcut_points, fill=False, color='purple', linewidth=2, label='Shortcut Hull')
            ax.add_patch(shortcut_patch)
        
        # Set plot properties
        ax.set_xlim(min(p.x for p in self.vertices) - 1, max(p.x for p in self.vertices) + 1)
        ax.set_ylim(min(p.y for p in self.vertices) - 1, max(p.y for p in self.vertices) + 1)
        ax.set_aspect('equal')
        ax.grid(True)
        ax.legend()
        
        plt.title(f'Shortcut Hull (λ = {self.lambda_param})')
        plt.show()
    
    def save_to_file(self, filename):
        """
        Save the shortcut hull to a file.
        
        Parameters:
        filename (str): The name of the file to save to
        """
        if not self.shortcut_hull:
            self.find_optimal_shortcut_hull()
        
        with open(filename, 'w') as f:
            # Write header
            f.write(f"# Shortcut Hull for polygon with {self.n} vertices\n")
            f.write(f"# Lambda parameter: {self.lambda_param}\n")
            f.write(f"# Number of vertices in shortcut hull: {len(self.shortcut_hull)}\n")
            f.write("# Format: x y\n")
            
            # Write vertices
            for p in self.shortcut_hull:
                f.write(f"{p.x} {p.y}\n")
    
    def print_statistics(self):
        """
        Print statistics about the shortcut hull.
        """
        if not self.shortcut_hull:
            self.find_optimal_shortcut_hull()
        
        # Compute statistics
        original_perimeter = self.compute_pocket_perimeter(self.vertices)
        shortcut_perimeter = self.compute_pocket_perimeter(self.shortcut_hull)
        original_area = self.compute_pocket_area(self.vertices)
        shortcut_area = self.compute_pocket_area(self.shortcut_hull)
        
        # Print statistics
        print("\nShortcut Hull Statistics:")
        print(f"Original polygon vertices: {self.n}")
        print(f"Shortcut hull vertices: {len(self.shortcut_hull)}")
        print(f"Perimeter reduction: {((original_perimeter - shortcut_perimeter) / original_perimeter * 100):.2f}%")
        print(f"Area reduction: {((original_area - shortcut_area) / original_area * 100):.2f}%")
        print(f"Cost (λ = {self.lambda_param}): {self.compute_cost(self.shortcut_hull):.2f}")

    def compute_pocket_area(self, pocket_vertices):
        """
        Compute the area of a pocket using the shoelace formula.
        """
        area = 0
        n = len(pocket_vertices)
        for i in range(n):
            j = (i + 1) % n
            area += pocket_vertices[i].x * pocket_vertices[j].y
            area -= pocket_vertices[j].x * pocket_vertices[i].y
        return abs(area) / 2
    
    def compute_pocket_perimeter(self, pocket_vertices):
        """
        Compute the perimeter of a pocket.
        """
        perimeter = 0
        n = len(pocket_vertices)
        for i in range(n):
            j = (i + 1) % n
            dx = pocket_vertices[j].x - pocket_vertices[i].x
            dy = pocket_vertices[j].y - pocket_vertices[i].y
            perimeter += (dx**2 + dy**2)**0.5
        return perimeter
    
    def compute_cost(self, path):
        """
        Compute the cost of a path using the cost function:
        c(Q) = λ·β(Q) + (1-λ)·α(Q)
        where β(Q) is perimeter and α(Q) is area
        """
        # Compute perimeter
        perimeter = 0
        for i in range(len(path) - 1):
            dx = path[i + 1].x - path[i].x
            dy = path[i + 1].y - path[i].y
            perimeter += (dx**2 + dy**2)**0.5
        
        # Compute area using shoelace formula
        area = 0
        for i in range(len(path)):
            j = (i + 1) % len(path)
            area += path[i].x * path[j].y
            area -= path[j].x * path[i].y
        area = abs(area) / 2
        
        # Compute final cost
        return self.lambda_param * perimeter + (1 - self.lambda_param) * area
    
    def find_optimal_shortcut_hull(self):
        """
        Compute the optimal shortcut hull using dynamic programming.
        """
        # First find all pockets
        self.find_pockets()
        
        # Initialize the shortcut hull with the convex hull
        self.shortcut_hull = self.convex_hull.copy()
        
        # Process each pocket
        for pocket in self.pockets:
            # Create a sub-polygon for the pocket
            pocket_vertices = [self.polygon.get_vertex(i) for i in pocket]
            pocket_polygon = Polygon(pocket_vertices)
            
            # Compute optimal path for the pocket
            if len(pocket) > 2:
                optimal_path = self._compute_optimal_path(pocket_polygon, 0, len(pocket) - 1)
                
                # Replace the pocket vertices in the shortcut hull with the optimal path
                start_idx = self.shortcut_hull.index(pocket_vertices[0])
                end_idx = self.shortcut_hull.index(pocket_vertices[-1])
                
                # Remove the pocket vertices
                del self.shortcut_hull[start_idx:end_idx + 1]
                
                # Insert the optimal path
                self.shortcut_hull[start_idx:start_idx] = optimal_path
        
        return self.shortcut_hull
    
    def _compute_optimal_path(self, polygon, i, j):
        """
        Compute the optimal path between vertices i and j using dynamic programming.
        
        Parameters:
        polygon (Polygon): The sub-polygon to process
        i, j (int): Start and end vertex indices
        
        Returns:
        list: List of points forming the optimal path
        """
        # Check if we've already computed this path
        if (i, j) in self.dp:
            return self.optimal_paths[(i, j)]
        
        # Base case: direct edge
        if j - i <= 1:
            path = [polygon.vertices[i], polygon.vertices[j]]
            self.dp[(i, j)] = self.compute_cost([polygon.vertices[i], polygon.vertices[j]])
            self.optimal_paths[(i, j)] = path
            return path
        
        # Initialize minimum cost and optimal path
        min_cost = float('inf')
        optimal_path = None
        
        # Try all possible intermediate points
        for k in range(i + 1, j):
            # Compute left and right subpaths
            left_path = self._compute_optimal_path(polygon, i, k)
            right_path = self._compute_optimal_path(polygon, k, j)
            
            # Combine paths
            combined_path = left_path[:-1] + right_path
            
            # Compute cost
            cost = self.compute_cost(combined_path)
            
            # Update if this is the best path so far
            if cost < min_cost:
                min_cost = cost
                optimal_path = combined_path
        
        # Store the result
        self.dp[(i, j)] = min_cost
        self.optimal_paths[(i, j)] = optimal_path
        
        return optimal_path
    
    def find_pockets(self):
        """
        Find pockets in the polygon.
        A pocket is a region enclosed by a convex hull segment and the polygon boundary.
        """
        # Get convex hull vertices in order
        hull_vertices = [self.vertices.index(p) + 1 for p in self.convex_hull]  # Convert to 1-indexed
        
        # Add the first vertex at the end to complete the cycle
        hull_vertices.append(hull_vertices[0])
        
        # Find pockets between consecutive hull vertices
        for i in range(len(hull_vertices) - 1):
            start = hull_vertices[i]
            end = hull_vertices[i + 1]
            
            # If the vertices are not consecutive in the polygon
            if (end - start) % self.n != 1:
                # Get the vertices in the pocket
                pocket = []
                current = start
                while current != end:
                    pocket.append(current)
                    current = (current % self.n) + 1  # Move to next vertex
                pocket.append(end)
                
                self.pockets.append(pocket)
    
    def compute_shortcut_hull(self):
        """
        Compute the shortcut hull for each pocket and combine them.
        """
        # First find all pockets
        self.find_pockets()
        
        # Initialize the shortcut hull with the convex hull
        self.shortcut_hull = self.convex_hull.copy()
        
        # Process each pocket
        for pocket in self.pockets:
            # Create a sub-polygon for the pocket
            pocket_vertices = [self.polygon.get_vertex(i) for i in pocket]
            pocket_polygon = Polygon(pocket_vertices)
            
            # Find optimal shortcut for the pocket
            if len(pocket) > 2:
                optimal_path = self.find_optimal_shortcut(pocket_polygon, 0, len(pocket) - 1)
                
                # Replace the pocket vertices in the shortcut hull with the optimal path
                start_idx = self.shortcut_hull.index(pocket_vertices[0])
                end_idx = self.shortcut_hull.index(pocket_vertices[-1])
                
                # Remove the pocket vertices
                del self.shortcut_hull[start_idx:end_idx + 1]
                
                # Insert the optimal path
                self.shortcut_hull[start_idx:start_idx] = optimal_path
        
        return self.shortcut_hull

def test_visualization():
    # Create a simple polygon (pentagon with a dent)
    vertices = [
        Point(0, 0),
        Point(4, 0),
        Point(4, 4),
        Point(2, 2),  # Dent
        Point(0, 4)
    ]
    polygon = Polygon(vertices)
    
    # Create the shortcut hull computer
    shortcut_hull = ShortcutHull(polygon, lambda_param=0.5)
    
    # Compute optimal shortcut hull
    shortcut_hull.find_optimal_shortcut_hull()
    
    # Visualize the results
    shortcut_hull.visualize()
    
    # Print statistics
    shortcut_hull.print_statistics()
    
    # Save to file
    shortcut_hull.save_to_file("shortcut_hull.txt")

if __name__ == "__main__":
    # test_polygon_representation()
    # test_edge_crossing_algorithm()
    # test_crossing_component_algorithm()
    # test_region_computation()
    # test_optimal_shortcut_hull()
    test_visualization()