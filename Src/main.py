import time
import matplotlib.pyplot as plt
from math import sqrt

import time
import matplotlib.pyplot as plt
from math import sqrt

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __str__(self):
        return f"({self.x}, {self.y})"
    
    def __repr__(self):
        return self.__str__()
     
    def __eq__(self, other):
        if not isinstance(other, Point):
            return False
        return self.x == other.x and self.y == other.y

class Polygon:
    def __init__(self, vertices):
        # Store vertices and count
        self.vertices = vertices
        self.n = len(vertices)
        
    def get_vertex(self, index):
        # Get vertex (1-indexed to match paper)
        return self.vertices[(index - 1) % self.n]
    
    def get_edge(self, i, j):
        # Get edge between two vertices
        return (self.get_vertex(i), self.get_vertex(j))
    
    def __str__(self):
        return f"Polygon with {self.n} vertices: {self.vertices}"

class VisibilityGraph:
    def __init__(self, polygon):
        # Create visibility graph for polygon
        self.polygon = polygon
        self.n = polygon.n
        
        # Init adjacency lists (1-indexed to match paper)
        self.outgoing = [[] for _ in range(self.n + 1)]  # N⁺(v)
        self.incoming = [[] for _ in range(self.n + 1)]  # N⁻(v)
        
    def add_edge(self, i, j):
        # Add edge ensuring i < j for topological order
        if i < j:
            self.outgoing[i].append(j)
            self.incoming[j].append(i)
        else:
            self.outgoing[j].append(i)
            self.incoming[i].append(j)
    
    def sort_adjacency_lists(self):
        # Sort in descending order
        for i in range(1, self.n + 1):
            self.outgoing[i].sort(reverse=True)
            self.incoming[i].sort(reverse=True)
    
    def is_visible(self, i, j):
        # Check if vertices can see each other
        if i < j:
            return j in self.outgoing[i]
        else:
            return i in self.outgoing[j]

class EdgeCrossingDetector:
    def __init__(self, polygon, candidate_edges):
        # Setup for detecting edge crossings
        self.polygon = polygon
        self.n = polygon.n
        
        # Normalize edges to ensure i < j
        self.candidate_edges = []
        for i, j in candidate_edges:
            if i < j:
                self.candidate_edges.append((i, j))
            else:
                self.candidate_edges.append((j, i))
        
        self.m = len(self.candidate_edges)
        
        # Create adjacency lists
        self.outgoing = [[] for _ in range(self.n + 1)]
        self.incoming = [[] for _ in range(self.n + 1)]
        
        # Fill adjacency lists
        for i, j in self.candidate_edges:
            self.outgoing[i].append(j)
            self.incoming[j].append(i)
        
        # Sort in descending order
        for i in range(1, self.n + 1):
            self.outgoing[i].sort(reverse=True)
            self.incoming[i].sort(reverse=True)
        
        # Setup doubly-linked list for non-empty lists
        self.next = [-1] * (self.n + 1)
        self.prev = [-1] * (self.n + 1)
        
        self._init_linked_list()
    
    def _init_linked_list(self):
        # Build linked list of vertices with outgoing edges
        non_empty = [i for i in range(1, self.n + 1) if self.outgoing[i]]
        
        if not non_empty:
            return
            
        # Link the vertices
        for i in range(len(non_empty) - 1):
            self.next[non_empty[i]] = non_empty[i + 1]
            self.prev[non_empty[i + 1]] = non_empty[i]
    
    def compute_all_crossings(self):
        # Basic algorithm to find all edge crossings
        crossings = []
        visited = set()
        
        # Check each vertex
        for u in range(1, self.n + 1):
            if not self.incoming[u]:
                continue
            
            # Check each incoming edge
            for v in self.incoming[u]:
                edge = (v, u) 
                
                if edge in visited:
                    continue
                    
                visited.add(edge)
                
                # Find edges that cross (v,u)
                # From Observation 1: edges (a,b) with v < a < u < b cross
                a = v + 1
                while a < u:
                    if not self.outgoing[a]:
                        a += 1
                        continue
                    
                    # Check outgoing edges
                    for b in self.outgoing[a]:
                        if b > u:  # v < a < u < b condition
                            crossings.append(((v, u), (a, b)))
                    
                    a += 1
        
        return crossings
    
    def compute_all_crossings_optimized(self):
        # Optimized column-wise sweep algorithm
        crossings = []
        
        # Check each column (vertex)
        for v in range(1, self.n + 1):
            if not self.incoming[v]:
                continue
            
            # Check each edge (u,v)
            for u in self.incoming[v]:
                # Check rectangle [u+1, v-1] × [v+1, n]
                curr = u + 1
                
                while curr < v:
                    # Find outgoing edges to vertices beyond v
                    i = 0
                    while i < len(self.outgoing[curr]) and self.outgoing[curr][i] > v:
                        t = self.outgoing[curr][i]
                        crossings.append(((u, v), (curr, t)))
                        i += 1
                    
                    curr += 1
        
        return crossings
    
    def compute_crossings_with_linked_list(self):
        # Algorithm using linked list optimization
        crossings = []
        
        # Column-wise sweep
        for v in range(1, self.n + 1):
            if not self.incoming[v]:
                continue
            
            # Check each edge (u,v)
            for u in self.incoming[v]:
                # Find edges (a,b) where u < a < v < b
                a = u + 1
                while a < v:
                    if self.outgoing[a]:
                        # Find outgoing edges beyond v
                        i = 0
                        while i < len(self.outgoing[a]) and self.outgoing[a][i] > v:
                            b = self.outgoing[a][i]
                            crossings.append(((u, v), (a, b)))
                            i += 1
                    
                    a += 1
        
        return crossings

class CrossingComponent:
    def __init__(self, polygon, edge_crossings):
        # Setup for finding connected components of crossing edges
        self.polygon = polygon
        self.n = polygon.n
        self.edge_crossings = edge_crossings
        
        # Pseudo-intersection graph
        self.GP = [[] for _ in range(self.n + 1)]
        
        # For doubly-linked list
        self.next = [-1] * (self.n + 1)
        self.prev = [-1] * (self.n + 1)
        
        # Track visited vertices and edges
        self.visited = [False] * (self.n + 1)
        self.visited_edges = set()
        
        # Components storage
        self.components = []
    
    def build_pseudo_intersection_graph(self):
        # Build graph from edge crossings
        sorted_crossings = sorted(self.edge_crossings, 
                              key=lambda x: x[1][1],
                              reverse=True)
        
        # Add edges to graph
        for (a, b), (u, v) in sorted_crossings:
            if (a, b) not in self.visited_edges:
                self.GP[a].append(b)
                self.visited_edges.add((a, b))
            
            if (u, v) not in self.visited_edges:
                self.GP[u].append(v)
                self.visited_edges.add((u, v))
    
    def find_connected_components(self):
        # Find components using BFS
        visited = set()
        
        for v in range(1, self.n + 1):
            if v not in visited and self.GP[v]:
                # Start new component
                component = []
                queue = [v]
                visited.add(v)
                
                while queue:
                    current = queue.pop(0)
                    component.append(current)
                    
                    # Visit neighbors
                    for neighbor in self.GP[current]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)
                
                # Sort vertices
                component.sort()
                self.components.append(component)
    
    def compute_crossing_components(self):
        # Get all connected components of crossing edges
        self.build_pseudo_intersection_graph()
        self.find_connected_components()
        return self.components

# Helper for geometric crossing detection
def detect_geometric_crossing(p1, p2, p3, p4):
    # Check if line segments cross geometrically
    
    # Skip if segments share endpoints
    if ((p1.x == p3.x and p1.y == p3.y) or 
        (p1.x == p4.x and p1.y == p4.y) or
        (p2.x == p3.x and p2.y == p3.y) or
        (p2.x == p4.x and p2.y == p4.y)):
        return False
    
    # Check if points are in counterclockwise order
    def ccw(A, B, C):
        return (C.y - A.y) * (B.x - A.x) > (B.y - A.y) * (C.x - A.x)
    
    # Test if each segment straddles the other's line
    return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)



# Example functions
def create_example_polygon():
    # Simple pentagon for testing
    vertices = [
        Point(0, 0),
        Point(2, 0),
        Point(3, 2),
        Point(1, 3),
        Point(-1, 1)
    ]
    return Polygon(vertices)

def create_example_visibility_graph(polygon):
    # Create visibility graph with sample edges
    vis_graph = VisibilityGraph(polygon)
    
    vis_graph.add_edge(1, 3)
    vis_graph.add_edge(1, 4)
    vis_graph.add_edge(2, 4)
    vis_graph.add_edge(2, 5)
    vis_graph.add_edge(3, 5)
    
    vis_graph.sort_adjacency_lists()
    
    return vis_graph



def orientation(p, q, r):
    # Determine orientation of triplet (p,q,r)
    # Returns: 0=collinear, 1=clockwise, 2=counterclockwise
    val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)
    if val == 0:
        return 0
    return 1 if val > 0 else 2

def convex_hull(points):
    # Andrew's monotone chain algorithm for convex hull
    n = len(points)
    if n < 3:
        return points
    
    # Sort by x, then y
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

    # Join hulls, removing duplicate end points
    return lower[:-1] + upper[:-1]


class ShortcutHull:
    def __init__(self, polygon, lambda_param=0.5):
        # Setup shortcut hull computation
        self.polygon = polygon
        self.n = polygon.n
        self.vertices = polygon.vertices
        self.lambda_param = lambda_param
        
        # Get convex hull
        self.convex_hull = convex_hull(self.vertices)
        
        # Storage
        self.pockets = []
        self.shortcut_hull = None
        self.dp = {}
        self.optimal_paths = {}
    
    def find_pockets(self):
        """Find regions between hull and polygon"""
        self.pockets = []
        
        # Match hull vertices to polygon vertices with tolerance
        hull_indices = []
        for hull_vertex in self.convex_hull:
            for i, poly_vertex in enumerate(self.vertices):
                if self._points_are_equal(hull_vertex, poly_vertex):
                    hull_indices.append(i + 1)  # Convert to 1-indexed
                    break
        
        # Sort hull indices in polygon order
        hull_indices.sort()
        hull_indices.append(hull_indices[0])  # Close the loop
        
        # Find pockets
        for i in range(len(hull_indices) - 1):
            start_idx = hull_indices[i]
            end_idx = hull_indices[i + 1]
            if (end_idx - start_idx) % self.n != 1:
                pocket = []
                current = start_idx
                while current != end_idx:
                    pocket.append(current)
                    current = current % self.n + 1
                pocket.append(end_idx)
                self.pockets.append(pocket)
        
    def _points_are_equal(self, p1, p2, tolerance=1e-6):
        """Compare points with tolerance for floating point precision"""
        return abs(p1.x - p2.x) < tolerance and abs(p1.y - p2.y) < tolerance
    
    def find_optimal_shortcut_hull(self):
        """Compute optimal shortcut hull with proper handling of pocket replacements"""
        # Find pockets between polygon and convex hull
        self.find_pockets()
        
        # Start with convex hull
        self.shortcut_hull = self.convex_hull.copy()
        
        # Process each pocket
        for pocket_idx, pocket in enumerate(self.pockets):
            if len(pocket) <= 2:
                continue  # Skip trivial pockets
                
            # Extract pocket vertices
            pocket_vertices = [self.polygon.get_vertex(i) for i in pocket]
            
            # Reset dynamic programming cache for each pocket
            self.dp = {}
            self.optimal_paths = {}
            
            # Create a polygon from the pocket vertices
            pocket_polygon = Polygon(pocket_vertices)
            
            # Compute optimal path for this pocket
            start_idx = 0
            end_idx = len(pocket_vertices) - 1
            
            # Always consider the direct path as a baseline
            direct_path = [pocket_vertices[start_idx], pocket_vertices[end_idx]]
            direct_cost = self.compute_cost(direct_path)
            
            # Compute optimal path
            optimal_path = self._compute_optimal_path(pocket_polygon, start_idx, end_idx)
            optimal_cost = self.compute_cost(optimal_path)
            
            # If optimal path is not better than direct path, just use direct path
            if optimal_cost >= direct_cost:
                optimal_path = direct_path
            
            # Find indices of pocket endpoints in hull
            start_point = pocket_vertices[0]
            end_point = pocket_vertices[-1]
            
            start_idx_in_hull = self._find_point_index_in_hull(start_point)
            end_idx_in_hull = self._find_point_index_in_hull(end_point)
            
            if start_idx_in_hull is None or end_idx_in_hull is None:
                continue
            
            # Create new hull by replacing segment
            new_hull = []
            
            if end_idx_in_hull < start_idx_in_hull:  # Wrap around case
                # Add vertices from end_idx+1 to start_idx-1
                new_hull.extend(self.shortcut_hull[end_idx_in_hull + 1:start_idx_in_hull])
                # Add optimal path
                new_hull.extend(optimal_path)
            else:
                # Add vertices before start_idx
                new_hull.extend(self.shortcut_hull[:start_idx_in_hull])
                # Add optimal path
                new_hull.extend(optimal_path)
                # Add vertices after end_idx
                new_hull.extend(self.shortcut_hull[end_idx_in_hull+1:])
            
            # Check if new hull is better than current hull
            if self.compute_cost(new_hull) <= self.compute_cost(self.shortcut_hull):
                self.shortcut_hull = new_hull
        
        return self.shortcut_hull
    
    def _find_point_index_in_hull(self, point, tolerance=1e-9):
        """Find index of point in shortcut hull with tolerance"""
        for i, p in enumerate(self.shortcut_hull):
            if self._points_are_equal(p, point, tolerance):
                return i
        return None
    
    def _compute_optimal_path(self, polygon, i, j):
        """Compute optimal path using dynamic programming"""
        # Check memo table
        if (i, j) in self.dp:
            return self.optimal_paths[(i, j)].copy()  # Return a copy to avoid mutation
        
        # Base case: direct edge
        if j - i <= 1:
            path = [polygon.vertices[i], polygon.vertices[j]]
            self.dp[(i, j)] = self.compute_cost(path)
            self.optimal_paths[(i, j)] = path
            return path.copy()
        
        # Initialize with direct path as default
        direct_path = [polygon.vertices[i], polygon.vertices[j]]
        direct_cost = self.compute_cost(direct_path)
        min_cost = direct_cost
        optimal_path = direct_path.copy()
        
        # Try all intermediate vertices
        for k in range(i + 1, j):
            # Get subpaths
            left_path = self._compute_optimal_path(polygon, i, k)
            right_path = self._compute_optimal_path(polygon, k, j)
            
            # Combine paths (avoiding duplicate vertex at k)
            combined_path = left_path[:-1] + right_path
            
            # Compute cost
            cost = self.compute_cost(combined_path)
            
            # Update if better
            if cost < min_cost:
                min_cost = cost
                optimal_path = combined_path.copy()
        
        # Save results
        self.dp[(i, j)] = min_cost
        self.optimal_paths[(i, j)] = optimal_path.copy()
        
        return optimal_path.copy()
    
    def compute_cost(self, path):
        """Cost function: λ·perimeter + (1-λ)·area"""
        # Calculate perimeter
        perimeter = 0
        for i in range(len(path) - 1):  # Don't use modulo for open paths
            dx = path[i+1].x - path[i].x
            dy = path[i+1].y - path[i].y
            perimeter += sqrt(dx**2 + dy**2)
        
        # For paths with only 2 vertices, there's no area
        if len(path) < 3:
            # Special case for λ = 0 (pure area minimization)
            if self.lambda_param == 0:
                # Add a small penalty for the direct path to encourage intermediate vertices
                return 0.1 * perimeter  # Small non-zero cost
            return self.lambda_param * perimeter
        
        # Calculate area using shoelace formula (for closed paths)
        area = 0
        for i in range(len(path) - 1):
            area += path[i].x * path[i+1].y
            area -= path[i+1].x * path[i].y
        
        # Close the path back to first vertex for area calculation
        area += path[-1].x * path[0].y
        area -= path[0].x * path[-1].y
        
        area = abs(area) / 2
        
        # Calculate weighted cost
        cost = self.lambda_param * perimeter + (1 - self.lambda_param) * area
        
        return cost
        
def test_fixed_shortcut_hull():
    """Test the fixed shortcut hull implementation with various lambda values"""
    # Create a polygon with very deep pockets for better visualization
    vertices = [
        Point(0, 0), Point(10, 0), Point(10, 1), 
        Point(2, 2),  # Deep indent
        Point(10, 3), Point(10, 7), 
        Point(2, 5),  # Another deep indent
        Point(10, 8), Point(10, 10), Point(0, 10),
        Point(1, 5),  # Side indent
        Point(0, 1)  # Return to near start
    ]
    
    polygon = Polygon(vertices)
    
    # Test with different lambda values
    lambda_values = [0.01, 0.25, 0.5, 0.75, 0.99]
    fig, axs = plt.subplots(1, len(lambda_values), figsize=(20, 5))
    
    # Original polygon points
    x = [p.x for p in vertices]
    y = [p.y for p in vertices]
    x.append(x[0])  # Close polygon
    y.append(y[0])
    
    for i, lambda_val in enumerate(lambda_values):
        # Plot original polygon
        axs[i].plot(x, y, 'b-', linewidth=1, label='Original')
        
        # Compute shortcut hull with this lambda
        shortcut = ShortcutHull(polygon, lambda_param=lambda_val)
        hull = shortcut.find_optimal_shortcut_hull()
        
        # Plot shortcut hull
        if hull:
            x_short = [p.x for p in hull]
            y_short = [p.y for p in hull]
            x_short.append(x_short[0])  # Close hull
            y_short.append(y_short[0])
            axs[i].plot(x_short, y_short, 'g-', linewidth=2, label=f'λ={lambda_val}')
            
            # Plot convex hull too
            x_conv = [p.x for p in shortcut.convex_hull]
            y_conv = [p.y for p in shortcut.convex_hull]
            x_conv.append(x_conv[0])  # Close hull
            y_conv.append(y_conv[0])
            axs[i].plot(x_conv, y_conv, 'r--', label='Convex Hull')
        
        axs[i].set_title(f'λ = {lambda_val}')
        axs[i].set_aspect('equal')
        axs[i].grid(True)
        axs[i].legend()
    
    plt.tight_layout()
    plt.savefig('fixed_lambda_effect.png')
    plt.show()
         
if __name__ == "__main__":

    #Create shortcut hull and test
    test_fixed_shortcut_hull()
