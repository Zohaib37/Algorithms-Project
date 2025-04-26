import unittest
import matplotlib.pyplot as plt
from main import *
import time
import random
import math

def generate_regular_polygon(n, radius=10):
    """Generate a regular n-gon (e.g., circle) centered at origin"""
    return [Point(radius * math.cos(2 * math.pi * i / n),
                  radius * math.sin(2 * math.pi * i / n)) for i in range(n)]

def benchmark_shortcut_hull(n, lambda_param=0.5):
    """
    Generate a polygon of size n and measure runtime for shortcut hull computation.
    
    Returns:
    - Time taken
    - Number of shortcut hull vertices
    """
    # Create regular polygon
    vertices = generate_regular_polygon(n)
    polygon = Polygon(vertices)
    shortcut = ShortcutHull(polygon, lambda_param)

    # Measure computation time
    start = time.time()
    shortcut.find_optimal_shortcut_hull()
    end = time.time()

    duration = end - start
    return duration, len(shortcut.shortcut_hull)


class FixedPoint(Point):
    def __eq__(self, other):
        # Override equality to compare coordinates with tolerance
        if not isinstance(other, Point):
            return False
        return abs(self.x - other.x) < 1e-9 and abs(self.y - other.y) < 1e-9


class TestPolygonOperations(unittest.TestCase):
    def setUp(self):
        # Create square for basic tests
        self.square_vertices = [
            Point(0, 0),
            Point(4, 0),
            Point(4, 4),
            Point(0, 4)
        ]
        self.square = Polygon(self.square_vertices)
        
        # Create complex polygon for advanced tests
        self.complex_vertices = [
            Point(0, 0),
            Point(4, 0),
            Point(6, 2),
            Point(4, 4),
            Point(2, 3),
            Point(0, 4)
        ]
        self.complex = Polygon(self.complex_vertices)
    
    def test_polygon_initialization(self):
        """Test polygon initialization and basic properties"""
        self.assertEqual(len(self.square.vertices), 4)
        self.assertEqual(self.square.n, 4)
        self.assertEqual(len(self.complex.vertices), 6)
        self.assertEqual(self.complex.n, 6)
    
    def test_get_vertex(self):
        """Test vertex retrieval by index"""
        # Test 1-indexed access
        self.assertEqual(self.square.get_vertex(1), Point(0, 0))
        self.assertEqual(self.square.get_vertex(2), Point(4, 0))
        self.assertEqual(self.square.get_vertex(3), Point(4, 4))
        self.assertEqual(self.square.get_vertex(4), Point(0, 4))
        
        # Test wraparound
        self.assertEqual(self.square.get_vertex(5), Point(0, 0))
        self.assertEqual(self.square.get_vertex(0), Point(0, 4))
    
    def test_get_edge(self):
        """Test edge retrieval"""
        self.assertEqual(self.square.get_edge(1, 2), (Point(0, 0), Point(4, 0)))
        self.assertEqual(self.square.get_edge(4, 1), (Point(0, 4), Point(0, 0)))
    
    def test_visibility_graph(self):
        """Test visibility graph construction and operations"""
        vis_graph = VisibilityGraph(self.square)
        
        # Add diagonals
        vis_graph.add_edge(1, 3)  # Bottom-left to top-right
        vis_graph.add_edge(2, 4)  # Bottom-right to top-left
        
        # Sort adjacency lists
        vis_graph.sort_adjacency_lists()
        
        # Check outgoing edges
        self.assertEqual(vis_graph.outgoing[1], [3])
        self.assertEqual(vis_graph.outgoing[2], [4])
        self.assertEqual(vis_graph.outgoing[3], [])
        self.assertEqual(vis_graph.outgoing[4], [])
        
        # Check incoming edges
        self.assertEqual(vis_graph.incoming[1], [])
        self.assertEqual(vis_graph.incoming[2], [])
        self.assertEqual(vis_graph.incoming[3], [1])
        self.assertEqual(vis_graph.incoming[4], [2])
        
        # Test visibility
        self.assertTrue(vis_graph.is_visible(1, 3))
        self.assertTrue(vis_graph.is_visible(2, 4))
        self.assertFalse(vis_graph.is_visible(1, 4))
        self.assertFalse(vis_graph.is_visible(2, 3))

class TestEdgeCrossings(unittest.TestCase):
    def setUp(self):
        # Create square polygon
        self.square_vertices = [
            Point(0, 0),
            Point(4, 0),
            Point(4, 4),
            Point(0, 4)
        ]
        self.square = Polygon(self.square_vertices)
        
        # Define crossing diagonals
        self.candidate_edges = [
            (1, 3),  # Bottom-left to top-right
            (2, 4)   # Bottom-right to top-left
        ]
        
        self.detector = EdgeCrossingDetector(self.square, self.candidate_edges)
    
    def test_crossing_detection_basic(self):
        """Test the basic crossing detection algorithm"""
        crossings = self.detector.compute_all_crossings()
        
        # Should find one crossing
        self.assertEqual(len(crossings), 1)
        
        # Check crossing edges
        edge_pairs = [(tuple(sorted(a)), tuple(sorted(b))) for a, b in crossings]
        expected_crossing = ((1, 3), (2, 4))
        self.assertTrue(
            ((1, 3), (2, 4)) in edge_pairs or 
            ((2, 4), (1, 3)) in edge_pairs
        )
    
    def test_crossing_detection_optimized(self):
        """Test the optimized crossing detection algorithm"""
        crossings = self.detector.compute_all_crossings_optimized()
        
        # Should find one crossing
        self.assertEqual(len(crossings), 1)
        
        # Check crossing edges
        edge_pairs = [(tuple(sorted(a)), tuple(sorted(b))) for a, b in crossings]
        self.assertTrue(
            ((1, 3), (2, 4)) in edge_pairs or 
            ((2, 4), (1, 3)) in edge_pairs
        )
    
    def test_crossing_detection_linked_list(self):
        """Test the linked list crossing detection algorithm"""
        crossings = self.detector.compute_crossings_with_linked_list()
        
        # Should find one crossing
        self.assertEqual(len(crossings), 1)
        
        # Check crossing edges
        edge_pairs = [(tuple(sorted(a)), tuple(sorted(b))) for a, b in crossings]
        self.assertTrue(
            ((1, 3), (2, 4)) in edge_pairs or 
            ((2, 4), (1, 3)) in edge_pairs
        )
    
    def test_geometric_crossing(self):
        """Test geometric crossing detection function"""
        p1 = Point(0, 0)
        p2 = Point(4, 4)
        p3 = Point(0, 4)
        p4 = Point(4, 0)
        
        # Test crossing segments
        self.assertTrue(detect_geometric_crossing(p1, p2, p3, p4))
        
        # Test non-crossing segments
        p5 = Point(1, 1)
        p6 = Point(2, 2)
        self.assertFalse(detect_geometric_crossing(p1, p5, p3, p4))
        self.assertFalse(detect_geometric_crossing(p5, p6, p3, p4))

class TestCrossingComponents(unittest.TestCase):
    def setUp(self):
        # Create square polygon
        self.square_vertices = [
            Point(0, 0),
            Point(4, 0),
            Point(4, 4),
            Point(0, 4)
        ]
        self.square = Polygon(self.square_vertices)
        
        # Define crossing diagonals
        self.candidate_edges = [
            (1, 3),  # Bottom-left to top-right
            (2, 4)   # Bottom-right to top-left
        ]
        
        self.detector = EdgeCrossingDetector(self.square, self.candidate_edges)
        self.crossings = self.detector.compute_all_crossings()
    
    def test_crossing_component_construction(self):
        """Test crossing component construction"""
        component_computer = CrossingComponent(self.square, self.crossings)
        
        # Build pseudo-intersection graph
        component_computer.build_pseudo_intersection_graph()
        
        # Check graph edges
        self.assertTrue(len(component_computer.GP[1]) > 0 or len(component_computer.GP[3]) > 0)
        self.assertTrue(len(component_computer.GP[2]) > 0 or len(component_computer.GP[4]) > 0)
    
    def test_connected_components(self):
        """Test connected component finding"""
        component_computer = CrossingComponent(self.square, self.crossings)
        components = component_computer.compute_crossing_components()
        
        # Should find at least one component
        self.assertTrue(len(components) >= 1)
        
        # Collect vertices from all components
        involved_vertices = set()
        for component in components:
            involved_vertices.update(component)
        
        # Get vertices from crossings
        crossing_vertices = set()
        for (a, b), (c, d) in self.crossings:
            crossing_vertices.update([a, b, c, d])
        
        # All crossing vertices should be in components
        self.assertTrue(crossing_vertices.issubset(involved_vertices))

class TestShortcutHull(unittest.TestCase):
    def setUp(self):
        # Create polygon with dent
        self.polygon_vertices = [
            Point(0, 0),
            Point(4, 0),
            Point(4, 4),
            Point(2, 2),  # Dent
            Point(0, 4)
        ]
        self.polygon = Polygon(self.polygon_vertices)
        self.shortcut_hull_computer = ShortcutHull(self.polygon, lambda_param=0.5)
    
    def test_convex_hull(self):
        """Test convex hull computation"""
        # Should be square without dent
        hull = convex_hull(self.polygon_vertices)
        
        # Should have 4 vertices
        self.assertEqual(len(hull), 4)
        
        # Dent should not be in hull
        dent = Point(2, 2)
        hull_points = [(p.x, p.y) for p in hull]
        self.assertNotIn((dent.x, dent.y), hull_points)
    
    def test_find_pockets(self):
        """Test pocket finding"""
        self.shortcut_hull_computer.find_pockets()
        
        # Should find at least one pocket
        self.assertTrue(len(self.shortcut_hull_computer.pockets) >= 1)
    
    def test_shortcut_hull_computation(self):
        """Test shortcut hull computation"""
        hull = self.shortcut_hull_computer.find_optimal_shortcut_hull()
        
        # Hull should not be empty
        self.assertTrue(len(hull) >= 4)
        
        # Cost should be less than or equal to original
        original_cost = self.shortcut_hull_computer.compute_cost(self.polygon_vertices)
        hull_cost = self.shortcut_hull_computer.compute_cost(hull)
        
        # Print debug information
        print(f"\nOriginal polygon cost: {original_cost}")
        print(f"Shortcut hull cost: {hull_cost}")
        print(f"Original vertices: {self.polygon_vertices}")
        print(f"Hull vertices: {hull}")
        
        # The shortcut hull should not have higher cost than original
        # If lambda=0, should match convex hull (minimum perimeter)
        # If lambda=1, should match original polygon (minimum area loss)
        # For lambda=0.5, we need to ensure it's not worse than original
        self.assertLessEqual(hull_cost, original_cost)
        
class TestBenchmarking(unittest.TestCase):
    def test_runtime_scaling(self):
        """Test and record runtime for increasing polygon sizes"""
        sizes = [100, 200, 400, 800, 1000, 2000, 4000, 10000] 
        for n in sizes:
            duration, result_size = benchmark_shortcut_hull(n)
            print(f"n={n} | time={duration:.4f}s | shortcut_hull_vertices={result_size}")
            self.assertTrue(duration < 10, f"Algorithm took too long on n={n}")

def visualize_test_cases():
    """Visualize various test polygons and their shortcut hulls with cases inspired by the paper"""
    test_cases = [
        # Case 1: Polygon with multiple pockets creating separate crossing components
        [Point(0, 0), Point(5, 0), Point(10, 0), Point(10, 10), Point(8, 8), 
         Point(5, 10), Point(2, 8), Point(0, 10)],
        
        # Case 2: Star polygon with many potential crossing components
        [Point(5, 10), Point(4, 7), Point(1, 7), Point(3, 5), Point(2, 2),
         Point(5, 4), Point(8, 2), Point(7, 5), Point(9, 7), Point(6, 7)],
        
        # Case 3: Polygon with nested pockets (to test component hierarchy)
        [Point(0, 0), Point(10, 0), Point(10, 10), Point(0, 10), Point(1, 9),
         Point(9, 9), Point(9, 1), Point(8, 1), Point(8, 8), Point(2, 8),
         Point(2, 2), Point(7, 2), Point(7, 7), Point(3, 7), Point(3, 3),
         Point(6, 3), Point(6, 6), Point(1, 6)],
        
        # Case 4: Polygon similar to Figure 5 in the paper (with potential conflicting areas)
        [Point(0, 0), Point(10, 0), Point(10, 10), Point(8, 8), Point(9, 6),
         Point(7, 5), Point(9, 4), Point(8, 2), Point(6, 3), Point(5, 1),
         Point(4, 3), Point(2, 2), Point(1, 4), Point(3, 5), Point(1, 6),
         Point(2, 8), Point(0, 10)]
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    lambda_values = [0.1, 0.5, 0.9, 0.5]  # Different lambda values for different test cases
    
    for i, vertices in enumerate(test_cases):
        if i >= len(axes):
            break
            
        polygon = Polygon(vertices)
        
        # Use different lambda values to show their effect
        shortcut = ShortcutHull(polygon, lambda_param=lambda_values[i])
        
        try:
            # Compute hull
            hull = shortcut.find_optimal_shortcut_hull()
            
            # Plot original polygon
            x = [p.x for p in vertices]
            y = [p.y for p in vertices]
            x.append(x[0])  # Close polygon
            y.append(y[0])
            axes[i].plot(x, y, 'b-', label='Original')
            
            # Plot convex hull
            x_hull = [p.x for p in shortcut.convex_hull]
            y_hull = [p.y for p in shortcut.convex_hull]
            x_hull.append(x_hull[0])  # Close hull
            y_hull.append(y_hull[0])
            axes[i].plot(x_hull, y_hull, 'r--', label='Convex Hull')
            
            # Plot shortcut hull
            if shortcut.shortcut_hull:
                x_short = [p.x for p in shortcut.shortcut_hull]
                y_short = [p.y for p in shortcut.shortcut_hull]
                x_short.append(x_short[0])  # Close hull
                y_short.append(y_short[0])
                axes[i].plot(x_short, y_short, 'g-', linewidth=2, label=f'Shortcut Hull (λ={lambda_values[i]})')
            
            # Add small markers for vertices
            axes[i].scatter(x, y, color='blue', s=30, zorder=5)
            
            axes[i].set_title(f'Test Case {i+1}')
            axes[i].grid(True)
            axes[i].set_aspect('equal')
            axes[i].legend()
        except Exception as e:
            print(f"Error in test case {i+1}: {e}")
            axes[i].text(0.5, 0.5, f"Error: {e}", ha='center', va='center')
            axes[i].set_title(f'Test Case {i+1} (Failed)')
    
    plt.tight_layout()
    plt.savefig('shortcut_hull_tests.png')
    plt.show()

def visualize_crossing_components():
    """Visualize a polygon with its crossing components as described in the paper"""
    # Create polygon with complex crossing pattern
    vertices = [Point(0, 0), Point(10, 0), Point(10, 10), Point(0, 10)]
    polygon = Polygon(vertices)
    
    # Define crossing diagonals similar to Figure 7 in the paper
    candidate_edges = [
        (1, 3),  # Bottom-left to top-right
        (2, 4),  # Bottom-right to top-left
        (1, 4),  # Additional shortcuts to create complex crossing patterns
        (2, 3)
    ]
    
    # Create plot
    fig, axs = plt.subplots(1, 2, figsize=(15, 7))
    
    # Draw original polygon with candidates
    x = [p.x for p in vertices]
    y = [p.y for p in vertices]
    x.append(x[0])  # Close polygon
    y.append(y[0])
    axs[0].plot(x, y, 'b-', linewidth=2)
    
    # Draw shortcut candidates
    for edge in candidate_edges:
        p1 = polygon.get_vertex(edge[0])
        p2 = polygon.get_vertex(edge[1])
        axs[0].plot([p1.x, p2.x], [p1.y, p2.y], 'r--')
    
    axs[0].set_title("Polygon with Shortcut Candidates")
    axs[0].set_aspect('equal')
    axs[0].grid(True)
    
    # Compute and draw crossing components
    detector = EdgeCrossingDetector(polygon, candidate_edges)
    crossings = detector.compute_all_crossings()
    component_computer = CrossingComponent(polygon, crossings)
    components = component_computer.compute_crossing_components()
    
    # Draw polygon in second plot
    axs[1].plot(x, y, 'b-', linewidth=2)
    
    # Draw each crossing component in a different color
    colors = ['red', 'green', 'purple', 'orange', 'brown']
    for i, component in enumerate(components):
        color = colors[i % len(colors)]
        # Draw edges in this component
        for vertex in component:
            for other in component:
                if (vertex, other) in candidate_edges or (other, vertex) in candidate_edges:
                    p1 = polygon.get_vertex(vertex)
                    p2 = polygon.get_vertex(other)
                    axs[1].plot([p1.x, p2.x], [p1.y, p2.y], color=color, linewidth=2)
    
    axs[1].set_title(f"Crossing Components (Found: {len(components)})")
    axs[1].set_aspect('equal')
    axs[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('crossing_components.png')
    plt.show()
    
def visualize_lambda_effect():
    """Visualize the effect of lambda parameter on shortcut hull computation"""
    # Create a polygon with multiple pockets
    vertices = [
        Point(0, 0), Point(10, 0), Point(10, 3), Point(8, 2), 
        Point(10, 5), Point(10, 10), Point(8, 8), Point(6, 10),
        Point(4, 7), Point(2, 10), Point(0, 8), Point(0, 5),
        Point(2, 3), Point(0, 2)
    ]
    polygon = Polygon(vertices)
    
    # Create plot with different lambda values
    lambda_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    fig, axs = plt.subplots(1, len(lambda_values), figsize=(20, 5))
    
    # Plot original polygon
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
        
        axs[i].set_title(f'λ = {lambda_val}')
        axs[i].set_aspect('equal')
        axs[i].grid(True)
        axs[i].legend()
    
    plt.tight_layout()
    plt.savefig('lambda_effect.png')
    plt.show()


def run_all_tests_with_summary():
    """Run all unit tests and provide a summary of results"""
    print("🔍 Running comprehensive tests for polygon code...")

    # Create test suites
    polygon_suite = unittest.TestLoader().loadTestsFromTestCase(TestPolygonOperations)
    crossing_suite = unittest.TestLoader().loadTestsFromTestCase(TestEdgeCrossings)
    component_suite = unittest.TestLoader().loadTestsFromTestCase(TestCrossingComponents)
    shortcut_suite = unittest.TestLoader().loadTestsFromTestCase(TestShortcutHull)
    benchmark_suite = unittest.TestLoader().loadTestsFromTestCase(TestBenchmarking)

    # Combine all test suites
    all_suites = unittest.TestSuite([
        polygon_suite, crossing_suite, component_suite, shortcut_suite, benchmark_suite
    ])

    # Run tests
    result = unittest.TextTestRunner(verbosity=2).run(all_suites)

    # Print summary
    print("\n=== ✅ TEST SUMMARY ===")
    print(f"Total tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if not result.failures and not result.errors:
        print("\n✅ All tests passed successfully!")
        print("\n📊 Generating visualizations...")
        try:
            visualize_test_cases()
            print("✅ Visualizations completed successfully.")
        except Exception as e:
            print(f"❌ Visualization failed: {e}")
    else:
        print("\n⚠️ Some tests failed. Please fix the issues before proceeding.")

def run_all_tests_with_visualizations():
    """Run all tests and generate visualizations"""
    run_all_tests_with_summary()
    print("\n📊 Generating additional visualizations...")
    try:
        visualize_test_cases()
        visualize_crossing_components()
        visualize_lambda_effect()
        print("✅ All visualizations completed successfully.")
    except Exception as e:
        print(f"❌ Some visualizations failed: {e}")

 

if __name__ == "__main__":
    run_all_tests_with_visualizations()
