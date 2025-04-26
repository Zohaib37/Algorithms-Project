# Efficient Computation of Crossing Components and Shortcut Hulls

This project implements and extends the algorithm described in the paper "Efficient Computation of Crossing Components and Shortcut Hulls" by Nikolas Alexander Schwarz and Sabine Storandt (2024). The algorithm efficiently simplifies complex polygon boundaries by creating shortcut hulls that reduce complexity while preserving shape integrity.

## 📚 Project Overview

In computational geometry, polygon simplification is essential for applications ranging from computer graphics to spatial analysis. This implementation focuses on efficiently computing shortcut hulls - simplified polygons that contain the original polygon while preserving shape characteristics.

The algorithm achieves significant improvements over previous approaches:
- Reduces computational complexity from O(n⁴) to O(n²) for shortcut hull computation
- Uses novel edge crossing detection and pseudo-intersection graph techniques
- Introduces a component hierarchy approach for efficient region processing
- Balances perimeter and area optimization through a λ parameter

## 📋 Key Features

- **Edge Crossing Detection**: Efficient O(n + m + k) implementation using topological ordering
- **Crossing Component Computation**: Enhanced pseudo-intersection graph approach
- **Component Hierarchy Construction**: Tree-based representation of containment relationships
- **Shortcut Hull Optimization**: Dynamic programming with λ-based cost function
- **Visualization Tools**: Colored components, lambda-effect illustrations, and comparison views
- **Performance Optimization**: Sub-quadratic scaling with polygon size

```

## 📁 Project Structure
.
├── src/ 
│   └── main.py              # Main implementation of the algorithm
├── tests/
│   └── test_polygon.py      # Test file to verify correctness and visualize output
├── results/
│   └── *.png                # Visualizations of crossing components & shortcut hulls
├── checkpoint1/
│   ├── cp1_report.tex       # LaTeX source of the progress report
│   └── cp1_report.pdf       # Compiled PDF progress report
├── checkpoint2/
│   ├── cp2_report.tex       # LaTeX source of the progress report
│   └── cp2_report.pdf       # Compiled PDF progress report
├── checkpoint3/
│   ├── cp3_report.tex       # LaTeX source of the progress report
│   └── cp3_report.pdf       # Compiled PDF progress report
├── checkpoint4/
│   ├── cp4_report.tex       # LaTeX source of the final report
│   ├── cp4_report.pdf       # Compiled PDF final report
│   └── presentation.pdf     # Final presentation slides
└── README.md                # You're here!

```

---

## 🔍 How to Run

1. Install Dependencies:
   ```bash
   pip install matplotlib numpy shapely
2. Download both `main.py` (from the `src` folder) and `test_polygon.py` (from the `tests` folder).
3. Place both files in the **same folder**.
4. Run the test file using:
   ```bash
   python test_polygon.py

---

## 🧪 Testing

We use a mix of:

- **Unit tests** for core components  
- **Integration tests** for overall flow  
- **Visual validations** via `matplotlib`  

**Test cases include:**

- Squares with diagonals  
- Star-shaped polygons  
- Large random polygons (up to 10,000 vertices)  

---

## 📈 Benchmarking & Performance

- **Runtime:** ~4 seconds for polygons with 10,000 vertices  

**Comparisons made against:**

- Naive edge crossing detection (**13x slower**)  
- Greedy shortcut selection (**our method is 18–30% better in cost**)  
- Douglas-Peucker (**faster** but less optimized on our cost function)  

---

## 🚀 Enhancements

- Colored crossing component visualization  
- λ-effect illustrations on hull shapes  
- Improved handling of floating-point edge cases  
- Optimized pocket detection logic  

---


🔧 Enhancements
We've extended the original algorithm with:

Expanded Testing on Diverse Datasets:

Custom datasets with up to 10,000 vertices
Procedurally generated polygons with controlled complexity


Enhanced Visualization Tools:

Lambda-effect visualizer showing parameter impact
Component coloring for visual representation
Side-by-side comparison visualization


Parameter Exploration:

Systematic testing with λ values from 0 to 1
Analysis of polygon characteristics and λ interaction



⚠️ Known Limitations

Assumes simple polygons (no self-intersections or holes)
Fixed λ parameter requires re-running for different values
Memory requirements could be significant for very large polygons
The shortcut hull optimization has an unresolved issue that may produce valid but suboptimal hulls

🔮 Future Work

Extension to handle polygons with holes or self-intersections
Implementation of dynamic λ parameter adjustment
Further parallelization of algorithm components
Application to specific real-world GIS and graphics problems

📝 Citation
If you use this implementation in your research, please cite:
Nikolas Alexander Schwarz and Sabine Storandt. Efficient Computation of Crossing Components and Shortcut Hulls. 
In 35th International Workshop on Combinatorial Algorithms (IWOCA 2024), 2024.
👥 Contributors

Zohaib Aslam (za08134)
M. Mansoor Alam (ma08322)
Team 24


This README provides a comprehensive overview of your project, including setup instructions, features, testing methodology, performance analysis, and future directions. It maintains the style of your original README while incorporating the detailed information from your final report.
