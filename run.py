"""
3D Support Calculator & Self-Learning Optimizer - Main Launcher.

Usage:
    # 1. Start Web Dashboard (default)
    python run.py

    # 2. Run Headless CLI Simulation & Optimizer
    python run.py --cli --model data/benchmark_models/overhang_torture.stl --iterations 30
"""

import argparse
import os
import sys
import webbrowser


def run_web_server(host: str = "127.0.0.1", port: int = 8000, open_browser: bool = True):
    import uvicorn
    from src.geometry.generate_benchmarks import generate_benchmark_models

    # Ensure benchmark models are generated
    generate_benchmark_models()

    url = f"http://{host}:{port}"
    print("=" * 65)
    print(" 3D Support Calculator & Self-Learning AI Optimizer")
    print(" Calibrated for Bambu Lab A1 (PLA, PETG, ABS, TPU)")
    print(f" Local Web Dashboard: {url}")
    print("=" * 65)

    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    uvicorn.run("src.server.app:app", host=host, port=port, reload=False, log_level="info")


def run_headless_cli(model_path: str, iterations: int = 25, material: str = "PLA"):
    from src.geometry.stl_parser import STLParser
    from src.geometry.overhang_analyzer import OverhangAnalyzer
    from src.ai.surrogate_model import NeuralSurrogateModel
    from src.ai.evolutionary_search import EvolutionarySupportOptimizer

    if not os.path.exists(model_path):
        print(f"Error: model file '{model_path}' not found.")
        sys.exit(1)

    print(f"Loading 3D model: {model_path}...")
    mesh = STLParser.load(model_path).center_on_bed()
    print(f"Triangles: {len(mesh.faces)}, Dimensions: {mesh.dimensions[0]:.1f}x{mesh.dimensions[1]:.1f}x{mesh.dimensions[2]:.1f} mm")

    analyzer = OverhangAnalyzer(mesh, material=material)
    report = analyzer.analyze()
    print(f"Overhang Area: {report['critical_overhang_area_mm2']:.1f} mm^2 ({report['critical_overhang_pct']:.1f}% of mesh)")

    surrogate = NeuralSurrogateModel(input_dim=12, hidden_dims=(64, 32), output_dim=4)
    optimizer = EvolutionarySupportOptimizer(mesh, surrogate, material=material)

    print(f"\nStarting continuous self-learning optimization ({iterations} iterations)...")
    for i in range(1, iterations + 1):
        step_res = optimizer.step_generation(population_size=6)
        rep = step_res["report"]
        chrom = step_res["chromosome"]
        print(
            f"Gen {i:02d} | Fitness: {rep.composite_fitness_score:5.1f} | "
            f"Filament Saved: {rep.filament_saved_vs_standard_grid_pct:4.1f}% | "
            f"Removal Score: {rep.removal_ease_score:3.1f}/10 | "
            f"Angle: {chrom['branch_angle_deg']:.1f}° | Merge: {chrom['merge_distance_mm']:.1f}mm"
        )

    best_rep = optimizer.best_report
    print("\n" + "=" * 65)
    print(" OPTIMIZATION SUMMARY (Best Discovered Support Topology)")
    print("=" * 65)
    print(f" Filament Saved vs Standard Grid: {best_rep.filament_saved_vs_standard_grid_pct:.1f}%")
    print(f" Filament Saved vs Default Tree:  {best_rep.filament_saved_vs_default_tree_pct:.1f}%")
    print(f" Support Mass:                    {best_rep.support_mass_grams:.2f} grams")
    print(f" Removal Ease Score:              {best_rep.removal_ease_score:.1f} / 10 ({best_rep.surface_scarring_risk})")
    print(f" Average Breakaway Force:         {best_rep.average_breakaway_force_n:.2f} N")
    print(f" Euler Buckling Safety Factor:    {best_rep.min_buckling_safety_factor:.1f}x")
    print(f" Overhang Coverage:               {best_rep.coverage_score_pct:.1f}%")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="3D Support Calculator & Self-Learning Optimizer")
    parser.add_argument("--cli", action="store_true", help="Run in headless CLI simulation mode")
    parser.add_argument("--model", type=str, default="data/benchmark_models/overhang_torture.stl", help="Path to 3D model STL")
    parser.add_argument("--iterations", type=int, default=20, help="Number of self-learning iterations for CLI")
    parser.add_argument("--material", type=str, default="PLA", choices=["PLA", "PETG", "ABS", "TPU"])
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Web server host")
    parser.add_argument("--port", type=int, default=8000, help="Web server port")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")

    args = parser.parse_args()

    if args.cli:
        run_headless_cli(args.model, args.iterations, args.material)
    else:
        run_web_server(args.host, args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
