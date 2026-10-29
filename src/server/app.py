"""
FastAPI Full-Stack Application for 3D Support Calculator.
Provides REST APIs for 3D model parsing (STL, OBJ, 3MF), overhang analysis,
instant support generation, live continuous self-learning optimizer streaming,
and Bambu Studio / OrcaSlicer project export.
"""

import os
import json
import io
import asyncio
from typing import Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from ..physics.constants import BAMBU_A1_PROFILE, MATERIAL_DATABASE
from ..physics.precalculated import lookup_optimal_z_gap
from ..geometry.mesh import TriangleMesh
from ..geometry.stl_parser import STLParser
from ..geometry.obj_parser import OBJParser
from ..geometry.bambu_3mf import Bambu3MFParser
from ..geometry.overhang_analyzer import OverhangAnalyzer
from ..generator.path_routing import BranchRouter
from ..generator.mesh_builder import SupportMeshBuilder
from ..simulator.physics_evaluator import SupportPhysicsEvaluator
from ..ai.continuous_engine import ContinuousLearningEngine


app = FastAPI(
    title="3D Support Calculator & Self-Learning Optimizer",
    description="Physical simulation and self-training AI for FDM 3D printing supports",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engine state
current_mesh: Optional[TriangleMesh] = None
current_support_mesh: Optional[TriangleMesh] = None
current_support_graph = None
current_model_filename = "default_model.stl"
current_metadata: Dict[str, Any] = {
    "material": "PLA",
    "layer_height": 0.20,
    "priority_profile": "balanced",
}

continuous_engine = ContinuousLearningEngine()


# Ensure directories exist
os.makedirs("data/benchmark_models", exist_ok=True)
os.makedirs("temp_uploads", exist_ok=True)


@app.get("/api/health")
def get_health():
    return {
        "status": "healthy",
        "printer": BAMBU_A1_PROFILE.name,
        "engine_running": continuous_engine.is_running,
        "total_iterations": continuous_engine.total_iterations,
        "active_model": current_model_filename if current_mesh else None,
    }


@app.get("/api/presets")
def get_presets():
    materials = {
        name: {
            "name": mat.name,
            "density": mat.density,
            "elastic_modulus_gpa": mat.elastic_modulus,
            "tensile_strength_mpa": mat.tensile_strength,
            "print_temp": mat.print_temp,
            "bed_temp": mat.bed_temp,
            "safe_overhang_base_deg": mat.safe_overhang_base_deg,
            "optimal_z_gap_020": lookup_optimal_z_gap(name, 0.20),
        }
        for name, mat in MATERIAL_DATABASE.items()
    }
    return {
        "printer": {
            "name": BAMBU_A1_PROFILE.name,
            "bed_x": BAMBU_A1_PROFILE.bed_x,
            "bed_y": BAMBU_A1_PROFILE.bed_y,
            "bed_z": BAMBU_A1_PROFILE.bed_z,
            "default_nozzle": BAMBU_A1_PROFILE.default_nozzle,
            "max_speed": BAMBU_A1_PROFILE.max_print_speed,
        },
        "materials": materials,
        "layer_heights": [0.08, 0.12, 0.16, 0.20, 0.24, 0.28],
        "priorities": ["balanced", "filament_saver", "clean_removal", "max_stability"],
    }


@app.get("/api/benchmarks")
def list_benchmarks():
    bench_dir = "data/benchmark_models"
    models = []
    if os.path.exists(bench_dir):
        for f in os.listdir(bench_dir):
            if f.endswith(".stl"):
                models.append({
                    "id": f,
                    "name": f.replace(".stl", "").replace("_", " ").title(),
                    "filename": f,
                })
    return {"models": models}


@app.post("/api/benchmarks/load/{model_id}")
def load_benchmark(model_id: str):
    global current_mesh, current_model_filename, current_support_mesh
    filepath = os.path.join("data/benchmark_models", model_id)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Benchmark model not found")

    mesh = STLParser.load(filepath)
    # Center on bed
    current_mesh = mesh.center_on_bed()
    current_model_filename = model_id
    current_support_mesh = None

    continuous_engine.set_active_mesh(
        current_mesh,
        material=current_metadata.get("material", "PLA"),
        layer_height=current_metadata.get("layer_height", 0.20),
        priority_profile=current_metadata.get("priority_profile", "balanced"),
    )

    analyzer = OverhangAnalyzer(current_mesh, material=current_metadata.get("material", "PLA"))
    analysis = analyzer.analyze()

    return {
        "status": "loaded",
        "filename": model_id,
        "vertex_count": len(current_mesh.vertices),
        "face_count": len(current_mesh.faces),
        "dimensions": [round(float(d), 2) for d in current_mesh.dimensions],
        "surface_area_mm2": round(current_mesh.compute_surface_area(), 2),
        "volume_cm3": round(current_mesh.compute_volume() / 1000.0, 3),
        "overhang_analysis": {
            "critical_overhang_pct": round(analysis["critical_overhang_pct"], 1),
            "critical_face_count": analysis["critical_face_count"],
            "bridge_face_count": analysis["bridge_face_count"],
            "safe_threshold_deg": analysis["critical_angle_threshold_deg"],
        },
    }


@app.post("/api/upload")
async def upload_model(
    file: UploadFile = File(...),
    material: str = Form("PLA"),
    layer_height: float = Form(0.20),
    priority: str = Form("balanced"),
):
    global current_mesh, current_model_filename, current_support_mesh, current_metadata
    content = await file.read()
    filename = file.filename.lower()

    try:
        if filename.endswith(".stl"):
            mesh = STLParser.load(content)
        elif filename.endswith(".obj"):
            mesh = OBJParser.load(content)
        elif filename.endswith(".3mf"):
            mesh, meta = Bambu3MFParser.load(content)
            if "material" in meta:
                material = meta["material"]
            if "layer_height" in meta:
                layer_height = meta["layer_height"]
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Please upload .stl, .obj, or .3mf")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse 3D file: {str(e)}")

    # Center on bed Z=0
    current_mesh = mesh.center_on_bed()
    current_model_filename = file.filename
    current_support_mesh = None
    current_metadata = {
        "material": material,
        "layer_height": layer_height,
        "priority_profile": priority,
    }

    # Register with self-learning engine
    continuous_engine.set_active_mesh(
        current_mesh,
        material=material,
        layer_height=layer_height,
        priority_profile=priority,
    )

    analyzer = OverhangAnalyzer(current_mesh, material=material, layer_height=layer_height)
    analysis = analyzer.analyze()

    return {
        "status": "success",
        "filename": file.filename,
        "vertex_count": len(current_mesh.vertices),
        "face_count": len(current_mesh.faces),
        "dimensions": [round(float(d), 2) for d in current_mesh.dimensions],
        "surface_area_mm2": round(current_mesh.compute_surface_area(), 2),
        "volume_cm3": round(current_mesh.compute_volume() / 1000.0, 3),
        "overhang_analysis": {
            "critical_overhang_pct": round(analysis["critical_overhang_pct"], 1),
            "critical_face_count": analysis["critical_face_count"],
            "bridge_face_count": analysis["bridge_face_count"],
            "safe_threshold_deg": analysis["critical_angle_threshold_deg"],
        },
    }


@app.post("/api/generate")
def generate_support(
    branch_angle_deg: float = Query(28.0),
    merge_distance_mm: float = Query(9.0),
    tip_diameter_mm: float = Query(0.70),
    grid_res_mm: float = Query(3.4),
    material: Optional[str] = Query(None),
    layer_height: Optional[float] = Query(None),
    priority: Optional[str] = Query(None),
):
    global current_mesh, current_support_mesh, current_support_graph, current_metadata
    if current_mesh is None:
        raise HTTPException(status_code=400, detail="No 3D model loaded. Please upload a model or load a benchmark.")

    if material:
        current_metadata["material"] = material
    if layer_height:
        current_metadata["layer_height"] = layer_height
    if priority:
        current_metadata["priority_profile"] = priority

    mat = current_metadata.get("material", "PLA")
    l_height = current_metadata.get("layer_height", 0.20)
    prio = current_metadata.get("priority_profile", "balanced")

    analyzer = OverhangAnalyzer(current_mesh, material=mat, layer_height=l_height)
    contact_points = analyzer.generate_contact_points(grid_resolution_mm=grid_res_mm)

    router = BranchRouter(
        mesh=current_mesh,
        material=material,
        layer_height=layer_height,
        branch_angle_deg=branch_angle_deg,
        merge_distance_mm=merge_distance_mm,
        tip_diameter_mm=tip_diameter_mm,
    )
    graph = router.build_tree_graph(contact_points)
    current_support_graph = graph

    # Build solid 3D mesh
    builder = SupportMeshBuilder(radial_segments=8)
    current_support_mesh = builder.build_mesh(graph)

    # Evaluate physics
    evaluator = SupportPhysicsEvaluator(current_mesh, material=material, layer_height=layer_height)
    report = evaluator.evaluate(graph, contact_points, priority_profile=priority)

    return {
        "status": "generated",
        "nodes_count": len(graph.nodes),
        "tips_count": len(graph.get_tips()),
        "roots_count": len(graph.get_roots()),
        "support_faces_count": len(current_support_mesh.faces),
        "report": {
            "coverage_score_pct": round(report.coverage_score_pct, 1),
            "support_volume_cm3": round(report.support_volume_cm3, 3),
            "support_mass_grams": round(report.support_mass_grams, 2),
            "filament_saved_vs_grid_pct": round(report.filament_saved_vs_standard_grid_pct, 1),
            "filament_saved_vs_tree_pct": round(report.filament_saved_vs_default_tree_pct, 1),
            "average_breakaway_force_n": round(report.average_breakaway_force_n, 2),
            "removal_ease_score": round(report.removal_ease_score, 1),
            "surface_scarring_risk": report.surface_scarring_risk,
            "min_buckling_safety_factor": round(report.min_buckling_safety_factor, 1),
            "bed_adhesion_safety_factor": round(report.bed_adhesion_safety_factor, 1),
            "print_time_minutes": round(report.estimated_support_print_time_minutes, 1),
            "composite_fitness_score": round(report.composite_fitness_score, 1),
        },
    }


@app.post("/api/optimizer/start")
def start_optimizer():
    if current_mesh is None:
        raise HTTPException(status_code=400, detail="No 3D model loaded.")
    success = continuous_engine.start()
    return {"status": "started" if success else "failed", "is_running": continuous_engine.is_running}


@app.post("/api/optimizer/stop")
def stop_optimizer():
    continuous_engine.stop()
    return {"status": "stopped", "is_running": continuous_engine.is_running}


@app.get("/api/optimizer/status")
def get_optimizer_status():
    status = continuous_engine.get_status()
    best_sol = continuous_engine.get_best_support_solution()
    if best_sol:
        status["best_report"] = {
            "filament_saved_pct": round(best_sol["report"].filament_saved_vs_standard_grid_pct, 1),
            "removal_ease": round(best_sol["report"].removal_ease_score, 1),
            "mass_grams": round(best_sol["report"].support_mass_grams, 2),
            "time_minutes": round(best_sol["report"].estimated_support_print_time_minutes, 1),
        }
    return status


@app.get("/api/optimizer/stream")
async def stream_optimizer():
    """Server-Sent Events (SSE) live updates for continuous learning."""
    async def event_generator():
        while True:
            status = continuous_engine.get_status()
            yield f"data: {json.dumps(status)}\n\n"
            await asyncio.sleep(0.5)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/mesh/current-geometry")
def get_current_geometry():
    """Returns vertex and face arrays formatted for Three.js rendering."""
    if current_mesh is None:
        return {"has_model": False}

    # Overhang angles for heatmap rendering
    analyzer = OverhangAnalyzer(current_mesh, material=current_metadata.get("material", "PLA"))
    angles = analyzer.compute_face_overhang_angles()

    model_data = {
        "has_model": True,
        "vertices": current_mesh.vertices.flatten().tolist(),
        "faces": current_mesh.faces.flatten().tolist(),
        "face_angles": angles.tolist(),
        "bounds": [
            [float(x) for x in current_mesh.bounds[0]],
            [float(x) for x in current_mesh.bounds[1]],
        ],
    }

    # Support geometry if generated
    if current_support_mesh is not None and len(current_support_mesh.faces) > 0:
        model_data["support"] = {
            "has_support": True,
            "vertices": current_support_mesh.vertices.flatten().tolist(),
            "faces": current_support_mesh.faces.flatten().tolist(),
        }
    else:
        model_data["support"] = {"has_support": False}

    return model_data


@app.get("/api/export/stl")
def export_stl(include_model: bool = Query(True)):
    global current_mesh, current_support_mesh
    if current_mesh is None:
        raise HTTPException(status_code=400, detail="No model loaded.")

    if include_model and current_support_mesh is not None:
        # Merge model and support
        v1 = current_mesh.vertices
        v2 = current_support_mesh.vertices
        f1 = current_mesh.faces
        f2 = current_support_mesh.faces + len(v1)
        merged_verts = np.vstack([v1, v2])
        merged_faces = np.vstack([f1, f2])
        mesh_to_export = TriangleMesh(merged_verts, merged_faces)
    elif current_support_mesh is not None:
        mesh_to_export = current_support_mesh
    else:
        mesh_to_export = current_mesh

    tmp_path = os.path.join("temp_uploads", "exported_supports.stl")
    STLParser.save_binary(mesh_to_export, tmp_path)
    return FileResponse(tmp_path, filename="optimized_supports.stl", media_type="application/octet-stream")


@app.get("/api/export/3mf")
def export_3mf():
    global current_mesh, current_support_mesh
    if current_mesh is None:
        raise HTTPException(status_code=400, detail="No model loaded.")

    tmp_path = os.path.join("temp_uploads", "bambu_project.3mf")
    Bambu3MFParser.export_bambu_3mf(
        model_mesh=current_mesh,
        support_mesh=current_support_mesh,
        output_filepath=tmp_path,
        material=current_metadata.get("material", "PLA"),
        layer_height=current_metadata.get("layer_height", 0.20),
        z_gap_mm=lookup_optimal_z_gap(
            current_metadata.get("material", "PLA"),
            current_metadata.get("layer_height", 0.20),
        ),
    )
    return FileResponse(tmp_path, filename="bambu_a1_optimized_project.3mf", media_type="application/zip")


# Mount static files for the full-stack Web dashboard
web_dir = os.path.join(os.path.dirname(__file__), "..", "web")
if os.path.exists(web_dir):
    app.mount("/", StaticFiles(directory=web_dir, html=True), name="web")
