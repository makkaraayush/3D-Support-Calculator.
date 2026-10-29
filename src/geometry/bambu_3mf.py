"""
Bambu Studio / OrcaSlicer 3MF Archive Parser and Exporter.
Extracts 3D mesh geometry, plate layouts, and slicer configuration from .3mf zip packages.
Allows re-packaging models with AI-optimized support structures and Bambu Studio configs.
"""

import io
import json
import os
import xml.etree.ElementTree as ET
import zipfile
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
from .mesh import TriangleMesh


class Bambu3MFParser:
    @staticmethod
    def load(filepath_or_bytes: Union[str, bytes]) -> Tuple[TriangleMesh, Dict[str, Any]]:
        """
        Loads a .3mf file, extracts the 3D model mesh and slicer metadata.
        Returns: (TriangleMesh, metadata_dict)
        """
        if isinstance(filepath_or_bytes, bytes):
            bio = io.BytesIO(filepath_or_bytes)
            archive = zipfile.ZipFile(bio, "r")
        else:
            archive = zipfile.ZipFile(filepath_or_bytes, "r")

        metadata: Dict[str, Any] = {
            "source_type": "3MF",
            "material": "PLA",
            "layer_height": 0.20,
            "nozzle_diameter": 0.4,
            "plate_name": "Plate 1",
            "bambu_config": {},
        }

        # 1. Look for Bambu / OrcaSlicer config files
        for name in archive.namelist():
            if name.endswith("slice_info.config") or name.endswith("process.json"):
                try:
                    raw_cfg = archive.read(name).decode("utf-8", errors="ignore")
                    if name.endswith(".json"):
                        cfg_json = json.loads(raw_cfg)
                        metadata["bambu_config"].update(cfg_json)
                    else:
                        # Parse key-value config
                        for line in raw_cfg.splitlines():
                            if "=" in line:
                                k, v = line.split("=", 1)
                                metadata["bambu_config"][k.strip()] = v.strip()
                except Exception:
                    pass

        # Detect layer height / nozzle / material from config if present
        bambu_cfg = metadata["bambu_config"]
        if "layer_height" in bambu_cfg:
            try:
                metadata["layer_height"] = float(bambu_cfg["layer_height"])
            except ValueError:
                pass
        if "nozzle_diameter" in bambu_cfg:
            try:
                metadata["nozzle_diameter"] = float(bambu_cfg["nozzle_diameter"])
            except ValueError:
                pass
        if "filament_type" in bambu_cfg:
            metadata["material"] = str(bambu_cfg["filament_type"]).upper()

        # 2. Extract 3D Model XML (typically at '3D/3dmodel.model')
        model_filename = None
        for name in archive.namelist():
            if name.endswith("3dmodel.model"):
                model_filename = name
                break

        if not model_filename:
            # Fallback search for any .model file
            for name in archive.namelist():
                if name.endswith(".model"):
                    model_filename = name
                    break

        if not model_filename:
            archive.close()
            raise ValueError("No 3D model XML found in .3mf package.")

        model_xml = archive.read(model_filename).decode("utf-8", errors="ignore")
        archive.close()

        mesh = Bambu3MFParser._parse_model_xml(model_xml)
        return mesh, metadata

    @staticmethod
    def _parse_model_xml(xml_content: str) -> TriangleMesh:
        """Parses standard 3MF XML namespace model for vertices and triangles."""
        # Strip XML namespaces for easier traversal
        # e.g., {http://schemas.microsoft.com/3dmanufacturing/core/2015/02}
        tree = ET.fromstring(xml_content)

        all_vertices: List[List[float]] = []
        all_faces: List[List[int]] = []

        # Find all <mesh> elements in the XML
        for elem in tree.iter():
            tag = elem.tag.split("}")[-1]  # strip namespace
            if tag == "mesh":
                v_offset = len(all_vertices)

                # Parse vertices
                for child in elem:
                    c_tag = child.tag.split("}")[-1]
                    if c_tag == "vertices":
                        for v_node in child:
                            x = float(v_node.attrib.get("x", 0.0))
                            y = float(v_node.attrib.get("y", 0.0))
                            z = float(v_node.attrib.get("z", 0.0))
                            all_vertices.append([x, y, z])
                    elif c_tag == "triangles":
                        for t_node in child:
                            v1 = int(t_node.attrib.get("v1", 0)) + v_offset
                            v2 = int(t_node.attrib.get("v2", 0)) + v_offset
                            v3 = int(t_node.attrib.get("v3", 0)) + v_offset
                            all_faces.append([v1, v2, v3])

        if not all_vertices or not all_faces:
            raise ValueError("3MF model XML contained no readable vertices or triangles.")

        verts_arr = np.array(all_vertices, dtype=np.float32)
        faces_arr = np.array(all_faces, dtype=np.int32)
        return TriangleMesh(verts_arr, faces_arr)

    @staticmethod
    def export_bambu_3mf(
        model_mesh: TriangleMesh,
        support_mesh: Optional[TriangleMesh],
        output_filepath: str,
        material: str = "PLA",
        layer_height: float = 0.20,
        z_gap_mm: float = 0.22,
    ) -> None:
        """
        Creates a valid Bambu Studio / OrcaSlicer compatible .3mf project file
        with the model and optional separate support structure.
        """
        bio = io.BytesIO()
        with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            # [Content_Types].xml
            content_types = (
                '<?xml version="1.0" encoding="UTF-8"?>\n'
                '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">\n'
                '  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>\n'
                '  <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodelxml"/>\n'
                '  <Default Extension="json" ContentType="application/json"/>\n'
                '</Types>'
            )
            zf.writestr("[Content_Types].xml", content_types)

            # _rels/.rels
            rels = (
                '<?xml version="1.0" encoding="UTF-8"?>\n'
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
                '  <Relationship Target="/3D/3dmodel.model" Id="rel0" '
                'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/>\n'
                '</Relationships>'
            )
            zf.writestr("_rels/.rels", rels)

            # 3D/3dmodel.model
            model_xml = Bambu3MFParser._build_3dmodel_xml(model_mesh, support_mesh)
            zf.writestr("3D/3dmodel.model", model_xml)

            # Metadata/process.json (Bambu Studio / OrcaSlicer configuration overrides)
            bambu_process = {
                "filament_type": material,
                "layer_height": str(layer_height),
                "nozzle_diameter": "0.4",
                "support_type": "tree_slim",
                "support_top_z_distance": str(z_gap_mm),
                "support_bottom_z_distance": str(z_gap_mm),
                "support_interface_spacing": "0.45",
                "support_threshold_angle": "62",
                "generator": "3D-Support-Calculator-AI",
            }
            zf.writestr("Metadata/process.json", json.dumps(bambu_process, indent=2))

        with open(output_filepath, "wb") as f:
            f.write(bio.getvalue())

    @staticmethod
    def _build_3dmodel_xml(model_mesh: TriangleMesh, support_mesh: Optional[TriangleMesh]) -> str:
        xml_lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<model unit="millimeter" xml:lang="en-US" xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">',
            '  <resources>',
        ]

        # Model object ID=1
        xml_lines.append('    <object id="1" type="model">')
        xml_lines.append('      <mesh>')
        xml_lines.append('        <vertices>')
        for v in model_mesh.vertices:
            xml_lines.append(f'          <vertex x="{v[0]:.4f}" y="{v[1]:.4f}" z="{v[2]:.4f}" />')
        xml_lines.append('        </vertices>')
        xml_lines.append('        <triangles>')
        for f in model_mesh.faces:
            xml_lines.append(f'          <triangle v1="{f[0]}" v2="{f[1]}" v3="{f[2]}" />')
        xml_lines.append('        </triangles>')
        xml_lines.append('      </mesh>')
        xml_lines.append('    </object>')

        # Optional Support object ID=2
        if support_mesh is not None and len(support_mesh.faces) > 0:
            xml_lines.append('    <object id="2" type="support">')
            xml_lines.append('      <mesh>')
            xml_lines.append('        <vertices>')
            for v in support_mesh.vertices:
                xml_lines.append(f'          <vertex x="{v[0]:.4f}" y="{v[1]:.4f}" z="{v[2]:.4f}" />')
            xml_lines.append('        </vertices>')
            xml_lines.append('        <triangles>')
            for f in support_mesh.faces:
                xml_lines.append(f'          <triangle v1="{f[0]}" v2="{f[1]}" v3="{f[2]}" />')
            xml_lines.append('        </triangles>')
            xml_lines.append('      </mesh>')
            xml_lines.append('    </object>')

        xml_lines.append('  </resources>')
        xml_lines.append('  <build>')
        xml_lines.append('    <item objectid="1" />')
        if support_mesh is not None and len(support_mesh.faces) > 0:
            xml_lines.append('    <item objectid="2" />')
        xml_lines.append('  </build>')
        xml_lines.append('</model>')

        return "\n".join(xml_lines)
