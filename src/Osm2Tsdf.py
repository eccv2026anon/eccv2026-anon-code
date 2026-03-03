# -*- coding: utf-8 -*-
   
from __future__ import annotations

import xml.etree.ElementTree as ET                        
from typing import List, Dict, Any, Tuple             
import numpy as np
from shapely.geometry import Point, LineString, Polygon
from shapely.geometry import GeometryCollection, MultiPolygon, MultiLineString                    
import matplotlib.pyplot as plt
import json
import os           
from tqdm import tqdm           
import math               
from scipy.stats import gaussian_kde               

                                                          
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']                       
    plt.rcParams['axes.unicode_minus'] = False                                
except Exception as e:
    print(f"Font setting warning: Unable to set SimHei font. Chinese characters in drawings may not be displayed correctly. mistake:{e}")
    print("Please make sure that a Chinese-capable font (such as SimHei, WenQuanYi Zen Hei, Noto Sans CJK SC) is installed on your system and that Matplotlib can find it.")
                                

                 
try:
    from skimage.measure import marching_cubes
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False

                                        
                             
OSM_PATH                  = "Datasets/ICPARKOSM/ICPARK.osm"
OSM_STYPE_TAG_KEY: str    = "sType"
OUTPUT_DIR: str           = "output/ICPARKOSM_generated/"            

VOXEL                     = 0.06            
TRUNC                     = 0.06            
WALL_THICK                = 0.24            
FLOOR_THICK               = 0.24            
FLOOR_H                   = 3.0             
CAR_SPACE_W               = 2.5                               

CLOSURE_TOLERANCE_OSM_UNITS = 1e-5                    

              
VISUALIZE_3D_MESH         = True                              
PLOT_2D_GEOMETRY_OVERVIEW = True                  
PERFORM_ROTATION_CORRECTION = True               

           
COLOR_WALL_PILLAR: str = "gray"                     
COLOR_PARKING_BG: str  = "lightblue"        


                                                                      
        
                                                                      
def plot_geometry(ax, geom, color, linewidth, alpha=0.7, fill=True, zorder=1):
    if geom is None or geom.is_empty: return
    if isinstance(geom, (Polygon, MultiPolygon)):
        polygons = [geom] if isinstance(geom, Polygon) else list(geom.geoms)
        for poly in polygons:
            if poly is None or poly.is_empty: continue
            x, y = poly.exterior.xy
            ax.plot(x, y, color=color, linewidth=linewidth, solid_capstyle='round', zorder=zorder)
            if fill: ax.fill(x, y, alpha=alpha*0.5, fc=color, ec='none', zorder=zorder-0.5 if zorder > 0 else 0)
            for interior in poly.interiors:
                x_int, y_int = interior.xy
                ax.plot(x_int, y_int, color=color, linewidth=linewidth*0.8, solid_capstyle='round', zorder=zorder)
                if fill: ax.fill(x_int, y_int, alpha=1.0, fc=ax.get_facecolor(), ec='none', zorder=zorder-0.4 if zorder > 0 else 0)
    elif isinstance(geom, (LineString, MultiLineString)):
        lines = [geom] if isinstance(geom, LineString) else list(geom.geoms)
        for line in lines:
            if line is None or line.is_empty: continue
            x, y = line.xy
            ax.plot(x, y, color=color, linewidth=linewidth, solid_capstyle='round', zorder=zorder)
    elif isinstance(geom, GeometryCollection):
        for sub_geom in geom.geoms:
            plot_geometry(ax, sub_geom, color, linewidth, alpha, fill, zorder)

                                                                      
            
                                                                      
def _get_angle(p1: np.ndarray, p2: np.ndarray) -> float:
                                       
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)
    return angle_deg if angle_deg >= 0 else angle_deg + 360

def _rotate_point(point: Tuple[float, float], angle_degrees: float, center: Tuple[float, float] = (0.0, 0.0)) -> Tuple[float, float]:
                             
    angle_rad = math.radians(angle_degrees)
    ox, oy = center
    px, py = point
    qx = ox + math.cos(angle_rad) * (px - ox) - math.sin(angle_rad) * (py - oy)
    qy = oy + math.sin(angle_rad) * (px - ox) + math.cos(angle_rad) * (py - oy)
    return qx, qy

def _calculate_map_rotation_angle(raw_nodes_coords: Dict[str, Tuple[float, float]],
                                  raw_ways: List[Dict[str, Any]]) -> float:
       
    angles = []
    for way_info in raw_ways:
        node_refs = way_info['nds']
        if len(node_refs) != 2: 
            continue
        
        node1_id, node2_id = node_refs[0], node_refs[1]
        if node1_id in raw_nodes_coords and node2_id in raw_nodes_coords:
            coord1 = np.array(raw_nodes_coords[node1_id])
            coord2 = np.array(raw_nodes_coords[node2_id])
            
            if np.allclose(coord1, coord2): continue

            angle = _get_angle(coord1, coord2) 
            angles.append(angle % 90) 

    if not angles:
        print("Warning: No valid line segments found for counting directions. No rotation is performed.")
        return 0.0

    angles_np = np.array(angles)
    if len(angles_np) < 2 : 
        print("Warning: There are too few valid segments (<2) for counting directions. Does not perform exact rotation, returns average value or 0.")
        return np.mean(angles_np) if angles_np.size > 0 else 0.0

    try:
        if np.var(angles_np) < 1e-3: 
            peak_angle_0_90 = np.mean(angles_np)
            print(f"Info: The angle distribution is highly concentrated, use the average angle:{peak_angle_0_90:.2f}message cleaned to English")
        else:
            kde = gaussian_kde(angles_np, bw_method='scott') 
            angle_range = np.linspace(0, 90, 500) 
            density = kde(angle_range)
            peak_angle_0_90 = angle_range[np.argmax(density)]
            print(f"INFO: Calculated map cardinal direction angle (KDE):{peak_angle_0_90:.2f}Degree (range 0-90).")
        return peak_angle_0_90
    except Exception as e:
        print(f"Warning: KDE angle calculation failed:{e}. A simple average of the angles will be used.")
        return np.mean(angles_np)


def _rotate_all_nodes(nodes_coords_dict: Dict[str, Tuple[float, float]],
                      rotation_angle_deg: float) -> Dict[str, Tuple[float, float]]:
                  
    rotated_nodes = {}
    all_coords_np = np.array(list(nodes_coords_dict.values()))
    if all_coords_np.size > 0:
        centroid = tuple(np.mean(all_coords_np, axis=0))
    else:
        centroid = (0.0, 0.0) 

    for node_id, coord in nodes_coords_dict.items():
        rotated_nodes[node_id] = _rotate_point(coord, rotation_angle_deg, center=centroid)
    return rotated_nodes

                                                                      
             
                                                                      
def _parse_raw_osm_data(file_path: str) -> Tuple[Dict[str, Tuple[float, float]], List[Dict[str, Any]]]:
                             
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"OSM file not found:{file_path}")
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
    except ET.ParseError as e:
        raise ValueError(f"Failed to parse OSM file:{e}")

    nodes_coords: Dict[str, Tuple[float, float]] = {}
    for node_el in root.findall("node"):
        try:
            nodes_coords[node_el.attrib["id"]] = (float(node_el.attrib["lon"]), float(node_el.attrib["lat"]))
        except ValueError:
            continue
    
    raw_ways: List[Dict[str, Any]] = []
    for way_el in root.findall("way"):
        raw_ways.append({
            "id": way_el.attrib["id"],
            "nds": [nd.attrib["ref"] for nd in way_el.findall("nd")],
            "tags": {tag.attrib["k"]: tag.attrib["v"] for tag in way_el.findall("tag")}
        })
    return nodes_coords, raw_ways


def _create_shapely_elements_from_osm_data(
        processed_nodes_coords: Dict[str, Tuple[float, float]],
        raw_ways: List[Dict[str, Any]],
        s_type_tag_key: str,
        closure_tolerance: float
    ) -> List[Dict[str, Any]]:
       
    elements: List[Dict[str, Any]] = []
    for way_info in tqdm(raw_ways, desc="1. Processing OSM ways into Shapely objects", unit="way"):
        way_id = way_info["id"]
        node_refs = way_info["nds"]
        tags = way_info["tags"]

        s_type_str_val = tags.get(s_type_tag_key)
        if s_type_str_val not in ("888", "1000", "1002"):
            continue
        s_type = s_type_str_val

        coords = []
        valid_refs = True
        for ref in node_refs:
            if ref in processed_nodes_coords:
                coords.append(processed_nodes_coords[ref])
            else:
                valid_refs = False
                break
        
        if not valid_refs or len(coords) < 2: continue

        shape = None
        is_osm_way_closed_by_refs = len(node_refs) > 2 and node_refs[0] == node_refs[-1]
        is_effectively_closed = is_osm_way_closed_by_refs
        coords_for_shape = list(coords)

        if not is_osm_way_closed_by_refs and len(coords) >= 3:
            p_start = Point(coords[0])
            p_end = Point(coords[-1])
            if p_start.distance(p_end) < closure_tolerance: 
                is_effectively_closed = True
                if p_start.distance(p_end) > 1e-9: 
                    coords_for_shape.append(coords[0])
        
        try:
            if is_effectively_closed:
                unique_point_tuples = set(map(tuple, coords_for_shape))
                if len(unique_point_tuples) < 3:
                    if len(coords) >= 2: shape = LineString(coords)
                else:
                    shape = Polygon(coords_for_shape)
                    if not shape.is_valid:
                        shape_fixed = shape.buffer(0)
                        if shape_fixed.is_valid and not shape_fixed.is_empty and isinstance(shape_fixed, Polygon):
                            shape = shape_fixed
                        elif s_type == "1000" and len(coords) >=2:
                            shape = LineString(coords)
                        else: shape = None
            elif len(coords) >= 2:
                shape = LineString(coords)

            if shape and shape.is_empty: shape = None
        except Exception: 
            shape = None

        if shape:
            elements.append({"id": way_id, "sType": s_type, "geometry": shape, "tags": tags})
            
    print(f"Processed and converted{len(elements)}valid Shapely geometric elements matching the specified sType.")
    return elements

                                       
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"INFO: All output files will be saved to directory:{OUTPUT_DIR}")

    print(f"\n[OSM] Start parsing raw OSM data:{OSM_PATH} ...")
    try:
        raw_nodes_coords, raw_ways = _parse_raw_osm_data(OSM_PATH)
    except Exception as e:
        print(f"mistake:{e}")
        return
    
    print(f"Original OSM data parsing completed:{len(raw_nodes_coords)}nodes,{len(raw_ways)}path.")

    final_nodes_coords = raw_nodes_coords
    if PERFORM_ROTATION_CORRECTION:
        print("\\n[Map Orientation Correction] Start calculating the main direction...")
        dominant_angle_0_90 = _calculate_map_rotation_angle(raw_nodes_coords, raw_ways)
        if abs(dominant_angle_0_90) > 1.0: 
            rotation_to_apply_deg = -dominant_angle_0_90 
            final_nodes_coords = _rotate_all_nodes(raw_nodes_coords, rotation_to_apply_deg)
            print(f"Message: Node coordinates have been rotated around the center of mass{rotation_to_apply_deg:.2f}message cleaned to English")
        else:
            print(f"INFO: Calculated main direction angle ({dominant_angle_0_90:.2f}degree) is close to 0 or cannot be determined, no rotation is performed.")
    else:
        print("\\n[Map Orientation Correction] has been skipped.")

    parsed_elements = _create_shapely_elements_from_osm_data(
        final_nodes_coords, raw_ways, OSM_STYPE_TAG_KEY, CLOSURE_TOLERANCE_OSM_UNITS
    )

    walls_osm_shapely: list[LineString] = []
    polygons_osm_shapely: list[Polygon] = []
    slots_osm_shapely: list[Polygon] = []

    for el in parsed_elements:
        s_type_val = el["sType"]
        geom = el["geometry"]
        if s_type_val == "888":
            if isinstance(geom, LineString): walls_osm_shapely.append(geom)
        elif s_type_val == "1000":
            if isinstance(geom, Polygon): polygons_osm_shapely.append(geom)
            elif isinstance(geom, LineString): walls_osm_shapely.append(geom)
        elif s_type_val == "1002":
            if isinstance(geom, Polygon): slots_osm_shapely.append(geom)

    print(f"\nShapely object classification completed:")
    print(f"Wall segments (sType 888 and Linear 1000):{len(walls_osm_shapely)}strip")
    print(f"Enclosed area/cylindrical polygon (sType 1000):{len(polygons_osm_shapely)}indivual")
    print(f"Parking space polygon (sType 1002):{len(slots_osm_shapely)}indivual")
    
    if not any([walls_osm_shapely, polygons_osm_shapely, slots_osm_shapely]):
        print("\\nError: Unable to parse any valid geometric elements (walls, columns, parking spaces) from the OSM file. Please check whether the OSM file content and sType tag are correct. The program terminates.")
        return

    SCALE = 1.0
    if slots_osm_shapely:
        parking_slot_short_sides = []
        for poly in slots_osm_shapely:
            if poly.exterior is None or len(list(poly.exterior.coords)) < 4: 
                continue
            coords = list(poly.exterior.coords)
            s1 = Point(coords[0]).distance(Point(coords[1]))
            s2 = Point(coords[1]).distance(Point(coords[2]))
            current_short_side = min(s1, s2)
            if current_short_side > 1e-6: parking_slot_short_sides.append(current_short_side)
            
        if parking_slot_short_sides:
            mean_osm_short_side_len = np.mean(parking_slot_short_sides)
            print(f"Information: Calculated average short side length of OSM parking spaces (based on the first two sides):{mean_osm_short_side_len:.4f}OSM units (after rotation)")
            if mean_osm_short_side_len > 1e-9: SCALE = CAR_SPACE_W / mean_osm_short_side_len
            else: print("Warning: The average short side length of the parking space is too small or zero. Use SCALE = 1.0"); SCALE = 1.0
        else: print("Warning: No valid parking space short side found for scale estimation. Use SCALE = 1.0"); SCALE = 1.0
    else: print("Warning: Parking space data not found (sType=1002) for scale estimation. Use SCALE = 1.0"); SCALE = 1.0
    
    if SCALE > 1000 or SCALE < 0.001 : print(f"Warning: Estimated scale (SCALE ={SCALE:.4f}) very large or very small. Please check the OSM coordinate units or the CAR_SPACE_W parameter.")
    else: print(f"SCALE ={SCALE:.4f}Meter/OSM coordinate unit (after rotation)")

    def project_coords_to_meters(coords_list_tuples):
        return [(lon * SCALE, lat * SCALE) for lon, lat in coords_list_tuples]

    walls_m = [LineString(project_coords_to_meters(list(line.coords))) for line in walls_osm_shapely if not line.is_empty]
    walls_m = [w for w in walls_m if not w.is_empty]

    closed_areas_m = []
    for poly in polygons_osm_shapely:
        if poly.exterior is not None and len(list(poly.exterior.coords)) > 0:
            projected_poly_coords = project_coords_to_meters(list(poly.exterior.coords))
            if len(projected_poly_coords) >= 3:
                new_poly = Polygon(projected_poly_coords)
                if new_poly.is_valid and not new_poly.is_empty: closed_areas_m.append(new_poly)
    print(f"Effective wall after projection:{len(walls_m)}Bar, valid closed area/cylindrical polygon (sType 1000):{len(closed_areas_m)}indivual")

    slots_m = []
    for poly in slots_osm_shapely:
        if poly.exterior is not None and len(list(poly.exterior.coords)) > 0:
            projected_slot_coords = project_coords_to_meters(list(poly.exterior.coords))
            if len(projected_slot_coords) >=3:
                new_poly = Polygon(projected_slot_coords)
                if new_poly.is_valid and not new_poly.is_empty: slots_m.append(new_poly)
    print(f"Valid parking spaces after projection:{len(slots_m)}indivual")

    if PLOT_2D_GEOMETRY_OVERVIEW:
        print("\\n[2D Overview] Generating an analytical geometry overview (metric coordinates)...")
        fig, ax = plt.subplots(figsize=(12, 10))
        
        for slot in tqdm(slots_m, desc="Plotting Slots", unit="slot", leave=False):
            plot_geometry(ax, slot, color=COLOR_PARKING_BG, linewidth=1, fill=True, alpha=0.5, zorder=1) 
        for area in tqdm(closed_areas_m, desc="Plotting Closed Areas", unit="area", leave=False):
            plot_geometry(ax, area, color=COLOR_WALL_PILLAR, linewidth=1.5, fill=True, alpha=0.7, zorder=2)
        for wall_line in tqdm(walls_m, desc="Plotting Walls", unit="wall", leave=False):
            plot_geometry(ax, wall_line, color=COLOR_WALL_PILLAR, linewidth=2, fill=False, zorder=3) 

        ax.set_title(f"2D Geometry Overview (Metric, Rotated) from {os.path.basename(OSM_PATH)}")
        ax.set_xlabel("X (meters)"); ax.set_ylabel("Y (meters)"); ax.axis('equal'); ax.grid(True, linestyle=':', alpha=0.5)
        from matplotlib.patches import Patch
        legend_elements = [ Patch(facecolor=COLOR_PARKING_BG, alpha=0.5, label='Parking spaces (1002)'), 
                            Patch(facecolor=COLOR_WALL_PILLAR, alpha=0.7, label='Enclosed area/column (1000 Poly)'), 
                            Patch(facecolor='none', edgecolor=COLOR_WALL_PILLAR, linewidth=2, label='Wall (888 & 1000 Line)')] 
        ax.legend(handles=legend_elements, loc='best'); plt.tight_layout()
        
        plot_filename = os.path.join(OUTPUT_DIR, "2d_geometry_overview.png")
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"[2D Overview] Saved to:{plot_filename}")
                     

    all_metric_points = []
    for collection in [walls_m, closed_areas_m, slots_m]:
        for geom_item in collection:
            if isinstance(geom_item, LineString): all_metric_points.extend(list(geom_item.coords))
            elif isinstance(geom_item, Polygon) and geom_item.exterior: all_metric_points.extend(list(geom_item.exterior.coords))
    
    if not all_metric_points:
        print("Error: There are no valid metric geometry points for calculating scene boundaries. Unable to generate TSDF. The program terminates.")
        return
    
    xs, ys = zip(*all_metric_points)
    bound_min_m = np.array([min(xs) - 1.0, min(ys) - 1.0, -FLOOR_THICK])                        
    bound_max_m = np.array([max(xs) + 1.0, max(ys) + 1.0, FLOOR_H])
    print(f"\nScene Boundary (meters): Min={np.round(bound_min_m, 2)}, Max={np.round(bound_max_m, 2)}")

                        
                                       
    tsdf_dims = np.ceil((bound_max_m - bound_min_m) / VOXEL).astype(int)
    Dx, Dy, Dz = tsdf_dims[0], tsdf_dims[1], tsdf_dims[2]
    
                
    expected_Dz = int(np.ceil((FLOOR_H - (-FLOOR_THICK)) / VOXEL))
    if Dz != expected_Dz:
        print(f"[WARNING] The calculation of the Z-axis dimension may be incorrect: the calculated value is Dz={Dz}, but the expected value is{expected_Dz}")
        print(f"Correcting Z axis dimension...")
        Dz = expected_Dz
        tsdf_dims[2] = Dz
    
    if not (Dx > 0 and Dy > 0 and Dz > 0): 
        print(f"Error: Invalid TSDF dimension ({Dx},{Dy},{Dz}). The program terminates."); return
    print(f"TSDF dimensions (voxels): Dx={Dx}, Dy={Dy}, Dz={Dz}")
    print(f"Verification: Z-axis range [{bound_min_m[2]:.2f}, {bound_max_m[2]:.2f}] meters, length{bound_max_m[2]-bound_min_m[2]:.2f}rice")
    print(f"voxel size{VOXEL:.2f}meters, theoretically required{(bound_max_m[2]-bound_min_m[2])/VOXEL:.2f}Voxels, rounded up to{Dz}indivual")
    
                   
    tsdf = np.ones((Dz, Dy, Dx), dtype=np.float32)

    def real_xy_to_voxel_idx(x_m, y_m):
        return int((x_m - bound_min_m[0]) / VOXEL), int((y_m - bound_min_m[1]) / VOXEL)

    print("Grid floor..."); tqdm_floor = tqdm(range(Dz), desc="2. Rasterizing floor", unit="slice")
                                        
    floor_z = 0.0             
    floor_bottom_z = floor_z - FLOOR_THICK             
    
    for iz in tqdm_floor:
        z_coord = bound_min_m[2] + (iz + 0.5) * VOXEL              
        
                         
        dist_to_floor_top = abs(z_coord - floor_z)
        dist_to_floor_bottom = abs(z_coord - floor_bottom_z)
        
                               
        if floor_bottom_z <= z_coord <= floor_z:
                         
                                   
            tsdf[iz, :, :] = np.minimum(tsdf[iz, :, :], -1.0)
        else:
                          
            dist_to_floor = min(dist_to_floor_top, dist_to_floor_bottom)
            sdf_floor_val = dist_to_floor / TRUNC           
            
                           
            if z_coord < floor_bottom_z:
                sdf_floor_val = -sdf_floor_val
                
                                     
            floor_normalized = np.clip(sdf_floor_val, -1.0, 1.0)
                             
            positive_mask = floor_normalized >= 0
            tsdf[iz, positive_mask] = np.minimum(tsdf[iz, positive_mask], floor_normalized[positive_mask])

                          
                             
    z_slice_start_idx = 0
                         
    if bound_min_m[2] < 0:
                        
        z_zero_plane_idx = -bound_min_m[2] / VOXEL
        z_slice_start_idx = np.clip(int(z_zero_plane_idx), 0, Dz -1)

    print(f"Rasterize{len(walls_m)}Strip wall (only in Z>=0 part, solid)..."); tqdm_walls = tqdm(walls_m, desc="3. Rasterizing walls as solid (Z>=0)", unit="wall")
    for line_m in tqdm_walls:
        wall_polygon_m = line_m.buffer(WALL_THICK / 2, cap_style='flat')
        if not wall_polygon_m.is_valid or wall_polygon_m.is_empty: continue
        minx_m, miny_m, maxx_m, maxy_m = wall_polygon_m.bounds
        ix0,iy0=real_xy_to_voxel_idx(minx_m-TRUNC,miny_m-TRUNC); ix1,iy1=real_xy_to_voxel_idx(maxx_m+TRUNC,maxy_m+TRUNC)
        ix0,iy0=max(0,ix0),max(0,iy0); ix1,iy1=min(Dx-1,ix1),min(Dy-1,iy1)
        for ix in range(ix0,ix1+1):
            for iy in range(iy0,iy1+1):
                pt = Point(bound_min_m[0]+(ix+0.5)*VOXEL, bound_min_m[1]+(iy+0.5)*VOXEL)
                               
                is_inside = wall_polygon_m.contains(pt) or wall_polygon_m.distance(pt) < 1e-9
                if is_inside:
                                           
                    dist_to_boundary = wall_polygon_m.exterior.distance(pt)
                    sdf_val = -dist_to_boundary if dist_to_boundary > 1e-9 else -TRUNC
                else:
                                       
                    sdf_val = wall_polygon_m.distance(pt)
                                                  
                                         
                sdf_normalized = np.clip(sdf_val/TRUNC, -1.0, 1.0)
                if sdf_normalized < 0:
                                      
                    tsdf[z_slice_start_idx:,iy,ix] = sdf_normalized
                else:
                                            
                    current_values = tsdf[z_slice_start_idx:,iy,ix]
                    positive_mask = current_values >= 0
                    tsdf[z_slice_start_idx:,iy,ix][positive_mask] = np.minimum(
                        current_values[positive_mask], sdf_normalized)

    print(f"Rasterize{len(closed_areas_m)}sType 1000 polygons (only in Z>=0 part, solid columns)..."); tqdm_areas = tqdm(closed_areas_m, desc="4. Rasterizing sType 1000 solid pillars (Z>=0)", unit="poly")
    for poly_m in tqdm_areas:
        if not poly_m.is_valid or poly_m.is_empty: continue
        minx_m,miny_m,maxx_m,maxy_m = poly_m.bounds
        ix0,iy0=real_xy_to_voxel_idx(minx_m-TRUNC,miny_m-TRUNC); ix1,iy1=real_xy_to_voxel_idx(maxx_m+TRUNC,maxy_m+TRUNC)
        ix0,iy0=max(0,ix0),max(0,iy0); ix1,iy1=min(Dx-1,ix1),min(Dy-1,iy1)
        for ix in range(ix0,ix1+1):
            for iy in range(iy0,iy1+1):
                pt = Point(bound_min_m[0]+(ix+0.5)*VOXEL, bound_min_m[1]+(iy+0.5)*VOXEL)
                               
                is_inside = poly_m.contains(pt) or poly_m.distance(pt) < 1e-9
                if is_inside:
                                           
                    dist_to_boundary = poly_m.exterior.distance(pt)
                    sdf_val = -dist_to_boundary if dist_to_boundary > 1e-9 else -TRUNC
                else:
                                       
                    sdf_val = poly_m.distance(pt)
                                                  
                                         
                sdf_normalized = np.clip(sdf_val/TRUNC, -1.0, 1.0)
                if sdf_normalized < 0:
                                      
                    tsdf[z_slice_start_idx:,iy,ix] = sdf_normalized
                else:
                                            
                    current_values = tsdf[z_slice_start_idx:,iy,ix]
                    positive_mask = current_values >= 0
                    tsdf[z_slice_start_idx:,iy,ix][positive_mask] = np.minimum(
                        current_values[positive_mask], sdf_normalized)

                     
    print("Perform thorough solidification...")
    
                         
    solid_mask = np.zeros((Dy, Dx), dtype=bool)
    
               
    for line_m in walls_m:
        wall_polygon_m = line_m.buffer(WALL_THICK / 2, cap_style='flat')
        if not wall_polygon_m.is_valid or wall_polygon_m.is_empty: continue
        minx_m, miny_m, maxx_m, maxy_m = wall_polygon_m.bounds
        ix0,iy0=real_xy_to_voxel_idx(minx_m-TRUNC,miny_m-TRUNC); ix1,iy1=real_xy_to_voxel_idx(maxx_m+TRUNC,maxy_m+TRUNC)
        ix0,iy0=max(0,ix0),max(0,iy0); ix1,iy1=min(Dx-1,ix1),min(Dy-1,iy1)
        for ix in range(ix0,ix1+1):
            for iy in range(iy0,iy1+1):
                pt = Point(bound_min_m[0]+(ix+0.5)*VOXEL, bound_min_m[1]+(iy+0.5)*VOXEL)
                if wall_polygon_m.contains(pt) or wall_polygon_m.distance(pt) < 1e-9:
                    solid_mask[iy, ix] = True
    
               
    for poly_m in closed_areas_m:
        if not poly_m.is_valid or poly_m.is_empty: continue
        minx_m,miny_m,maxx_m,maxy_m = poly_m.bounds
        ix0,iy0=real_xy_to_voxel_idx(minx_m-TRUNC,miny_m-TRUNC); ix1,iy1=real_xy_to_voxel_idx(maxx_m+TRUNC,maxy_m+TRUNC)
        ix0,iy0=max(0,ix0),max(0,iy0); ix1,iy1=min(Dx-1,ix1),min(Dy-1,iy1)
        for ix in range(ix0,ix1+1):
            for iy in range(iy0,iy1+1):
                pt = Point(bound_min_m[0]+(ix+0.5)*VOXEL, bound_min_m[1]+(iy+0.5)*VOXEL)
                if poly_m.contains(pt) or poly_m.distance(pt) < 1e-9:
                    solid_mask[iy, ix] = True
    
                        
    solid_count = np.sum(solid_mask)
    print(f"tagged{solid_count}Voxels are solid areas")
    
    for iz in range(z_slice_start_idx, Dz):
                             
        tsdf[iz, solid_mask] = -0.8                 
    
    print("[INFO]")

    tsdf_output_path = os.path.join(OUTPUT_DIR, "Dparking_prior_tsdf.npy")
    slots_output_path = os.path.join(OUTPUT_DIR, "parking_slots.json")
    boundary_config_path = os.path.join(OUTPUT_DIR, "boundary_config.txt")
    mesh_output_filename = "tsdf_3d_mesh.ply" 

    np.save(tsdf_output_path, tsdf); print(f"\nTSDF data saved:{tsdf_output_path}, shape:{tsdf.shape}")
    
                                     
    try:
        print(f"Reloading and validating saved .npy files...")
        reloaded_tsdf = np.load(tsdf_output_path)
        print(f"[INFO] The reloaded TSDF dimensions are:{reloaded_tsdf.shape}")
        if not np.array_equal(tsdf.shape, reloaded_tsdf.shape):
            print(f"[WARNING] Reloaded dimensions do not match original dimensions! Original dimensions:{tsdf.shape}")
    except Exception as e:
        print(f"[ERROR] Error reloading .npy file:{e}")
                    
    
              
    try:
        import time
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(boundary_config_path, "w", encoding="utf-8") as f:
            f.write(f"""# StructRecon Scene Boundary & Origin Configuration
# Generated by Osm2Tsdf.py on: {timestamp}
# ----------------------------------------------------------------
#
# Instructions:
# 1. Open your target config file (e.g., configs/structrecon.yaml or a specific scene config).
# 2. Update the 'mapping.bound' parameter with the values below.
# 3. Update the 'prior_tsdf_origin_xyz' parameter with the value below.
#
# ----------------------------------------------------------------

# For 'mapping.bound' in YAML
# This defines the 3D boundary of the scene for the SLAM system.
# Format: [[min_x, max_x], [min_y, max_y], [min_z, max_z]]
mapping:
  bound: [[{bound_min_m[0]:.4f}, {bound_max_m[0]:.4f}], [{bound_min_m[1]:.4f}, {bound_max_m[1]:.4f}], [{bound_min_m[2]:.4f}, {bound_max_m[2]:.4f}]]

# For 'prior_tsdf_origin_xyz' in YAML
# This tells the system where the origin of the pre-optimized TSDF numpy array is.
# It MUST match the [min_x, min_y, min_z] from the bound above.
prior_tsdf_origin_xyz: [{bound_min_m[0]:.4f}, {bound_min_m[1]:.4f}, {bound_min_m[2]:.4f}]
""")
        print(f"Boundary profile saved:{boundary_config_path}")
    except Exception as e:
        print(f"Error saving boundary configuration file:{e}")
    
    slots_out_data = [{'cx':p.centroid.x, 'cy':p.centroid.y, 'hull':list(p.exterior.coords)} for p in slots_m if p.exterior]
    try:
        with open(slots_output_path,"w",encoding="utf-8") as f: json.dump(slots_out_data,f,indent=2,ensure_ascii=False)
        print(f"Parking space data has been saved:{len(slots_out_data)}parking spaces ->{slots_output_path}")
    except IOError as e: print(f"Error: Failed to save parking space JSON file:{e}")

    z_slice_meters=1.0; z_idx=np.clip(int((z_slice_meters-bound_min_m[2])/VOXEL),0,Dz-1)
    print(f"\nShow TSDF at Z{z_slice_meters:.2f}m (voxel layer{z_idx}) slice...")
    plt.figure(figsize=(12,12*Dy/Dx if Dx>0 and Dy>0 else 10))
    plt.imshow(tsdf[z_idx,:,:],cmap="coolwarm_r",origin="lower",vmin=-1,vmax=1,extent=[bound_min_m[0],bound_max_m[0],bound_min_m[1],bound_max_m[1]])
    plt.colorbar(label="SDF Value"); plt.title(f"TSDF Slice @ Z{z_slice_meters:.2f}m (Layer {z_idx})")
    plt.xlabel(f"X (m), Dx={Dx}"); plt.ylabel(f"Y (m), Dy={Dy}"); plt.axis('equal'); 
    
    tsdf_slice_filename = os.path.join(OUTPUT_DIR, "tsdf_slice_overview.png")
    plt.savefig(tsdf_slice_filename, dpi=300, bbox_inches='tight')
    print(f"TSDF slice map has been saved to:{tsdf_slice_filename}")
                

                                     
    can_generate_mesh = ('tsdf' in locals() and tsdf is not None and 
                         bound_min_m is not None and 
                         SKIMAGE_AVAILABLE and OPEN3D_AVAILABLE)

    if can_generate_mesh:
        print(f"\n[3D Mesh Processing] Start mesh generation from TSDF...")
        try:
                                  
            verts, faces, _, _ = marching_cubes(volume=tsdf, level=0.0, spacing=(VOXEL, VOXEL, VOXEL))
            
            if verts.shape[0] == 0 or faces.shape[0] == 0:
                print("Warning: Marching cubes failed to extract any vertices or faces. Unable to generate or save mesh file.")
            else:
                print(f"[3D Mesh Processing] generated{verts.shape[0]}vertices and{faces.shape[0]}A triangular surface.")
                
                                        
                verts_world = np.zeros_like(verts)
                verts_world[:, 0] = verts[:, 2] + bound_min_m[0]
                verts_world[:, 1] = verts[:, 1] + bound_min_m[1]
                verts_world[:, 2] = verts[:, 0] + bound_min_m[2]
                mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(verts_world), o3d.utility.Vector3iVector(faces))
                mesh.compute_vertex_normals()

                                  
                try:
                    mesh_output_full_path = os.path.join(OUTPUT_DIR, mesh_output_filename)
                    o3d.io.write_triangle_mesh(mesh_output_full_path, mesh)
                    print(f"[INFO] Mesh successfully saved to:{mesh_output_full_path}")
                except Exception as e:
                    print(f"[ERROR] Saving grid file '{mesh_output_full_path}' fail:{e}")

                                                          
                if VISUALIZE_3D_MESH:
                    print("[3D Visualization] Trying to start the Open3D interactive viewer...")
                    try:
                        o3d.visualization.draw_geometries([mesh], window_name="TSDF 3D Mesh", width=1024, height=768, mesh_show_back_face=True)
                    except Exception as e:
                        print(f"[ERROR] Failed to start interactive viewer:{e}")
                else:
                    print("\\n[3D Visualization] skipped with VISUALIZE_3D_MESH=False.")

        except Exception as e:
            print(f"[ERROR] A serious error occurred during Marching cubes or mesh processing:{e}")
    else:
        print("\\n[3D Mesh Processing] Skip. Reason: The dependent library is missing or the TSDF data was not successfully generated.")
    
                         
    with open(os.path.join(OUTPUT_DIR, "world_from_osm.json"), "w", encoding="utf-8") as f:
        json.dump({
            "rotation_deg": float(dominant_angle_0_90),                      
            "applied_rotation_deg": float(-dominant_angle_0_90),                  
            "centroid_osm": [float(centroid[0]), float(centroid[1])],
            "scale_m_per_osm": float(SCALE),
            "prior_tsdf_origin_xyz": [float(bound_min_m[0]), float(bound_min_m[1]), float(bound_min_m[2])]
        }, f, indent=2, ensure_ascii=False)



if __name__ == "__main__":
    if not SKIMAGE_AVAILABLE: print("Error: scikit-image library not found, unable to march_cubes.")
    if not OPEN3D_AVAILABLE: print("Error: open3d library not found, unable to process and save mesh.")
    if OSM_STYPE_TAG_KEY not in ["stype", "sType"]: print(f"NOTE: OSM_STYPE_TAG_KEY is set to '{OSM_STYPE_TAG_KEY}'.")
    if not os.path.exists(OSM_PATH): 
        print(f"Error: OSM file path '{OSM_PATH}' Doesn't exist! The program terminates."); 
        exit()
    main()
