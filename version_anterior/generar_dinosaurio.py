"""
Generador procedural de un Tyrannosaurus rex para Blender 5.x

Uso:
    blender -b --python generar_dinosaurio.py
    blender -b --python generar_dinosaurio.py -- --render salida.png

Construye:
  * Malla orgánica del cuerpo a partir de un esqueleto (modificador Skin) + subdivisión
  * Mandíbula independiente con lengua, dientes, garras, ojos y escamas dorsales
  * Materiales procedurales (piel escamosa con bump, dientes, garras, ojos con pupila rasgada)
  * Rig con armadura (pesos automáticos) y pose dinámica
  * Escenario (suelo, rocas), iluminación de tres puntos y cámara con profundidad de campo
"""

import math
import random
import sys
from pathlib import Path

import bmesh
import bpy
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree

PROJECT_DIR = Path(__file__).resolve().parent
BLEND_PATH = PROJECT_DIR / "dinosaurio.blend"

random.seed(7)
SUN_AZIMUTH, SUN_ELEVATION = 15.0, 26.0  # grados


# ---------------------------------------------------------------------------
# Utilidades generales
# ---------------------------------------------------------------------------

def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = 'METRIC'
    scene.unit_settings.length_unit = 'METERS'
    return scene


def make_collection(name, parent=None):
    col = bpy.data.collections.new(name)
    (parent or bpy.context.scene.collection).children.link(col)
    return col


def link(obj, col):
    for c in obj.users_collection:
        c.objects.unlink(obj)
    col.objects.link(obj)


def activate(obj):
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def smoothstep(e0, e1, x):
    t = max(0.0, min(1.0, (x - e0) / (e1 - e0)))
    return t * t * (3 - 2 * t)


def shade_smooth(obj):
    for p in obj.data.polygons:
        p.use_smooth = True


# ---------------------------------------------------------------------------
# Esqueleto base (mirando hacia +X, Z arriba, +Y = lado izquierdo)
# ---------------------------------------------------------------------------

CENTER = {
    'tail6': ((-6.30, 0, 2.05), 0.04),
    'tail5': ((-5.40, 0, 2.25), 0.12),
    'tail4': ((-4.40, 0, 2.45), 0.22),
    'tail3': ((-3.40, 0, 2.65), 0.34),
    'tail2': ((-2.40, 0, 2.85), 0.48),
    'tail1': ((-1.60, 0, 3.00), 0.62),
    'pelvis': ((-0.80, 0, 3.08), 0.86),
    'belly': ((0.20, 0, 3.05), 0.88),
    'chest': ((1.10, 0, 3.15), 0.78),
    'neck1': ((1.85, 0, 3.50), 0.60),
    'neck2': ((2.30, 0, 3.85), 0.52),
    'head': ((2.75, 0, 4.02), 0.58),
    'head2': ((3.32, 0, 3.98), 0.50),
    'snout': ((3.90, 0, 3.86), 0.38),
    'snout_tip': ((4.38, 0, 3.76), 0.26),
}

LEG = {
    'hip': ((-0.68, 0.36, 2.92), 0.82),
    'thigh': ((-0.42, 0.58, 2.20), 0.64),
    'knee': ((-0.18, 0.68, 1.52), 0.38),
    'shin': ((-0.52, 0.66, 1.05), 0.28),
    'ankle': ((-0.82, 0.64, 0.58), 0.19),
    'ball': ((-0.42, 0.64, 0.16), 0.15),
    'toe_in': ((0.10, 0.44, 0.08), 0.075),
    'toe_mid': ((0.30, 0.66, 0.08), 0.085),
    'toe_out': ((0.06, 0.86, 0.08), 0.075),
}

ARM = {
    'shoulder': ((1.35, 0.45, 2.75), 0.17),
    'elbow': ((1.45, 0.55, 2.35), 0.10),
    'wrist': ((1.75, 0.50, 2.25), 0.07),
    'finger1': ((1.95, 0.45, 2.15), 0.04),
    'finger2': ((1.95, 0.56, 2.17), 0.04),
}

JAW = {
    'jaw_hinge': ((2.62, 0, 3.58), 0.36),
    'jaw_mid': ((3.25, 0, 3.36), 0.29),
    'jaw_front': ((3.85, 0, 3.16), 0.21),
    'jaw_tip': ((4.24, 0, 3.06), 0.15),
}

SPINE_ORDER = ['tail6', 'tail5', 'tail4', 'tail3', 'tail2', 'tail1', 'pelvis',
               'belly', 'chest', 'neck1', 'neck2', 'head', 'head2', 'snout', 'snout_tip']


def skeleton_points():
    """Devuelve dict nombre -> (Vector, radio) con los lados .L/.R expandidos."""
    pts = {k: (Vector(v[0]), v[1]) for k, v in CENTER.items()}
    for side, s in (('L', 1), ('R', -1)):
        for k, (p, r) in LEG.items():
            pts[f'{k}.{side}'] = (Vector((p[0], p[1] * s, p[2])), r)
        for k, (p, r) in ARM.items():
            pts[f'{k}.{side}'] = (Vector((p[0], p[1] * s, p[2])), r)
    return pts


def skeleton_edges():
    edges = list(zip(SPINE_ORDER[:-1], SPINE_ORDER[1:]))
    for side in ('L', 'R'):
        leg = ['pelvis'] + [f'{k}.{side}' for k in ('hip', 'thigh', 'knee', 'shin', 'ankle', 'ball')]
        edges += list(zip(leg[:-1], leg[1:]))
        edges += [(f'ball.{side}', f'{t}.{side}') for t in ('toe_in', 'toe_mid', 'toe_out')]
        arm = ['chest'] + [f'{k}.{side}' for k in ('shoulder', 'elbow', 'wrist')]
        edges += list(zip(arm[:-1], arm[1:]))
        edges += [(f'wrist.{side}', f'{f}.{side}') for f in ('finger1', 'finger2')]
    return edges


def skin_object(name, points, edges, root, col):
    names = list(points.keys())
    index = {n: i for i, n in enumerate(names)}
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([points[n][0] for n in names],
                     [(index[a], index[b]) for a, b in edges], [])
    obj = bpy.data.objects.new(name, mesh)
    col.objects.link(obj)
    activate(obj)
    bpy.ops.object.modifier_add(type='SKIN')
    mod = obj.modifiers[-1]
    mod.branch_smoothing = 0.6
    mod.use_smooth_shade = True
    for n in names:
        sv = mesh.skin_vertices[0].data[index[n]]
        r = points[n][1]
        sv.radius = (r, r)
        sv.use_root = (n == root)
    bpy.ops.object.modifier_apply(modifier=mod.name)
    # Un nivel de subdivisión aplicado para tener una base orgánica editable
    sub = obj.modifiers.new("Base", 'SUBSURF')
    sub.levels = 1
    bpy.ops.object.modifier_apply(modifier=sub.name)
    shade_smooth(obj)
    return obj


def narrow_head(obj, start=2.2, end=2.8, amount=0.22, zmin=3.2):
    """Estrecha lateralmente el cráneo (los T. rex tienen el hocico alto y estrecho)."""
    for v in obj.data.vertices:
        if v.co.z > zmin:
            v.co.y *= 1.0 - amount * smoothstep(start, end, v.co.x)


def add_subsurf(obj, viewport=2, render=2):
    m = obj.modifiers.new("Subdivision", 'SUBSURF')
    m.levels = viewport
    m.render_levels = render
    return m


# ---------------------------------------------------------------------------
# Accesorios generados con bmesh
# ---------------------------------------------------------------------------

def orient(direction, up=Vector((0, 0, 1))):
    z = direction.normalized()
    if abs(z.dot(up)) > 0.98:
        up = Vector((1, 0, 0))
    x = up.cross(z).normalized()
    y = z.cross(x)
    return Matrix((x, y, z)).transposed().to_4x4()


def cone_into(bm, base, direction, length, radius, flatten=1.0, bend=0.0, segments=10):
    """Cono (diente, garra, escama) con base en `base` apuntando en `direction`."""
    geom = bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=segments,
                                 radius1=radius, radius2=0.0, depth=length)
    verts = geom['verts']
    # curvatura: desplaza los vértices proporcionalmente a la altura (en -Y local)
    for v in verts:
        t = (v.co.z + length / 2) / length
        v.co.y = v.co.y * flatten - bend * length * t * t
    m = Matrix.Translation(base + direction.normalized() * (length / 2)) @ orient(direction)
    bmesh.ops.transform(bm, matrix=m, verts=verts)


def mesh_object(name, bm, col):
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    obj = bpy.data.objects.new(name, mesh)
    col.objects.link(obj)
    return obj


def bvh_of(obj):
    dg = bpy.context.evaluated_depsgraph_get()
    return BVHTree.FromObject(obj, dg)


def cast(bvh, origin, direction):
    loc, nor, _, _ = bvh.ray_cast(Vector(origin), Vector(direction).normalized())
    return loc, nor


def centerline(table, order, x):
    """Interpola (z, radio) de la línea central de un esqueleto en la coordenada x."""
    pts = sorted((Vector(table[k][0]), table[k][1]) for k in order)
    pts = sorted(pts, key=lambda p: p[0].x)
    for (a, ra), (b, rb) in zip(pts[:-1], pts[1:]):
        if a.x <= x <= b.x:
            t = (x - a.x) / (b.x - a.x)
            return a.z + (b.z - a.z) * t, ra + (rb - ra) * t
    return pts[-1][0].z, pts[-1][1]


def build_teeth(body, jaw, col):
    bvh_body, bvh_jaw = bvh_of(body), bvh_of(jaw)
    upper, lower = bmesh.new(), bmesh.new()
    head_keys = ['head', 'head2', 'snout', 'snout_tip']
    jaw_keys = list(JAW.keys())

    for side in (1, -1):
        # Dientes superiores: bajo el labio del cráneo, apuntando hacia abajo
        xs = [3.05 + i * 0.14 for i in range(10)]
        for i, x in enumerate(xs):
            zc, _ = centerline(CENTER, head_keys, x)
            hit, _ = cast(bvh_body, (x, side * 3, zc - 0.05), (0, -side, 0))
            if hit is None:
                continue
            w = abs(hit.y)
            loc, _ = cast(bvh_body, (x, side * w * 0.72, zc - 2), (0, 0, 1))
            if loc is None:
                continue
            size = 0.75 + 0.5 * math.sin(math.pi * (i + 1) / (len(xs) + 1))
            size *= random.uniform(0.85, 1.1)
            d = Vector((-0.25, side * 0.12, -1))
            cone_into(upper, loc + Vector((0, 0, 0.035)), d, 0.17 * size, 0.038 * size,
                      flatten=0.65, bend=0.12)
        # Dientes inferiores: sobre la mandíbula, apuntando hacia arriba
        xs = [3.15 + i * 0.13 for i in range(8)]
        for i, x in enumerate(xs):
            zc, _ = centerline(JAW, jaw_keys, x)
            hit, _ = cast(bvh_jaw, (x, side * 3, zc), (0, -side, 0))
            if hit is None:
                continue
            w = abs(hit.y)
            loc, _ = cast(bvh_jaw, (x, side * w * 0.7, zc + 2), (0, 0, -1))
            if loc is None:
                continue
            size = (0.7 + 0.4 * math.sin(math.pi * (i + 1) / (len(xs) + 1))) * random.uniform(0.85, 1.1)
            d = Vector((-0.2, side * 0.1, 1))
            cone_into(lower, loc - Vector((0, 0, 0.03)), d, 0.12 * size, 0.03 * size,
                      flatten=0.65, bend=0.1)
    return mesh_object("Dientes_Superiores", upper, col), mesh_object("Dientes_Inferiores", lower, col)


def build_claws(col):
    bm = bmesh.new()
    pts = skeleton_points()
    for side in ('L', 'R'):
        ball = pts[f'ball.{side}'][0]
        for toe in ('toe_in', 'toe_mid', 'toe_out'):
            tip, r = pts[f'{toe}.{side}']
            d = (tip - ball).normalized()
            d.z -= 0.35
            cone_into(bm, tip - d.normalized() * 0.02, d, 0.24, r * 1.1, flatten=0.8, bend=0.25)
        wrist = pts[f'wrist.{side}'][0]
        for f in ('finger1', 'finger2'):
            tip, r = pts[f'{f}.{side}']
            d = (tip - wrist).normalized()
            d.z -= 0.4
            cone_into(bm, tip, d, 0.11, r * 1.1, flatten=0.8, bend=0.3)
    return mesh_object("Garras", bm, col)


def build_scutes(body, col):
    """Hilera doble de osteodermos a lo largo del lomo y la cola."""
    bvh = bvh_of(body)
    bm = bmesh.new()
    x = -5.6
    while x < 2.6:
        zc, r = centerline(CENTER, SPINE_ORDER, x)
        for side in (1, -1):
            y = side * r * 0.22
            loc, nor = cast(bvh, (x, y, zc + 3), (0, 0, -1))
            if loc is None:
                continue
            s = max(0.25, min(1.0, r / 0.8)) * random.uniform(0.8, 1.15)
            d = (nor + Vector((-0.4, 0, 0.6))).normalized()
            cone_into(bm, loc - nor * 0.03, d, 0.11 * s, 0.09 * s, flatten=0.6, bend=-0.2,
                      segments=8)
        x += 0.16 + 0.05 * random.random()
    obj = mesh_object("Escamas_Dorsales", bm, col)
    shade_smooth(obj)
    return obj


def build_eyes(body, col):
    bvh = bvh_of(body)
    eyes = []
    for side, tag in ((1, 'L'), (-1, 'R')):
        hit, nor = cast(bvh, (2.92, side * 3, 4.24), (0, -side, 0))
        n = (nor + Vector((0.45, 0, 0.15))).normalized()
        radius = 0.095
        bm = bmesh.new()
        bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=16, radius=1.0)
        obj = mesh_object(f"Ojo.{tag}", bm, col)
        shade_smooth(obj)
        obj.matrix_world = (Matrix.Translation(hit - n * 0.035) @ orient(n)
                            @ Matrix.Diagonal((radius, radius, radius, 1)))
        # Párpado / arco superciliar: pequeño engrosamiento de piel sobre el ojo
        bm = bmesh.new()
        bmesh.ops.create_uvsphere(bm, u_segments=24, v_segments=12, radius=1.0)
        brow = mesh_object(f"Ceja.{tag}", bm, col)
        shade_smooth(brow)
        brow.matrix_world = (Matrix.Translation(hit + Vector((-0.02, 0, 0.09)) - n * 0.03)
                             @ orient(Vector((1, 0, 0)))
                             @ Matrix.Diagonal((0.07, 0.06, 0.19, 1)))
        eyes.append((obj, brow))
    return eyes


def build_tongue(col):
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=24, v_segments=12, radius=1.0)
    obj = mesh_object("Lengua", bm, col)
    shade_smooth(obj)
    obj.matrix_world = (Matrix.Translation((3.4, 0, 3.5)) @ Matrix.Rotation(math.radians(18), 4, 'Y')
                        @ Matrix.Diagonal((0.55, 0.16, 0.07, 1)))
    return obj


# ---------------------------------------------------------------------------
# Materiales
# ---------------------------------------------------------------------------

class NodeBuilder:
    def __init__(self, mat):
        mat.use_nodes = True
        self.nt = mat.node_tree
        self.nt.nodes.clear()
        self.x = 0

    def add(self, kind, loc=(0, 0), **props):
        n = self.nt.nodes.new(kind)
        n.location = loc
        for k, v in props.items():
            setattr(n, k, v)
        return n

    def link(self, a, b):
        self.nt.links.new(a, b)


def ramp(nb, loc, stops):
    r = nb.add('ShaderNodeValToRGB', loc)
    els = r.color_ramp.elements
    while len(els) > len(stops):
        els.remove(els[-1])
    while len(els) < len(stops):
        els.new(0.5)
    for el, (pos, col) in zip(els, stops):
        el.position = pos
        el.color = col if len(col) == 4 else (*col, 1.0)
    return r


def mix_rgb(nb, loc, blend='MIX', fac=None):
    m = nb.add('ShaderNodeMix', loc, data_type='RGBA', blend_type=blend)
    if fac is not None:
        m.inputs[0].default_value = fac
    return m


def principled(nb, loc=(600, 0), **inputs):
    p = nb.add('ShaderNodeBsdfPrincipled', loc)
    for k, v in inputs.items():
        p.inputs[k].default_value = v
    out = nb.add('ShaderNodeOutputMaterial', (loc[0] + 350, loc[1]))
    nb.link(p.outputs[0], out.inputs['Surface'])
    return p


def material_skin(name="Piel_Dinosaurio", dark=(0.035, 0.05, 0.022), mid=(0.11, 0.12, 0.05),
                  belly=(0.42, 0.36, 0.22)):
    mat = bpy.data.materials.new(name)
    nb = NodeBuilder(mat)
    tc = nb.add('ShaderNodeTexCoord', (-1600, 0))

    # Gradiente dorso -> vientre usando la altura relativa al eje del cuerpo
    sep = nb.add('ShaderNodeSeparateXYZ', (-1400, 300))
    nb.link(tc.outputs['Object'], sep.inputs[0])
    mr_h = nb.add('ShaderNodeMapRange', (-1200, 300))
    mr_h.inputs['From Min'].default_value = 1.0
    mr_h.inputs['From Max'].default_value = 3.8
    nb.link(sep.outputs['Z'], mr_h.inputs['Value'])
    geo = nb.add('ShaderNodeNewGeometry', (-1600, 700))
    sep_n = nb.add('ShaderNodeSeparateXYZ', (-1400, 700))
    nb.link(geo.outputs['Normal'], sep_n.inputs[0])
    mr_n = nb.add('ShaderNodeMapRange', (-1200, 700))
    mr_n.inputs['From Min'].default_value = -0.7
    mr_n.inputs['From Max'].default_value = 0.8
    nb.link(sep_n.outputs['Z'], mr_n.inputs['Value'])
    mr = nb.add('ShaderNodeMix', (-1000, 650), data_type='FLOAT')
    mr.inputs[0].default_value = 0.4
    nb.link(mr_n.outputs['Result'], mr.inputs[2])
    nb.link(mr_h.outputs['Result'], mr.inputs[3])
    noise_warp = nb.add('ShaderNodeTexNoise', (-1400, 550))
    noise_warp.inputs['Scale'].default_value = 3.0
    noise_warp.inputs['Detail'].default_value = 4.0
    nb.link(tc.outputs['Object'], noise_warp.inputs['Vector'])
    add = nb.add('ShaderNodeMath', (-1000, 400), operation='MULTIPLY_ADD')
    add.inputs[1].default_value = 0.18
    nb.link(noise_warp.outputs['Fac'], add.inputs[0])
    nb.link(mr.outputs[0], add.inputs[2])
    base_ramp = ramp(nb, (-800, 400), [(0.2, belly), (0.48, mid), (0.82, dark)])
    nb.link(add.outputs[0], base_ramp.inputs['Fac'])

    # Rayas transversales oscuras en el lomo
    wave = nb.add('ShaderNodeTexWave', (-1200, 0), wave_type='BANDS', bands_direction='X')
    wave.inputs['Scale'].default_value = 0.55
    wave.inputs['Distortion'].default_value = 6.0
    wave.inputs['Detail'].default_value = 3.0
    nb.link(tc.outputs['Object'], wave.inputs['Vector'])
    stripe = ramp(nb, (-1000, 0), [(0.55, (1, 1, 1)), (0.8, (0.35, 0.33, 0.3))])
    nb.link(wave.outputs['Fac'], stripe.inputs['Fac'])
    stripe_mask = ramp(nb, (-1000, -250), [(0.5, (0, 0, 0)), (0.8, (1, 1, 1))])
    nb.link(mr.outputs[0], stripe_mask.inputs['Fac'])
    stripe_mix = mix_rgb(nb, (-600, 100), 'MULTIPLY')
    nb.link(stripe_mask.outputs['Color'], stripe_mix.inputs[0])
    nb.link(base_ramp.outputs['Color'], stripe_mix.inputs[6])
    nb.link(stripe.outputs['Color'], stripe_mix.inputs[7])

    # Escamas: dos capas de Voronoi (distancia al borde) para surcos entre escamas
    vor_s = nb.add('ShaderNodeTexVoronoi', (-1200, -500), feature='DISTANCE_TO_EDGE')
    vor_s.inputs['Scale'].default_value = 38.0
    vor_s.inputs['Randomness'].default_value = 0.9
    nb.link(tc.outputs['Object'], vor_s.inputs['Vector'])
    vor_l = nb.add('ShaderNodeTexVoronoi', (-1200, -800), feature='DISTANCE_TO_EDGE')
    vor_l.inputs['Scale'].default_value = 11.0
    nb.link(tc.outputs['Object'], vor_l.inputs['Vector'])
    groove_s = ramp(nb, (-950, -500), [(0.0, (0, 0, 0)), (0.12, (1, 1, 1))])
    nb.link(vor_s.outputs['Distance'], groove_s.inputs['Fac'])
    groove_l = ramp(nb, (-950, -800), [(0.0, (0, 0, 0)), (0.08, (1, 1, 1))])
    nb.link(vor_l.outputs['Distance'], groove_l.inputs['Fac'])
    scales = mix_rgb(nb, (-650, -600), 'MULTIPLY', fac=1.0)
    nb.link(groove_s.outputs['Color'], scales.inputs[6])
    nb.link(groove_l.outputs['Color'], scales.inputs[7])

    # Arrugas finas
    wrinkle = nb.add('ShaderNodeTexNoise', (-1200, -1100))
    wrinkle.inputs['Scale'].default_value = 90.0
    wrinkle.inputs['Detail'].default_value = 8.0
    nb.link(tc.outputs['Object'], wrinkle.inputs['Vector'])

    # Oscurecer surcos (cavidad)
    cavity = mix_rgb(nb, (-350, 0), 'MULTIPLY')
    cavity_fac = ramp(nb, (-650, -300), [(0.0, (0.55, 0.55, 0.55)), (1.0, (1, 1, 1))])
    nb.link(scales.outputs[2], cavity_fac.inputs['Fac'])
    cavity.inputs[0].default_value = 1.0
    nb.link(stripe_mix.outputs[2], cavity.inputs[6])
    nb.link(cavity_fac.outputs['Color'], cavity.inputs[7])

    # Bump combinado
    bump1 = nb.add('ShaderNodeBump', (-300, -600))
    bump1.inputs['Strength'].default_value = 0.55
    bump1.inputs['Distance'].default_value = 0.02
    nb.link(scales.outputs[2], bump1.inputs['Height'])
    bump2 = nb.add('ShaderNodeBump', (-100, -700))
    bump2.inputs['Strength'].default_value = 0.15
    bump2.inputs['Distance'].default_value = 0.005
    nb.link(wrinkle.outputs['Fac'], bump2.inputs['Height'])
    nb.link(bump1.outputs['Normal'], bump2.inputs['Normal'])

    # Rugosidad variable: escamas algo más brillantes que los surcos
    rough = ramp(nb, (-350, -300), [(0.0, (0.85, 0.85, 0.85)), (1.0, (0.5, 0.5, 0.5))])
    nb.link(scales.outputs[2], rough.inputs['Fac'])

    p = principled(nb, (200, 0))
    p.inputs['Subsurface Weight'].default_value = 0.08
    p.inputs['Subsurface Radius'].default_value = (1.0, 0.35, 0.2)
    p.inputs['Subsurface Scale'].default_value = 0.05
    p.inputs['Coat Weight'].default_value = 0.08
    p.inputs['Coat Roughness'].default_value = 0.35
    nb.link(cavity.outputs[2], p.inputs['Base Color'])
    nb.link(rough.outputs['Color'], p.inputs['Roughness'])
    nb.link(bump2.outputs['Normal'], p.inputs['Normal'])
    return mat


def material_simple(name, color, roughness, **extra):
    mat = bpy.data.materials.new(name)
    nb = NodeBuilder(mat)
    p = principled(nb, (0, 0))
    p.inputs['Base Color'].default_value = (*color, 1.0)
    p.inputs['Roughness'].default_value = roughness
    for k, v in extra.items():
        p.inputs[k].default_value = v
    return mat, nb, p


def material_teeth():
    mat, nb, p = material_simple("Dientes", (0.78, 0.7, 0.52), 0.32,
                                 **{'Subsurface Weight': 0.25, 'Subsurface Scale': 0.02})
    # Raíz más amarillenta, punta más clara
    tc = nb.add('ShaderNodeTexCoord', (-900, 0))
    noise = nb.add('ShaderNodeTexNoise', (-700, 0))
    noise.inputs['Scale'].default_value = 25.0
    nb.link(tc.outputs['Object'], noise.inputs['Vector'])
    r = ramp(nb, (-450, 0), [(0.3, (0.55, 0.45, 0.28)), (0.7, (0.86, 0.8, 0.64))])
    nb.link(noise.outputs['Fac'], r.inputs['Fac'])
    nb.link(r.outputs['Color'], p.inputs['Base Color'])
    return mat


def material_eye():
    mat = bpy.data.materials.new("Ojo")
    nb = NodeBuilder(mat)
    tc = nb.add('ShaderNodeTexCoord', (-1400, 0))
    sep = nb.add('ShaderNodeSeparateXYZ', (-1200, 0))
    nb.link(tc.outputs['Object'], sep.inputs[0])
    # Pupila vertical rasgada: elipse (x/0.14)^2 + (y/0.75)^2 < 1 en la cara frontal (z>0)
    sx = nb.add('ShaderNodeMath', (-1000, 150), operation='DIVIDE')
    sx.inputs[1].default_value = 0.14
    nb.link(sep.outputs['X'], sx.inputs[0])
    sy = nb.add('ShaderNodeMath', (-1000, -50), operation='DIVIDE')
    sy.inputs[1].default_value = 0.75
    nb.link(sep.outputs['Y'], sy.inputs[0])
    px = nb.add('ShaderNodeMath', (-850, 150), operation='POWER')
    px.inputs[1].default_value = 2.0
    nb.link(sx.outputs[0], px.inputs[0])
    py = nb.add('ShaderNodeMath', (-850, -50), operation='POWER')
    py.inputs[1].default_value = 2.0
    nb.link(sy.outputs[0], py.inputs[0])
    s = nb.add('ShaderNodeMath', (-700, 50), operation='ADD')
    nb.link(px.outputs[0], s.inputs[0])
    nb.link(py.outputs[0], s.inputs[1])
    pupil = ramp(nb, (-550, 50), [(0.85, (0, 0, 0)), (1.0, (1, 1, 1))])
    nb.link(s.outputs[0], pupil.inputs['Fac'])
    # Iris: gradiente radial ámbar con vetas
    noise = nb.add('ShaderNodeTexNoise', (-1000, -300))
    noise.inputs['Scale'].default_value = 12.0
    noise.inputs['Detail'].default_value = 6.0
    nb.link(tc.outputs['Object'], noise.inputs['Vector'])
    iris_grad = nb.add('ShaderNodeMath', (-750, -300), operation='MULTIPLY_ADD')
    iris_grad.inputs[1].default_value = 0.35
    nb.link(noise.outputs['Fac'], iris_grad.inputs[0])
    nb.link(sep.outputs['Z'], iris_grad.inputs[2])
    iris = ramp(nb, (-550, -300), [(0.35, (0.12, 0.04, 0.0)), (0.75, (0.6, 0.26, 0.02)),
                                   (1.05, (0.85, 0.55, 0.12))])
    nb.link(iris_grad.outputs[0], iris.inputs['Fac'])
    col = mix_rgb(nb, (-250, 0), 'MULTIPLY', fac=1.0)
    nb.link(iris.outputs['Color'], col.inputs[6])
    nb.link(pupil.outputs['Color'], col.inputs[7])
    p = principled(nb, (0, 0))
    p.inputs['Roughness'].default_value = 0.3
    p.inputs['Coat Weight'].default_value = 1.0
    p.inputs['Coat Roughness'].default_value = 0.02
    p.inputs['Emission Strength'].default_value = 0.15
    nb.link(col.outputs[2], p.inputs['Base Color'])
    nb.link(col.outputs[2], p.inputs['Emission Color'])
    return mat


def material_ground():
    mat = bpy.data.materials.new("Suelo_Tierra")
    nb = NodeBuilder(mat)
    tc = nb.add('ShaderNodeTexCoord', (-1200, 0))
    n1 = nb.add('ShaderNodeTexNoise', (-1000, 200))
    n1.inputs['Scale'].default_value = 0.35
    n1.inputs['Detail'].default_value = 8.0
    nb.link(tc.outputs['Object'], n1.inputs['Vector'])
    n2 = nb.add('ShaderNodeTexNoise', (-1000, -200))
    n2.inputs['Scale'].default_value = 6.0
    n2.inputs['Detail'].default_value = 10.0
    nb.link(tc.outputs['Object'], n2.inputs['Vector'])
    c = ramp(nb, (-750, 200), [(0.35, (0.09, 0.065, 0.04)), (0.5, (0.17, 0.13, 0.08)),
                               (0.65, (0.1, 0.12, 0.04))])
    nb.link(n1.outputs['Fac'], c.inputs['Fac'])
    pebbles = nb.add('ShaderNodeTexVoronoi', (-1000, -500))
    pebbles.inputs['Scale'].default_value = 14.0
    nb.link(tc.outputs['Object'], pebbles.inputs['Vector'])
    peb = ramp(nb, (-750, -500), [(0.0, (1, 1, 1)), (0.3, (0, 0, 0))])
    nb.link(pebbles.outputs['Distance'], peb.inputs['Fac'])
    hmix = nb.add('ShaderNodeMath', (-500, -300), operation='MULTIPLY_ADD')
    hmix.inputs[1].default_value = 0.5
    nb.link(peb.outputs['Color'], hmix.inputs[0])
    nb.link(n2.outputs['Fac'], hmix.inputs[2])
    bump = nb.add('ShaderNodeBump', (-250, -300))
    bump.inputs['Strength'].default_value = 0.6
    nb.link(hmix.outputs[0], bump.inputs['Height'])
    p = principled(nb, (0, 0))
    p.inputs['Roughness'].default_value = 0.95
    p.inputs['Specular IOR Level'].default_value = 0.2
    nb.link(c.outputs['Color'], p.inputs['Base Color'])
    nb.link(bump.outputs['Normal'], p.inputs['Normal'])
    return mat


def material_rock():
    mat = bpy.data.materials.new("Roca")
    nb = NodeBuilder(mat)
    tc = nb.add('ShaderNodeTexCoord', (-1000, 0))
    n = nb.add('ShaderNodeTexNoise', (-800, 0))
    n.inputs['Scale'].default_value = 4.0
    n.inputs['Detail'].default_value = 12.0
    n.inputs['Roughness'].default_value = 0.65
    nb.link(tc.outputs['Object'], n.inputs['Vector'])
    c = ramp(nb, (-550, 100), [(0.35, (0.05, 0.045, 0.04)), (0.6, (0.16, 0.15, 0.13)),
                               (0.75, (0.1, 0.12, 0.05))])
    nb.link(n.outputs['Fac'], c.inputs['Fac'])
    bump = nb.add('ShaderNodeBump', (-300, -200))
    bump.inputs['Strength'].default_value = 0.8
    nb.link(n.outputs['Fac'], bump.inputs['Height'])
    p = principled(nb, (0, 0))
    p.inputs['Roughness'].default_value = 0.85
    nb.link(c.outputs['Color'], p.inputs['Base Color'])
    nb.link(bump.outputs['Normal'], p.inputs['Normal'])
    return mat


def assign(obj, mat):
    obj.data.materials.clear()
    obj.data.materials.append(mat)


# ---------------------------------------------------------------------------
# Rig
# ---------------------------------------------------------------------------

def build_rig(col):
    pts = skeleton_points()
    P = lambda k: pts[k][0]
    J = {k: Vector(v[0]) for k, v in JAW.items()}

    arm_data = bpy.data.armatures.new("Rig_TRex")
    arm_data.display_type = 'OCTAHEDRAL'
    rig = bpy.data.objects.new("Rig_TRex", arm_data)
    col.objects.link(rig)
    rig.show_in_front = True
    activate(rig)
    bpy.ops.object.mode_set(mode='EDIT')
    eb = arm_data.edit_bones

    def bone(name, head, tail, parent=None, connect=False, deform=True):
        b = eb.new(name)
        b.head, b.tail = Vector(head), Vector(tail)
        b.roll = 0.0
        if parent:
            b.parent = eb[parent]
            b.use_connect = connect
        b.use_deform = deform
        return b

    bone('root', (0, 0, 0), (1.5, 0, 0), deform=False)
    bone('hips', P('pelvis'), P('pelvis') + Vector((0, 0, -0.6)), 'root')
    bone('spine_01', P('pelvis'), P('belly'), 'hips')
    bone('spine_02', P('belly'), P('chest'), 'spine_01', True)
    bone('spine_03', P('chest'), P('neck1'), 'spine_02', True)
    bone('neck_01', P('neck1'), P('neck2'), 'spine_03', True)
    bone('neck_02', P('neck2'), P('head'), 'neck_01', True)
    bone('head', P('head'), P('snout_tip'), 'neck_02', True)
    bone('jaw', J['jaw_hinge'], J['jaw_tip'], 'head')
    tail = ['pelvis', 'tail1', 'tail2', 'tail3', 'tail4', 'tail5', 'tail6']
    for i, (a, b) in enumerate(zip(tail[:-1], tail[1:]), 1):
        bone(f'tail_{i:02d}', P(a), P(b), 'hips' if i == 1 else f'tail_{i - 1:02d}', i > 1)
    for s in ('L', 'R'):
        bone(f'thigh.{s}', P(f'hip.{s}'), P(f'knee.{s}'), 'hips')
        bone(f'shin.{s}', P(f'knee.{s}'), P(f'ankle.{s}'), f'thigh.{s}', True)
        bone(f'foot.{s}', P(f'ankle.{s}'), P(f'ball.{s}'), f'shin.{s}', True)
        bone(f'toes.{s}', P(f'ball.{s}'), P(f'toe_mid.{s}'), f'foot.{s}', True)
        bone(f'upperarm.{s}', P(f'shoulder.{s}'), P(f'elbow.{s}'), 'spine_02')
        bone(f'forearm.{s}', P(f'elbow.{s}'), P(f'wrist.{s}'), f'upperarm.{s}', True)
        fingers = (P(f'finger1.{s}') + P(f'finger2.{s}')) / 2
        bone(f'hand.{s}', P(f'wrist.{s}'), fingers, f'forearm.{s}', True)
    bpy.ops.object.mode_set(mode='OBJECT')
    return rig


def skin_to_rig(body, rig):
    """Pesos automáticos (bone heat). La mandíbula se excluye: tiene su propio objeto."""
    rig.data.bones['jaw'].use_deform = False
    bpy.ops.object.select_all(action='DESELECT')
    body.select_set(True)
    rig.select_set(True)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.parent_set(type='ARMATURE_AUTO')
    rig.data.bones['jaw'].use_deform = True
    # El armature debe evaluarse antes que la subdivisión
    activate(body)
    bpy.ops.object.modifier_move_to_index(modifier=body.modifiers[-1].name, index=0)
    if not any(len(v.groups) for v in body.data.vertices):
        print("AVISO: bone heat falló, usando envolventes")
        bpy.ops.object.parent_set(type='ARMATURE_ENVELOPE')


def transfer_weights(src, dst, rig):
    """Copia a `dst` los pesos del vértice más cercano de `src`."""
    kd = KDTree(len(src.data.vertices))
    for v in src.data.vertices:
        kd.insert(v.co, v.index)
    kd.balance()
    names = {g.index: g.name for g in src.vertex_groups}
    groups = {}
    mw = dst.matrix_world
    for v in dst.data.vertices:
        _, idx, _ = kd.find(mw @ v.co)
        for g in src.data.vertices[idx].groups:
            name = names[g.group]
            if name not in groups:
                groups[name] = dst.vertex_groups.new(name=name)
            groups[name].add([v.index], g.weight, 'REPLACE')
    parent_to_rig(dst, rig)


def bind_rigid(obj, rig, bone_name):
    vg = obj.vertex_groups.new(name=bone_name)
    vg.add([v.index for v in obj.data.vertices], 1.0, 'REPLACE')
    parent_to_rig(obj, rig)


def parent_to_rig(obj, rig):
    mw = obj.matrix_world.copy()
    obj.parent = rig
    obj.matrix_parent_inverse = rig.matrix_world.inverted()
    obj.matrix_world = mw
    mod = obj.modifiers.new("Armature", 'ARMATURE')
    mod.object = rig
    # El armature debe evaluarse antes que la subdivisión
    activate(obj)
    bpy.ops.object.modifier_move_to_index(modifier=mod.name, index=0)


def pose_rig(rig):
    def rot(name, axis, deg):
        pb = rig.pose.bones[name]
        m = pb.bone.matrix_local.to_3x3()
        r = m.inverted() @ Matrix.Rotation(math.radians(deg), 3, axis) @ m
        pb.rotation_mode = 'QUATERNION'
        pb.rotation_quaternion = (pb.rotation_quaternion.to_matrix() @ r).to_quaternion()

    # Gira cuello y cabeza hacia la cámara, cabeza alzada y fauces abiertas (rugido)
    rot('neck_01', 'Z', 7)
    rot('neck_02', 'Z', 9)
    rot('head', 'Z', 6)
    rot('neck_02', 'Y', -4)
    rot('head', 'Y', -6)
    rot('jaw', 'Y', 14)
    # Cola en curva suave
    for i, deg in enumerate((-3, -5, -6, -7, -7, -6), 1):
        rot(f'tail_{i:02d}', 'Z', deg)
    for i, deg in enumerate((2, 3, 3, 2), 2):
        rot(f'tail_{i:02d}', 'Y', -deg)
    # Brazos ligeramente flexionados
    for s in ('L', 'R'):
        rot(f'forearm.{s}', 'Y', -15)


# ---------------------------------------------------------------------------
# Escenario, luces y cámara
# ---------------------------------------------------------------------------

def build_environment(col):
    bpy.ops.mesh.primitive_plane_add(size=2000, location=(0, 0, 0))
    ground = bpy.context.active_object
    ground.name = "Suelo"
    link(ground, col)
    assign(ground, material_ground())

    rock_mat = material_rock()
    tex = bpy.data.textures.new("Ruido_Roca", 'CLOUDS')
    tex.noise_scale = 0.6
    tex.noise_depth = 4
    spots = [((2.6, 2.4), 0.55), ((-3.6, 3.0), 0.8), ((6.2, -1.8), 1.1), ((-2.4, -3.2), 0.6),
             ((1.2, -2.8), 0.4), ((-8.0, 1.2), 1.4), ((4.4, 4.2), 0.35), ((-5.2, -1.6), 0.45)]
    for i, ((x, y), s) in enumerate(spots):
        bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=4, radius=1, location=(x, y, s * 0.25))
        rock = bpy.context.active_object
        rock.name = f"Roca.{i:02d}"
        rock.scale = (s * random.uniform(0.9, 1.4), s * random.uniform(0.8, 1.2), s * random.uniform(0.5, 0.75))
        rock.rotation_euler = (0, 0, random.uniform(0, math.tau))
        disp = rock.modifiers.new("Desplazamiento", 'DISPLACE')
        disp.texture = tex
        disp.strength = 0.35
        disp.texture_coords = 'GLOBAL'
        link(rock, col)
        assign(rock, rock_mat)
        shade_smooth(rock)


def build_lights_camera(col, scene):
    def light(name, kind, loc, target, energy, color, size=None):
        data = bpy.data.lights.new(name, kind)
        data.energy = energy
        data.color = color
        if size is not None:
            if kind == 'SUN':
                data.angle = math.radians(size)
            else:
                data.size = size
        obj = bpy.data.objects.new(name, data)
        col.objects.link(obj)
        obj.location = loc
        obj.rotation_euler = (Vector(target) - Vector(loc)).to_track_quat('-Z', 'Y').to_euler()
        return obj

    az, el = math.radians(SUN_AZIMUTH), math.radians(SUN_ELEVATION)
    sun_pos = Vector((math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el))) * 20
    light("Luz_Sol", 'SUN', sun_pos, (0, 0, 0), 5.0, (1.0, 0.88, 0.72), 2.0)
    fill = light("Luz_Relleno", 'AREA', (7, 14, 9), (0, 0, 2.5), 900, (0.7, 0.82, 1.0), 8.0)
    fill.data.shape = 'DISK'
    rim = light("Luz_Contra", 'AREA', (-6, -7, 7), (1, 0, 3), 4000, (1.0, 0.85, 0.7), 4.0)
    # Light linking: el contraluz solo recorta la silueta del dinosaurio, no ilumina el suelo
    rim.light_linking.receiver_collection = bpy.data.collections["Dinosaurio"]

    target = bpy.data.objects.new("Camara_Objetivo", None)
    col.objects.link(target)
    target.location = (-0.5, 0.3, 2.4)

    cam_data = bpy.data.cameras.new("Camara")
    cam_data.lens = 45
    cam_data.clip_end = 2000
    cam_data.dof.use_dof = True
    cam_data.dof.aperture_fstop = 5.6
    cam = bpy.data.objects.new("Camara", cam_data)
    col.objects.link(cam)
    cam.location = (9.5, 12.0, 2.6)
    c = cam.constraints.new('TRACK_TO')
    c.target = target
    c.track_axis = 'TRACK_NEGATIVE_Z'
    c.up_axis = 'UP_Y'
    scene.camera = cam
    return cam


def build_world(scene):
    world = bpy.data.worlds.new("Cielo")
    scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()
    sky = nt.nodes.new('ShaderNodeTexSky')
    sky.sky_type = 'MULTIPLE_SCATTERING'
    sky.sun_disc = False
    sky.sun_elevation = math.radians(SUN_ELEVATION)
    sky.sun_rotation = math.radians(SUN_AZIMUTH)
    sky.air_density = 1.2
    sky.aerosol_density = 2.0
    bg = nt.nodes.new('ShaderNodeBackground')
    bg.inputs['Strength'].default_value = 0.12
    nt.links.new(sky.outputs['Color'], bg.inputs['Color'])
    out = nt.nodes.new('ShaderNodeOutputWorld')
    nt.links.new(bg.outputs[0], out.inputs['Surface'])


def setup_render(scene):
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = 256
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.use_denoising = True
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.view_settings.look = 'AgX - Medium High Contrast'
    scene.view_settings.exposure = -1.0
    scene.render.image_settings.file_format = 'PNG'


# ---------------------------------------------------------------------------
# Ensamblado
# ---------------------------------------------------------------------------

def main():
    scene = reset_scene()
    col_dino = make_collection("Dinosaurio")
    col_rig = make_collection("Rig")
    col_env = make_collection("Escenario")
    col_lc = make_collection("Luces_Camara")

    body = skin_object("TRex_Cuerpo", skeleton_points(), skeleton_edges(), 'pelvis', col_dino)
    narrow_head(body)
    jaw = skin_object("TRex_Mandibula", {k: (Vector(p), r) for k, (p, r) in JAW.items()},
                      list(zip(list(JAW)[:-1], list(JAW)[1:])), 'jaw_hinge', col_dino)
    narrow_head(jaw, start=2.5, end=3.0, amount=0.2, zmin=0.0)
    for o in (body, jaw):
        add_subsurf(o, 2, 2)

    upper_teeth, lower_teeth = build_teeth(body, jaw, col_dino)
    claws = build_claws(col_dino)
    scutes = build_scutes(body, col_dino)
    eyes = build_eyes(body, col_dino)
    tongue = build_tongue(col_dino)

    skin = material_skin()
    scute_mat, _, _ = material_simple("Escamas_Dorsales", (0.05, 0.045, 0.03), 0.75)
    teeth = material_teeth()
    claw_mat, _, _ = material_simple("Garras", (0.05, 0.04, 0.03), 0.35, **{'Coat Weight': 0.4})
    mouth, _, _ = material_simple("Boca", (0.35, 0.06, 0.06), 0.4, **{'Subsurface Weight': 0.3,
                                                                       'Subsurface Scale': 0.03})
    eye_mat = material_eye()
    for o in (body, jaw):
        assign(o, skin)
    assign(upper_teeth, teeth)
    assign(lower_teeth, teeth)
    assign(claws, claw_mat)
    assign(scutes, scute_mat)
    assign(tongue, mouth)
    for eye, brow in eyes:
        assign(eye, eye_mat)
        assign(brow, skin)
        add_subsurf(brow, 1, 2)

    # Rig
    rig = build_rig(col_rig)
    skin_to_rig(body, rig)
    for o in (upper_teeth, claws, scutes, *[e for pair in eyes for e in pair]):
        transfer_weights(body, o, rig)
    for o in (jaw, lower_teeth, tongue):
        bind_rigid(o, rig, 'jaw')
    for o in (body, jaw):
        o.modifiers["Subdivision"].levels = 1  # viewport ligero; render a nivel 2
    pose_rig(rig)

    build_environment(col_env)
    build_lights_camera(col_lc, scene)
    build_world(scene)
    setup_render(scene)

    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = rig
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(BLEND_PATH), compress=True)
    print(f"Guardado: {BLEND_PATH}")

    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    if "--render" in argv:
        out = argv[argv.index("--render") + 1]
        pct = int(argv[argv.index("--pct") + 1]) if "--pct" in argv else 100
        samples = int(argv[argv.index("--samples") + 1]) if "--samples" in argv else 256
        scene.render.resolution_percentage = pct
        scene.cycles.samples = samples
        scene.render.filepath = out
        bpy.ops.render.render(write_still=True)
        print(f"Render: {out}")


if __name__ == "__main__":
    main()
