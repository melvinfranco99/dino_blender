"""
Escenario: llanura aluvial del Cretácico superior (formación Hell Creek) al atardecer.

* Terreno con huellas tridáctilas del propio T. rex grabadas en un mapa de alturas
  (imagen flotante generada con numpy) y barro húmedo a lo largo del rastro.
* Rocas, helechos procedurales y coníferas al fondo con perspectiva aérea.
* Cielo físico (Nishita, dispersión múltiple), sol cálido y cámara de "foto de campo".
"""

import math

import bpy
import numpy as np
from mathutils import Euler, Matrix, Vector

from . import materiales as M
from .geometria import mesh_from_arrays
from .sdf import fbm, smoothstep

SUN_ELEVATION = 27.0   # grados
SUN_AZIMUTH = 128.0    # grados desde +X hacia +Y (dirección hacia el sol)

# Huellas y mapa de alturas cercano
HMAP_MIN = np.array([-26.0, -9.0])
HMAP_SIZE = np.array([36.0, 18.0])
HMAP_RES = 0.025


def sun_vector():
    az, el = math.radians(SUN_AZIMUTH), math.radians(SUN_ELEVATION)
    return Vector((math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)))


# ---------------------------------------------------------------------------
# Terreno
# ---------------------------------------------------------------------------

def footprint_height(px, py, x0, y0, heading, depth=0.07):
    """Huella tridáctila (dedos II-IV + almohadilla metatarsal) con reborde de barro."""
    c, s = math.cos(-heading), math.sin(-heading)
    lx = (px - x0) * c - (py - y0) * s
    ly = (px - x0) * s + (py - y0) * c
    d = np.full(px.shape, 10.0, np.float32)
    # almohadilla
    d = np.minimum(d, np.sqrt((lx / 0.26) ** 2 + (ly / 0.2) ** 2) - 1.0)
    for ang, length in ((-25, 0.52), (0, 0.62), (25, 0.54)):
        a = math.radians(ang)
        dx, dy = math.cos(a), math.sin(a)
        t = np.clip((lx * dx + ly * dy - 0.05) / length, 0, 1)
        qx, qy = lx - (0.05 + t * length) * dx, ly - t * length * dy
        r = 0.085 * (1 - 0.45 * t)
        d = np.minimum(d, np.sqrt(qx * qx + qy * qy) / r - 1.0)
        # marca de la garra
        tx, ty = lx - (0.05 + length + 0.07) * dx, ly - (length + 0.07) * dy
        d = np.minimum(d, np.sqrt((tx / 0.05) ** 2 + (ty / 0.02) ** 2) - 1.0)
    inside = 1 - smoothstep(-0.25, 0.15, d)
    rim = np.exp(-((d - 0.45) / 0.35) ** 2)
    return -depth * inside + depth * 0.35 * rim, inside


def build_heightmap():
    nx, ny = (HMAP_SIZE / HMAP_RES).astype(int)
    xs = HMAP_MIN[0] + (np.arange(nx) + 0.5) * HMAP_RES
    ys = HMAP_MIN[1] + (np.arange(ny) + 0.5) * HMAP_RES
    X, Y = np.meshgrid(xs, ys)  # (ny, nx)
    H = np.zeros_like(X, np.float32)
    mask = np.zeros_like(X, np.float32)
    # rastro: pisadas alternas por detrás del animal, ligeramente curvado
    stride = 3.7
    # (x, y, profundidad): las dos primeras están bajo los pies actuales
    steps = [(0.70, 0.68, 0.03), (-0.43, -0.64, 0.03)]
    for k in range(1, 7):
        steps.append((0.70 - stride * k, 0.68, 0.075))
        steps.append((-0.43 - stride * k, -0.64, 0.075))
    for x, y, depth in steps:
        y = y + 0.012 * x * x  # curva suave del rastro
        heading = math.atan2(0.024 * x, 1.0)
        sel = (np.abs(X - x) < 1.4) & (np.abs(Y - y) < 1.2)
        h, m = footprint_height(X[sel], Y[sel], x - 0.1, y, heading, depth)
        H[sel] += h
        mask[sel] = np.maximum(mask[sel], m)
    # humedad: banda de barro siguiendo el rastro y charcos
    path = np.exp(-((Y - 0.012 * X * X) / 1.6) ** 2) * smoothstep(3.0, -1.0, X)
    wet = np.clip(path * 0.8 + 0.4 * (fbm(np.stack([X * 0.4, Y * 0.4, np.zeros_like(X)], -1), 3, seed=5) > 0.25),
                  0, 1)
    img = bpy.data.images.new("Huellas_Altura", nx, ny, alpha=True, float_buffer=True)
    px = np.stack([H, mask, wet.astype(np.float32), np.ones_like(H)], -1)
    img.pixels.foreach_set(px.astype(np.float32).ravel())
    img.pack()
    img.colorspace_settings.name = 'Non-Color'
    return img


def terrain_mesh(col):
    """Malla base de 260 m con ondulaciones suaves; el detalle lo aporta la subdivisión adaptativa."""
    n = 260
    size = 260.0
    lin = np.linspace(-size / 2, size / 2, n + 1)
    X, Y = np.meshgrid(lin - 60.0, lin - 40.0)
    pts = np.stack([X * 0.02, Y * 0.02, np.zeros_like(X)], -1)
    Z = fbm(pts, 4, seed=9) * 2.2
    r = np.sqrt(X ** 2 + Y ** 2)
    Z *= smoothstep(10.0, 45.0, r)          # llano cerca del animal
    Z += smoothstep(60.0, 130.0, r) * 3.0   # el terreno se eleva suavemente a lo lejos
    verts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    idx = np.arange((n + 1) ** 2).reshape(n + 1, n + 1)
    faces = np.stack([idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, 1:].ravel(), idx[1:, :-1].ravel()], 1)
    obj = mesh_from_arrays("Terreno", verts, faces, col)
    return obj


def ground_material_with_prints(img):
    mat = M.ground()
    nt = mat.node_tree
    tex = nt.nodes.new('ShaderNodeTexImage')
    tex.image = img
    tex.extension = 'CLIP'
    tex.interpolation = 'Cubic'
    tex.location = (-2400, 800)
    mp = nt.nodes.new('ShaderNodeMapping')
    mp.location = (-2600, 800)
    mp.inputs['Location'].default_value = (-HMAP_MIN[0] / HMAP_SIZE[0], -HMAP_MIN[1] / HMAP_SIZE[1], 0)
    mp.inputs['Scale'].default_value = (1 / HMAP_SIZE[0], 1 / HMAP_SIZE[1], 1)
    tc = next(n for n in nt.nodes if n.type == 'TEX_COORD')
    nt.links.new(tc.outputs['Object'], mp.inputs['Vector'])
    nt.links.new(mp.outputs['Vector'], tex.inputs['Vector'])
    sep = nt.nodes.new('ShaderNodeSeparateColor')
    sep.location = (-2200, 800)
    nt.links.new(tex.outputs['Color'], sep.inputs[0])
    # sustituye los atributos 'huella' y 'humedad' por los canales de la imagen
    for node in list(nt.nodes):
        if node.type == 'ATTRIBUTE' and node.attribute_name in ('huella', 'humedad'):
            ch = sep.outputs['Green'] if node.attribute_name == 'huella' else sep.outputs['Blue']
            for link in list(node.outputs['Fac'].links):
                nt.links.new(ch, link.to_socket)
            nt.nodes.remove(node)
    # suma la altura de las huellas al desplazamiento
    disp = next(n for n in nt.nodes if n.type == 'DISPLACEMENT')
    old = disp.inputs['Height'].links[0].from_socket
    add = nt.nodes.new('ShaderNodeMath')
    add.operation = 'ADD'
    add.location = (disp.location.x - 200, disp.location.y - 200)
    nt.links.new(old, add.inputs[0])
    nt.links.new(sep.outputs['Red'], add.inputs[1])
    nt.links.new(add.outputs[0], disp.inputs['Height'])
    return mat


def adaptive(obj, pixel_size=1.0, viewport_levels=0):
    m = obj.modifiers.new("Subdivision", 'SUBSURF')
    m.levels = viewport_levels
    m.render_levels = 1
    m.use_adaptive_subdivision = True
    m.adaptive_pixel_size = pixel_size
    return m


# ---------------------------------------------------------------------------
# Rocas y vegetación
# ---------------------------------------------------------------------------

def make_rock(name, col, mat, seed, scale):
    rng = np.random.default_rng(seed)
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=5, radius=1.0)
    obj = bpy.context.active_object
    obj.name = name
    v = np.array([vv.co[:] for vv in obj.data.vertices])
    n = v / np.linalg.norm(v, axis=1, keepdims=True)
    # forma angulosa: intersección de planos aleatorios + ruido
    planes = rng.normal(size=(9, 3))
    planes /= np.linalg.norm(planes, axis=1, keepdims=True)
    r = np.ones(len(v))
    for p in planes:
        cut = rng.uniform(0.62, 0.85)
        dot = n @ p
        r = np.minimum(r, np.where(dot > 0, cut / np.maximum(dot, 1e-3), 1.5))
    r = np.minimum(r, 1.0) * (1 + 0.08 * fbm(v * 2.0, 4, seed=seed))
    v = n * r[:, None]
    v[:, 2] *= rng.uniform(0.6, 0.85)
    for vv, co in zip(obj.data.vertices, v):
        vv.co = co
    obj.data.shade_smooth()
    obj.scale = scale
    for c in obj.users_collection:
        c.objects.unlink(obj)
    col.objects.link(obj)
    obj.data.materials.append(mat)
    adaptive(obj, 4.0)
    return obj


def fern_mesh(name, rng, fronds=9):
    """Helecho: frondes arqueadas con pinnas alternas que se estrechan hacia la punta."""
    verts, faces = [], []

    def quad(a, b, c, d):
        i = len(verts)
        verts.extend([a, b, c, d])
        faces.append((i, i + 1, i + 2, i + 3))

    for f in range(fronds):
        yaw = 2 * math.pi * f / fronds + rng.uniform(-0.3, 0.3)
        length = rng.uniform(0.7, 1.25)
        rise = rng.uniform(0.9, 1.3)
        dirv = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        side = np.array([-dirv[1], dirv[0], 0.0])
        nseg = 16
        pts = []
        for i in range(nseg + 1):
            t = i / nseg
            # arco: sube y cae hacia la punta
            pts.append(dirv * length * t + np.array([0, 0, length * (rise * t - 1.1 * t * t) * 0.8]))
        for i in range(1, nseg):
            t = i / nseg
            p = pts[i]
            tan = pts[i + 1] - pts[i - 1]
            tan /= np.linalg.norm(tan)
            pl = 0.22 * length * (1 - t) ** 0.8 * (0.4 + 0.6 * math.sin(math.pi * min(t * 1.6, 1.0)))
            w = 0.035 * (1 - t * 0.6)
            for sgn in (1, -1):
                d = side * sgn * 0.9 + tan * 0.35
                d = d / np.linalg.norm(d)
                droop = np.array([0, 0, -0.35 * pl])
                tip = p + d * pl + droop
                mid = p + d * pl * 0.5 + droop * 0.35
                quad(p - tan * w * 0.5, p + tan * w * 0.5, mid + tan * w * 0.6, mid - tan * w * 0.6)
                quad(mid - tan * w * 0.6, mid + tan * w * 0.6, tip + tan * 0.004, tip - tan * 0.004)
        # raquis
        for i in range(nseg):
            a, b = pts[i], pts[i + 1]
            quad(a - side * 0.006, a + side * 0.006, b + side * 0.004, b - side * 0.004)
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([tuple(v) for v in verts], [], faces)
    mesh.update()
    return mesh


def conifer_mesh(name, rng, height):
    """Conífera tipo Metasequoia: tronco cónico y racimos de follaje en espiral con
    tamaños irregulares, para que la silueta no sea un cono perfecto."""
    import bmesh
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=8, radius1=0.018 * height, radius2=0.003 * height,
                          depth=height, matrix=Matrix.Translation((0, 0, height / 2)))
    trunk_faces = len(bm.faces)
    n = int(rng.integers(26, 40))
    for k in range(n):
        t = 0.18 + 0.8 * (k / n) ** 0.9
        reach = (0.24 * (1 - t) ** 0.85 + 0.03) * height * rng.uniform(0.6, 1.15)
        ang = k * 2.4 + rng.uniform(-0.4, 0.4)
        c = Vector((math.cos(ang) * reach * 0.55, math.sin(ang) * reach * 0.55, t * height))
        r = reach * rng.uniform(0.45, 0.7)
        ret = bmesh.ops.create_icosphere(bm, subdivisions=2, radius=1.0)
        m = (Matrix.Translation(c) @ Matrix.Rotation(ang, 4, 'Z')
             @ Matrix.Diagonal((r * 1.3, r, r * rng.uniform(0.35, 0.55), 1)))
        bmesh.ops.transform(bm, matrix=m, verts=ret['verts'])
        for v in ret['verts']:
            v.co += Vector(rng.normal(0, 0.18 * r, 3))
    mesh = bpy.data.meshes.new(name)
    bm.to_mesh(mesh)
    bm.free()
    for i, p in enumerate(mesh.polygons):
        p.material_index = 0 if i < trunk_faces else 1
        p.use_smooth = True
    return mesh


def grass_mesh(name, rng, blades=34):
    """Mata de juncias/hierba: hojas finas y curvadas que nacen de un mismo punto."""
    verts, faces = [], []
    for _ in range(blades):
        yaw = rng.uniform(0, 2 * math.pi)
        tilt = rng.uniform(0.1, 0.7)
        h = rng.uniform(0.25, 0.6)
        w = rng.uniform(0.006, 0.012)
        d = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        side = np.array([-d[1], d[0], 0.0])
        base = d * rng.uniform(0, 0.05)
        seg = 5
        i0 = len(verts)
        for k in range(seg + 1):
            t = k / seg
            p = base + d * (math.sin(tilt) * h * t + 0.35 * h * t * t * tilt) + np.array([0, 0, h * t * math.cos(tilt * t)])
            ww = w * (1 - t) + 0.0005
            verts += [tuple(p - side * ww), tuple(p + side * ww)]
        for k in range(seg):
            a = i0 + 2 * k
            faces.append((a, a + 1, a + 3, a + 2))
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    return mesh


def hills_mesh(col):
    """Colinas lejanas (anillo polar) que rompen el horizonte; la niebla las funde con el cielo."""
    nr, na = 40, 220
    r = np.geomspace(140.0, 4000.0, nr)
    a = np.linspace(0, 2 * math.pi, na, endpoint=False)
    R, A = np.meshgrid(r, a, indexing='ij')
    X, Y = 12 + R * np.cos(A), 9 + R * np.sin(A)
    ridge = fbm(np.stack([np.cos(A) * 3.0, np.sin(A) * 3.0, R * 0.0008], -1), 5, seed=33)
    Z = (0.5 + ridge) * smoothstep(150.0, 1500.0, R) * (60.0 + 0.05 * R) + 3.0 * smoothstep(140, 260, R)
    verts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], 1)
    idx = np.arange(nr * na).reshape(nr, na)
    j = np.roll(idx, -1, axis=1)
    faces = np.stack([idx[:-1].ravel(), j[:-1].ravel(), j[1:].ravel(), idx[1:].ravel()], 1)
    return mesh_from_arrays("Colinas", verts, faces, col)


def scatter(col, name, meshes, n, rng, region, accept, scale=(0.7, 1.4), sink=0.02):
    placed, tries = 0, 0
    while placed < n and tries < n * 30:
        tries += 1
        x, y = rng.uniform(*region[0]), rng.uniform(*region[1])
        if not accept(x, y):
            continue
        o = bpy.data.objects.new(f"{name}.{placed:04d}", meshes[placed % len(meshes)])
        s = rng.uniform(*scale)
        o.scale = (s, s, s * rng.uniform(0.8, 1.25))
        o.rotation_euler = (rng.uniform(-0.12, 0.12), rng.uniform(-0.12, 0.12), rng.uniform(0, 6.28))
        o.location = (x, y, -sink)
        col.objects.link(o)
        placed += 1


def instancer(name, terrain, protos, density_attr, density, scale, seed, col):
    """Geometry Nodes: reparte instancias de `protos` sobre el terreno según un atributo
    de densidad (0-1) por vértice. Son instancias: decenas de miles cuestan muy poca memoria."""
    ng = bpy.data.node_groups.new(name, 'GeometryNodeTree')
    ng.interface.new_socket("Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    ng.interface.new_socket("Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    N = ng.nodes
    L = ng.links.new
    gin, gout = N.new('NodeGroupInput'), N.new('NodeGroupOutput')
    attr = N.new('GeometryNodeInputNamedAttribute')
    attr.data_type = 'FLOAT'
    attr.inputs['Name'].default_value = density_attr
    mul = N.new('ShaderNodeMath')
    mul.operation = 'MULTIPLY'
    mul.inputs[1].default_value = density
    L(attr.outputs['Attribute'], mul.inputs[0])
    dist = N.new('GeometryNodeDistributePointsOnFaces')
    dist.distribute_method = 'RANDOM'
    dist.inputs['Seed'].default_value = seed
    L(gin.outputs[0], dist.inputs['Mesh'])
    L(mul.outputs[0], dist.inputs['Density'])
    coll = N.new('GeometryNodeCollectionInfo')
    coll.inputs['Collection'].default_value = protos
    coll.inputs['Separate Children'].default_value = True
    coll.inputs['Reset Children'].default_value = True
    inst = N.new('GeometryNodeInstanceOnPoints')
    inst.inputs['Pick Instance'].default_value = True
    L(dist.outputs['Points'], inst.inputs['Points'])
    L(coll.outputs[0], inst.inputs['Instance'])
    ridx = N.new('FunctionNodeRandomValue')
    ridx.data_type = 'INT'
    ridx.inputs['Min'].default_value = 0
    ridx.inputs['Max'].default_value = len(protos.objects) - 1
    ridx.inputs['Seed'].default_value = seed + 1
    L(ridx.outputs['Value'], inst.inputs['Instance Index'])
    rrot = N.new('FunctionNodeRandomValue')
    rrot.data_type = 'FLOAT_VECTOR'
    rrot.inputs['Min'].default_value = (-0.12, -0.12, 0.0)
    rrot.inputs['Max'].default_value = (0.12, 0.12, 6.283)
    rrot.inputs['Seed'].default_value = seed + 2
    L(rrot.outputs['Value'], inst.inputs['Rotation'])
    rsc = N.new('FunctionNodeRandomValue')
    rsc.data_type = 'FLOAT'
    rsc.inputs['Min'].default_value = scale[0]
    rsc.inputs['Max'].default_value = scale[1]
    rsc.inputs['Seed'].default_value = seed + 3
    L(rsc.outputs['Value'], inst.inputs['Scale'])
    L(inst.outputs['Instances'], gout.inputs[0])
    obj = bpy.data.objects.new(name, terrain.data)
    col.objects.link(obj)
    mod = obj.modifiers.new(name, 'NODES')
    mod.node_group = ng
    return obj


def vegetation_density(terrain):
    """Atributos de densidad por vértice del terreno: manchas de ruido, fuera del rastro
    pisoteado, del animal y de la cámara, y desvaneciéndose a lo lejos."""
    me = terrain.data
    co = np.zeros(len(me.vertices) * 3)
    me.vertices.foreach_get("co", co)
    co = co.reshape(-1, 3)
    x, y = co[:, 0], co[:, 1]
    path = np.abs(y - 0.012 * x * x) < 2.0 + 0.4 * np.maximum(0, -x) * 0.05
    path &= x < 8
    under = (x > -8.5) & (x < 8) & (np.abs(y) < 3.2)
    cam = np.hypot(x - CAM_XY[0], y - CAM_XY[1]) < 2.0
    free = (~path & ~under & ~cam).astype(float)
    far = 1 - smoothstep(60.0, 140.0, np.hypot(x - CAM_XY[0], y - CAM_XY[1]))
    pts = np.stack([x * 0.09, y * 0.09, np.zeros_like(x)], 1)
    grass = smoothstep(-0.25, 0.15, fbm(pts, 3, seed=23)) * free * far
    ferns = smoothstep(0.0, 0.3, fbm(pts * 1.3, 3, seed=17)) * free * far
    # el borde del rastro pisoteado tiene hierba más rala
    for name, val in (("dens_juncia", grass), ("dens_helecho", ferns)):
        a = me.attributes.new(name, 'FLOAT', 'POINT')
        a.data.foreach_set("value", val.astype(np.float32))


CAM_XY = (12.5, 9.6)


def build_environment(col):
    rng = np.random.default_rng(21)
    img = build_heightmap()
    terrain = terrain_mesh(col)
    terrain.data.materials.append(ground_material_with_prints(img))
    adaptive(terrain, 4.0)
    hills = hills_mesh(col)
    hills.data.materials.append(M.hills())

    rock_mat = M.rock()
    rocks = [((7.0, 6.6), (1.3, 1.1, 0.9)), ((-9.0, 9.5), (2.2, 1.7, 1.5)), ((-22.0, -11.0), (3.0, 2.2, 2.0)),
             ((10.5, -2.5), (0.6, 0.5, 0.45)), ((4.5, 7.8), (0.35, 0.3, 0.25)), ((-5.5, -6.5), (1.0, 0.8, 0.7)),
             ((12.8, 4.5), (0.3, 0.25, 0.22)), ((-34.0, 16.0), (4.0, 3.0, 3.0)), ((9.5, 8.6), (0.22, 0.2, 0.18))]
    for i, ((x, y), s) in enumerate(rocks):
        r = make_rock(f"Roca.{i:02d}", col, rock_mat, 100 + i, s)
        r.location = (x, y, -0.25 * s[2])
        r.rotation_euler = (rng.uniform(-0.15, 0.15), rng.uniform(-0.15, 0.15), rng.uniform(0, 6.28))

    def patch(x, y, thr, seed):
        return fbm(np.array([[x * 0.09, y * 0.09, 0.0]]), 3, seed=seed)[0] > thr

    def free(x, y, margin=2.2):
        if abs(y - 0.012 * x * x) < margin and x < 7:
            return False          # rastro pisoteado
        if -8 < x < 7.5 and -3 < y < 3:
            return False          # bajo el animal
        return math.hypot(x - CAM_XY[0], y - CAM_XY[1]) > 2.5

    vegetation_density(terrain)
    protos_f = bpy.data.collections.new("Prototipos_Helechos")
    protos_g = bpy.data.collections.new("Prototipos_Juncias")
    fern_mat = M.foliage("Helecho", (0.03, 0.05, 0.012), (0.085, 0.11, 0.028), 0.4)
    for k in range(5):
        m = fern_mesh(f"Helecho_{k}", rng, int(rng.integers(8, 13)))
        m.materials.append(fern_mat)
        protos_f.objects.link(bpy.data.objects.new(f"Helecho_{k}", m))
    grass_mat = M.foliage("Juncia", (0.05, 0.05, 0.018), (0.15, 0.13, 0.05), 0.3)
    for k in range(6):
        m = grass_mesh(f"Juncia_{k}", rng, int(rng.integers(26, 44)))
        m.materials.append(grass_mat)
        protos_g.objects.link(bpy.data.objects.new(f"Juncia_{k}", m))
    instancer("Vegetacion_Juncias", terrain, protos_g, "dens_juncia", 2.2, (0.6, 1.5), 5, col)
    instancer("Vegetacion_Helechos", terrain, protos_f, "dens_helecho", 0.35, (0.6, 1.6), 9, col)

    bark_mat = M.bark()
    leaf_mat = M.foliage("Follaje_Conifera", (0.018, 0.03, 0.012), (0.045, 0.06, 0.025), 0.2)
    trees = [conifer_mesh(f"Conifera_{k}", rng, h) for k, h in enumerate((22.0, 28.0, 34.0, 26.0))]
    for m in trees:
        m.materials.append(bark_mat)
        m.materials.append(leaf_mat)
    k = 0
    for _ in range(260):
        ang = rng.uniform(math.radians(160), math.radians(300))
        dist = rng.uniform(220, 650)
        # bosquetes: densidad modulada por ruido angular
        if fbm(np.array([[math.cos(ang) * 4, math.sin(ang) * 4, dist * 0.004]]), 2, seed=41)[0] < -0.05:
            continue
        x, y = CAM_XY[0] + dist * math.cos(ang), CAM_XY[1] + dist * math.sin(ang)
        o = bpy.data.objects.new(f"Conifera.{k:03d}", trees[k % len(trees)])
        s = rng.uniform(0.6, 1.2)
        o.scale = (s, s, s * rng.uniform(0.85, 1.25))
        o.rotation_euler = (0, 0, rng.uniform(0, 6.28))
        o.location = (x, y, -0.5)
        o.visible_shadow = False  # el bosque lejano no debe tapar el sol rasante
        col.objects.link(o)
        k += 1
    return terrain


# ---------------------------------------------------------------------------
# Cielo, luz y cámara
# ---------------------------------------------------------------------------

def build_world(scene):
    world = bpy.data.worlds.new("Cielo_Cretacico")
    scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()
    sky = nt.nodes.new('ShaderNodeTexSky')
    sky.sky_type = 'MULTIPLE_SCATTERING'
    sky.sun_disc = False
    sky.sun_elevation = math.radians(SUN_ELEVATION)
    # Nishita mide la rotación desde +Y en sentido horario visto desde arriba
    sky.sun_rotation = math.radians(90.0 - SUN_AZIMUTH)
    sky.altitude = 200.0
    sky.air_density = 1.0
    sky.aerosol_density = 1.2
    sky.ozone_density = 1.0
    bg = nt.nodes.new('ShaderNodeBackground')
    bg.inputs['Strength'].default_value = 0.12
    nt.links.new(sky.outputs['Color'], bg.inputs['Color'])
    out = nt.nodes.new('ShaderNodeOutputWorld')
    nt.links.new(bg.outputs[0], out.inputs['Surface'])
    return world


def build_lights_camera(col, scene, target=(1.3, 0.1, 2.75)):
    sun = bpy.data.lights.new("Sol", 'SUN')
    sun.energy = 9.0
    sun.color = (1.0, 0.82, 0.62)
    sun.angle = math.radians(1.2)
    sun_obj = bpy.data.objects.new("Sol", sun)
    col.objects.link(sun_obj)
    sun_obj.rotation_euler = (-sun_vector()).to_track_quat('-Z', 'Y').to_euler()

    tgt = bpy.data.objects.new("Camara_Objetivo", None)
    col.objects.link(tgt)
    tgt.location = target
    cam_data = bpy.data.cameras.new("Camara")
    cam_data.lens = 50
    cam_data.sensor_width = 36
    cam_data.clip_end = 3000
    cam_data.dof.use_dof = True
    cam_data.dof.aperture_fstop = 4.0
    cam_data.dof.aperture_blades = 7
    cam = bpy.data.objects.new("Camara", cam_data)
    col.objects.link(cam)
    cam.location = (14.5, 11.0, 1.65)
    c = cam.constraints.new('TRACK_TO')
    c.target = tgt
    c.track_axis = 'TRACK_NEGATIVE_Z'
    c.up_axis = 'UP_Y'
    scene.camera = cam
    return cam


def setup_render(scene, samples=256):
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = samples
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.015
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = 'OPENIMAGEDENOISE'
    scene.cycles.max_bounces = 8
    scene.cycles.diffuse_bounces = 3
    scene.cycles.glossy_bounces = 3
    scene.cycles.transmission_bounces = 8
    scene.cycles.transparent_max_bounces = 8
    scene.cycles.caustics_reflective = False
    scene.cycles.caustics_refractive = False
    scene.cycles.blur_glossy = 1.0
    scene.cycles.dicing_rate = 1.0
    scene.cycles.offscreen_dicing_scale = 8.0
    scene.cycles.max_subdivisions = 4
    scene.render.resolution_x = 1920
    scene.render.resolution_y = 1080
    scene.render.resolution_percentage = 100
    scene.view_settings.view_transform = 'AgX'
    scene.view_settings.look = 'AgX - Medium High Contrast'
    scene.view_settings.exposure = -0.5
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_depth = '16'
