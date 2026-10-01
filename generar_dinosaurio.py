"""
Generador procedural de un Tyrannosaurus rex hiperrealista para Blender 5.x (Cycles).

Uso:
    blender -b --python generar_dinosaurio.py
    blender -b --python generar_dinosaurio.py -- --render salida.png [--pct 50] [--samples 128]
                                                 [--camara principal|retrato] [--voxel 0.02]

Flujo:
  1. Anatomía como campo de distancia con signo (numpy): tronco, cuello y cola por
     barrido de secciones, cráneo y mandíbula abierta, musculatura de muslos y gemelos,
     dedos con almohadillas, ojos hundidos bajo la ceja, boca tallada con lengua.
  2. Malla por *surface nets*, relajada y reproyectada a la isosuperficie.
  3. Atributos por vértice (contrasombreado, pliegues, labios, cavidades) que alimentan
     una piel procedural con desplazamiento real mediante subdivisión adaptativa.
  4. Dientes, garras, ojos con córnea, hilos de saliva y un esqueleto de animación.
  5. Llanura del Cretácico con huellas del propio animal, rocas, helechos, bosque de
     coníferas con perspectiva aérea, cielo físico y cámara de fotografía de campo.
"""

import sys
import time
from pathlib import Path

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector

PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from trex import anatomia, escenario, geometria, materiales, rig, sdf  # noqa: E402

BLEND_PATH = PROJECT_DIR / "dinosaurio.blend"


def log(msg, t0=[time.time()]):
    print(f"[{time.time() - t0[0]:7.1f}s] {msg}", flush=True)


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []

    def opt(name, default, cast=str):
        return cast(argv[argv.index(name) + 1]) if name in argv else default

    return dict(render=opt("--render", None), pct=opt("--pct", 100, int), samples=opt("--samples", 256, int),
                camera=opt("--camara", "principal"), voxel=opt("--voxel", 0.016, float),
                no_env="--sin-entorno" in argv, save="--no-guardar" not in argv)


def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = 'METRIC'
    return scene


def make_collection(name):
    col = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(col)
    return col


# ---------------------------------------------------------------------------
# Cuerpo
# ---------------------------------------------------------------------------

def build_body(dino, col, voxel):
    F = dino.build(h=voxel)
    log(f"campo {F.shape}")
    verts, faces = sdf.surface_nets(F)
    verts = sdf.relax(verts, faces, F, iterations=4)
    log(f"malla: {len(verts)} vértices, {len(faces)} caras")
    attrs, _ = dino.regions(verts)
    boca = attrs.pop('boca')
    body = geometria.mesh_from_arrays("TRex_Cuerpo", verts, faces, col, attrs=attrs)
    body.data.materials.append(materiales.skin())
    body.data.materials.append(materiales.mouth())
    face_mouth = boca[faces].mean(1) > 0.5
    body.data.polygons.foreach_set("material_index", face_mouth.astype(np.int32))
    # sin subdivisión: el desplazamiento se aplica a los vértices de la malla base (densa)
    # y la escama fina va en bump; la subdivisión adaptativa no cabe en 8 GB de RAM
    return body, verts


def build_teeth(dino, col):
    rng = np.random.default_rng(7)
    upper, lower = [], []
    for base, d, back, length, rad, is_lower in dino.teeth_layout(rng):
        piece = geometria.horn(base - d * 0.035, d, back, length + 0.035, length * rad * 1.2,
                               flatten=0.78, bend=0.2, rings=11, seg=12, keel=0.12, profile_pow=1.8)
        (lower if is_lower else upper).append(piece)
    mat = materiales.teeth()
    objs = []
    for name, pieces in (("Dientes_Superiores", upper), ("Dientes_Inferiores", lower)):
        o = geometria.horns_object(name, pieces, col)
        o.data.materials.append(mat)
        objs.append(o)
    return objs


def build_claws(dino, col):
    pieces = []
    for tip, d, r in dino.toe_tips:
        big = r > 0.04
        length = 0.17 if big else 0.08
        pieces.append(geometria.horn(tip - d * 0.03, d,
                                     np.array([0, 0, -1.0]), length, r * 0.95, flatten=0.62, bend=0.32,
                                     rings=10, seg=10, keel=0.25))
    for tip, d, r in dino.finger_tips:
        pieces.append(geometria.horn(tip - d * 0.01, d, np.array([0, 0, -1.0]), 0.085, r * 0.9,
                                     flatten=0.6, bend=0.45, rings=10, seg=10, keel=0.25))
    o = geometria.horns_object("Garras", pieces, col)
    o.data.materials.append(materiales.claws())
    return o


def build_eyes(dino, col):
    eye_mat, cornea_mat = materiales.eye(), materiales.cornea()
    out = []
    for c, n, r, s in dino.eyes:
        z = Vector(n).normalized()
        up = Vector(dino.hf.U)
        x = up.cross(z).normalized()
        y = z.cross(x)
        rot = Matrix((x, y, z)).transposed().to_4x4()
        for name, mat, rad, off in (("Ojo", eye_mat, r, 0.0), ("Cornea", cornea_mat, r * 0.86, r * 0.24)):
            mesh = bpy.data.meshes.new(f"{name}.{'L' if s > 0 else 'R'}")
            bm = bmesh.new()
            bmesh.ops.create_uvsphere(bm, u_segments=48, v_segments=24, radius=1.0)
            bm.to_mesh(mesh)
            bm.free()
            mesh.shade_smooth()
            mesh.materials.append(mat)
            o = bpy.data.objects.new(mesh.name, mesh)
            col.objects.link(o)
            o.matrix_world = Matrix.Translation(Vector(c) + z * off) @ rot @ Matrix.Diagonal((rad, rad, rad, 1))
            out.append(o)
    return out


def build_saliva(dino, col):
    """Hilos de saliva entre las mandíbulas, combados por la gravedad."""
    hf = dino.hf
    mat = materiales.saliva()
    rng = np.random.default_rng(3)
    curve = bpy.data.curves.new("Saliva", 'CURVE')
    curve.dimensions = '3D'
    curve.bevel_depth = 1.0
    curve.bevel_resolution = 3
    curve.use_fill_caps = True
    for u, v in ((0.42, 0.20), (0.55, -0.17), (0.78, 0.12), (0.36, -0.22), (0.95, -0.05), (0.62, 0.02)):
        a = Vector(hf.p(u, v, anatomia.skull_lip(u) + 0.04))
        b = Vector(hf.jp(u - rng.uniform(0.0, 0.15), v * 0.9, anatomia.jaw_top(u) - 0.04))
        mid = (a + b) / 2 + Vector(hf.F) * rng.uniform(0.0, 0.05) - Vector((0, 0, rng.uniform(0.04, 0.1)))
        sp = curve.splines.new('BEZIER')
        sp.bezier_points.add(2)
        for bp, p, rad in zip(sp.bezier_points, (a, mid, b), (0.007, 0.0028, 0.007)):
            bp.co = p
            bp.handle_left_type = bp.handle_right_type = 'AUTO'
            bp.radius = rad * rng.uniform(0.8, 1.2)
    curve.materials.append(mat)
    o = bpy.data.objects.new("Saliva", curve)
    col.objects.link(o)
    # a malla para poder deformarla con el esqueleto
    mesh = bpy.data.meshes.new_from_object(o.evaluated_get(bpy.context.evaluated_depsgraph_get()))
    bpy.data.objects.remove(o)
    o = bpy.data.objects.new("Saliva", mesh)
    col.objects.link(o)
    mesh.shade_smooth()
    return o


# ---------------------------------------------------------------------------
# Ensamblado
# ---------------------------------------------------------------------------

CAMERAS = {
    # dicing: tamaño en píxeles de los micropolígonos (más alto = menos memoria)
    'principal': dict(location=(12.5, 9.6, 1.35), target=(1.0, 0.0, 2.85), lens=42, fstop=4.0, dicing=1.0),
    'retrato': dict(location=(8.4, 4.6, 3.3), target=(4.75, 0.5, 3.6), lens=70, fstop=3.5, dicing=3.0),
    'lateral': dict(location=(-1.0, 19.0, 2.2), target=(-0.8, 0.0, 2.4), lens=40, fstop=8.0, dicing=1.0),
}


def setup_camera(scene, which, dino):
    cfg = CAMERAS[which]
    cam = scene.camera
    cam.location = cfg['location']
    cam.data.lens = cfg['lens']
    cam.data.dof.aperture_fstop = cfg['fstop']
    scene.cycles.dicing_rate = cfg['dicing']
    bpy.data.objects["Camara_Objetivo"].location = cfg['target']
    # enfoca al ojo izquierdo (el más cercano a la cámara)
    eye = next(e for e in dino.eyes if e[3] > 0)
    cam.data.dof.focus_distance = (Vector(eye[0]) - Vector(cfg['location'])).length


def main():
    args = parse_args()
    scene = reset_scene()
    col_dino = make_collection("TRex")
    col_rig = make_collection("Rig")
    col_env = make_collection("Escenario")
    col_lc = make_collection("Luz_Camara")

    dino = anatomia.Dino()
    body, verts = build_body(dino, col_dino, args['voxel'])
    teeth_up, teeth_low = build_teeth(dino, col_dino)
    claws = build_claws(dino, col_dino)
    eyes = build_eyes(dino, col_dino)
    saliva = build_saliva(dino, col_dino)
    log("accesorios listos")

    arm, specs = rig.build_rig(dino, col_rig)
    W, names = rig.skin_weights(dino, verts, specs)
    rig.assign_weights(body, W, names)
    rig.attach_mesh(body, arm)
    cv = np.array([v.co[:] for v in claws.data.vertices])
    Wc, _ = rig.skin_weights(dino, cv, specs)
    rig.assign_weights(claws, Wc, names)
    rig.attach_mesh(claws, arm)
    for o in [teeth_up, saliva, *eyes]:
        rig.bind_rigid(o, arm, 'head')
    rig.bind_rigid(teeth_low, arm, 'jaw')
    arm.hide_render = True
    log("rig listo")

    if not args['no_env']:
        escenario.build_environment(col_env)
        log("entorno listo")
    escenario.build_world(scene)
    escenario.build_lights_camera(col_lc, scene)
    escenario.setup_render(scene, args['samples'])
    setup_camera(scene, args['camera'], dino)

    if args['save']:
        bpy.context.preferences.filepaths.save_version = 0
        bpy.ops.wm.save_as_mainfile(filepath=str(BLEND_PATH), compress=True)
        log(f"guardado: {BLEND_PATH}")

    if args['render']:
        scene.render.resolution_percentage = args['pct']
        scene.render.filepath = str(Path(args['render']).resolve())
        log("renderizando…")
        bpy.ops.render.render(write_still=True)
        log(f"render: {scene.render.filepath}")


if __name__ == "__main__":
    main()
