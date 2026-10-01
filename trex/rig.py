"""
Esqueleto de animación construido sobre los mismos puntos de la anatomía.

La pose de reposo es la pose esculpida. Los pesos se calculan con numpy (influencia
gaussiana por distancia a cada hueso y separación cráneo/mandíbula usando el propio
campo de distancia), lo que es robusto con mallas densas en las que el "bone heat"
de Blender falla o tarda demasiado.
"""

import bpy
import numpy as np
from mathutils import Vector

from . import sdf
from .anatomia import ARMS, HINGE, LEGS, TOES, TRUNK


def bone_specs(dino):
    hf = dino.hf
    t = sdf.catmull_rom(TRUNK, 6)

    def trunk_pt(x):
        return np.array([x, np.interp(x, t[:, 0], t[:, 1]),
                         (np.interp(x, t[:, 0], t[:, 2]) + np.interp(x, t[:, 0], t[:, 3])) / 2 + 0.25])

    B = []  # (nombre, cabeza, cola, padre, radio de influencia, deforma)
    B.append(('root', (0, 0, 0), (0, -1.5, 0), None, 0, False))
    B.append(('hips', trunk_pt(0.0), trunk_pt(0.0) - np.array([0, 0, 0.6]), 'root', 0.9, True))
    spine = [trunk_pt(x) for x in (0.0, 1.1, 2.2)]
    neck0 = trunk_pt(2.95)
    head0 = hf.p(-0.15, 0, 0.12)
    chain = spine + [neck0, (neck0 + head0) / 2 + np.array([0, 0, 0.12]), head0]
    names = ['spine_01', 'spine_02', 'neck_01', 'neck_02', 'neck_03']
    radii = [0.95, 0.95, 0.8, 0.6, 0.5]
    parent = 'hips'
    for name, a, b, r in zip(names, chain[:-1], chain[1:], radii):
        B.append((name, a, b, parent, r, True))
        parent = name
    B.append(('head', head0, hf.p(1.35, 0, 0.12), 'neck_03', 0.8, True))
    B.append(('jaw', hf.p(*HINGE), hf.jp(1.3, 0, -0.35), 'head', 0.7, True))
    xs = [0.0, -1.0, -2.0, -3.0, -4.0, -4.9, -5.7, -6.4, -7.05]
    parent = 'hips'
    for i, (a, b) in enumerate(zip(xs[:-1], xs[1:]), 1):
        r = max(np.interp(a, t[:, 0], t[:, 4]) * 1.5, 0.12)
        B.append((f'tail_{i:02d}', trunk_pt(a) - np.array([0, 0, 0.2]), trunk_pt(b) - np.array([0, 0, 0.2]),
                  parent, r, True))
        parent = f'tail_{i:02d}'
    for s, leg in LEGS.items():
        hip, knee, ankle, ball = (np.array(leg[k]) for k in ('hip', 'knee', 'ankle', 'ball'))
        toe_tip = ball + np.array([0.55, 0, -0.06])
        B += [(f'thigh.{s}', hip, knee, 'hips', 0.65, True),
              (f'shin.{s}', knee, ankle, f'thigh.{s}', 0.38, True),
              (f'foot.{s}', ankle, ball, f'shin.{s}', 0.22, True),
              (f'toes.{s}', ball, toe_tip, f'foot.{s}', 0.22, True)]
    for s, arm in ARMS.items():
        sh, el, wr = (np.array(arm[k]) for k in ('shoulder', 'elbow', 'wrist'))
        B += [(f'upperarm.{s}', sh, el, 'spine_02', 0.17, True),
              (f'forearm.{s}', el, wr, f'upperarm.{s}', 0.11, True),
              (f'hand.{s}', wr, wr + np.array([0.18, 0, -0.1]), f'forearm.{s}', 0.09, True)]
    return B


def build_rig(dino, col):
    arm_data = bpy.data.armatures.new("Rig_TRex")
    arm_data.display_type = 'OCTAHEDRAL'
    rig = bpy.data.objects.new("Rig_TRex", arm_data)
    col.objects.link(rig)
    rig.show_in_front = True
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode='EDIT')
    specs = bone_specs(dino)
    for name, a, b, parent, r, deform in specs:
        eb = arm_data.edit_bones.new(name)
        eb.head, eb.tail = Vector(a), Vector(b)
        eb.use_deform = deform
        if parent:
            eb.parent = arm_data.edit_bones[parent]
    bpy.ops.object.mode_set(mode='OBJECT')
    return rig, specs


def _seg_dist(p, a, b):
    ab = b - a
    t = np.clip(((p - a) @ ab) / (ab @ ab), 0, 1)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1)


def skin_weights(dino, verts, specs):
    """Pesos (n_vert, n_huesos) normalizados, con separación cráneo/mandíbula."""
    deform = [s for s in specs if s[5]]
    W = np.zeros((len(verts), len(deform)), np.float32)
    for j, (name, a, b, parent, r, _) in enumerate(deform):
        d = _seg_dist(verts, np.asarray(a, float), np.asarray(b, float))
        W[:, j] = np.exp(-2.5 * (d / r) ** 2)
    names = [s[0] for s in deform]
    # mandíbula frente a cráneo: decide el campo de distancia de cada pieza
    skull, jaw = dino.skull(), dino.jaw()
    x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
    near_head = np.linalg.norm(verts - dino.hf.O, axis=1) < 2.0
    jw = np.zeros(len(verts), np.float32)
    idx = np.nonzero(near_head)[0]
    ds = skull.eval(x[idx], y[idx], z[idx])
    dj = jaw.eval(x[idx], y[idx], z[idx])
    jw[idx] = sdf.smoothstep(0.04, -0.04, dj - ds)
    ih, ij = names.index('head'), names.index('jaw')
    head_total = W[:, ih] + W[:, ij]
    W[:, ih] = head_total * (1 - jw)
    W[:, ij] = head_total * jw
    # conserva las 4 influencias mayores
    if W.shape[1] > 4:
        thr = -np.partition(-W, 3, axis=1)[:, 3:4]
        W[W < thr] = 0
    W /= np.maximum(W.sum(1, keepdims=True), 1e-6)
    return W, names


def assign_weights(obj, W, names, levels=64):
    """Asigna pesos por lotes cuantizados (mucho más rápido que vértice a vértice)."""
    for j, name in enumerate(names):
        col = W[:, j]
        if not np.any(col > 0.5 / levels):
            continue
        vg = obj.vertex_groups.new(name=name)
        q = np.round(col * levels).astype(int)
        for lvl in np.unique(q[q > 0]):
            vg.add(np.nonzero(q == lvl)[0].tolist(), lvl / levels, 'REPLACE')


def attach_mesh(obj, rig):
    mod = obj.modifiers.new("Armature", 'ARMATURE')
    mod.object = rig
    obj.parent = rig
    # el esqueleto debe evaluarse antes que la subdivisión
    while obj.modifiers.find(mod.name) > 0:
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.modifier_move_up(modifier=mod.name)


def bind_rigid(obj, rig, bone):
    """Une una pieza rígida a un hueso mediante un grupo de vértices de peso 1."""
    vg = obj.vertex_groups.new(name=bone)
    vg.add(list(range(len(obj.data.vertices))), 1.0, 'REPLACE')
    attach_mesh(obj, rig)
