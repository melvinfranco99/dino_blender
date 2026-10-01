"""
Creación de mallas de Blender a partir de arrays de numpy y generadores de piezas
queratinosas (dientes y garras) con sección comprimida y curvatura.
"""

import math

import bpy
import numpy as np

from .sdf import normalize


def mesh_from_arrays(name, verts, faces, col, smooth=True, attrs=None):
    mesh = bpy.data.meshes.new(name)
    faces = np.asarray(faces)
    nv, nf = len(verts), len(faces)
    corners = faces.shape[1]
    mesh.vertices.add(nv)
    mesh.vertices.foreach_set("co", np.asarray(verts, np.float32).ravel())
    mesh.loops.add(nf * corners)
    mesh.loops.foreach_set("vertex_index", faces.astype(np.int32).ravel())
    mesh.polygons.add(nf)
    mesh.polygons.foreach_set("loop_start", np.arange(0, nf * corners, corners, dtype=np.int32))
    mesh.update(calc_edges=True)
    mesh.validate(clean_customdata=False)
    if smooth:
        mesh.shade_smooth()
    for key, values in (attrs or {}).items():
        a = mesh.attributes.new(key, 'FLOAT', 'POINT')
        a.data.foreach_set("value", np.asarray(values, np.float32))
    obj = bpy.data.objects.new(name, mesh)
    col.objects.link(obj)
    return obj


def horn(base, direction, back, length, radius, flatten=0.7, bend=0.25, rings=12, seg=12,
         keel=0.0, profile_pow=1.35):
    """
    Pieza cónica curvada (diente o garra).
    base: centro de la raíz; direction: hacia la punta; back: hacia donde se curva.
    flatten: compresión lateral de la sección; keel: arista (carena) delante/detrás.
    Devuelve (verts, faces, t) con t = 0 en la raíz y 1 en la punta.
    """
    d = normalize(direction)
    b = np.asarray(back, float)
    b = normalize(b - d * (b @ d))
    lat = np.cross(d, b)
    verts, ts = [], []
    for i in range(rings):
        t = i / (rings - 1)
        r = radius * max(1.0 - t ** profile_pow, 0.0) ** 0.75
        c = np.asarray(base) + d * (length * t) + b * (bend * length * t * t)
        # la tangente se inclina con la curvatura para que la sección siga el eje
        tan = normalize(d + b * (2 * bend * t))
        bb = normalize(b - tan * (b @ tan))
        ll = np.cross(tan, bb)
        for j in range(seg):
            a = 2 * math.pi * j / seg
            ca, sa = math.cos(a), math.sin(a)
            # carenas anterior y posterior (dientes serrados de terópodo)
            k = 1.0 + keel * (abs(ca) ** 8)
            verts.append(c + bb * (ca * r * k) + ll * (sa * r * flatten))
            ts.append(t)
    tip = np.asarray(base) + d * length * 1.02 + b * (bend * length * 1.04)
    verts.append(tip)
    ts.append(1.0)
    verts.append(np.asarray(base) - d * radius * 0.2)
    ts.append(0.0)
    faces = []
    for i in range(rings - 1):
        for j in range(seg):
            a0, a1 = i * seg + j, i * seg + (j + 1) % seg
            faces.append((a0, a1, a1 + seg, a0 + seg))
    it, ib = len(verts) - 2, len(verts) - 1
    last = (rings - 1) * seg
    tris = [(last + j, last + (j + 1) % seg, it) for j in range(seg)]
    tris += [((j + 1) % seg, j, ib) for j in range(seg)]
    return np.array(verts), faces, tris, np.array(ts)


def horns_object(name, pieces, col):
    """Une varias piezas `horn` en un objeto; atributo `largo` (0 raíz → 1 punta)."""
    allv, allq, allt, allp, idx = [], [], [], [], []
    off = 0
    for n, (v, quads, tris, t) in enumerate(pieces):
        allv.append(v)
        allq += [tuple(i + off for i in q) for q in quads]
        allt += [tuple(i + off for i in q) for q in tris]
        allp.append(t)
        idx.append(np.full(len(v), (n * 0.6180339) % 1.0))
        off += len(v)
    verts = np.concatenate(allv)
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(verts.tolist(), [], allq + allt)
    mesh.update()
    mesh.shade_smooth()
    for key, vals in (('largo', np.concatenate(allp)), ('pieza', np.concatenate(idx))):
        a = mesh.attributes.new(key, 'FLOAT', 'POINT')
        a.data.foreach_set("value", vals.astype(np.float32))
    obj = bpy.data.objects.new(name, mesh)
    col.objects.link(obj)
    return obj
