"""
Modelado implícito: campos de distancia con signo (SDF) evaluados con numpy sobre una
rejilla de vóxeles y convertidos a malla con *surface nets* (contorneado dual).

Cada primitiva se evalúa solo dentro de su caja envolvente, así que el coste depende de
la superficie del modelo y no del volumen total de la rejilla.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Utilidades vectoriales
# ---------------------------------------------------------------------------

def normalize(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def frame_from(forward, up=(0.0, 0.0, 1.0)):
    """Base ortonormal (T, N, B): T hacia delante, N a la izquierda, B hacia arriba."""
    t = normalize(forward)
    up = np.asarray(up, float)
    if abs(np.dot(t, up)) > 0.98:
        up = np.array([1.0, 0.0, 0.0])
    n = normalize(np.cross(up, t))
    b = np.cross(t, n)
    return t, n, b


def catmull_rom(table, per_segment=6):
    """Densifica una tabla de filas (todas las columnas) con splines de Catmull-Rom."""
    p = np.asarray(table, float)
    p = np.vstack([2 * p[0] - p[1], p, 2 * p[-1] - p[-2]])
    out = []
    for i in range(1, len(p) - 2):
        p0, p1, p2, p3 = p[i - 1], p[i], p[i + 1], p[i + 2]
        for t in np.linspace(0, 1, per_segment, endpoint=False):
            out.append(0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(p[-2])
    return np.array(out)


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# ---------------------------------------------------------------------------
# Ruido de valor 3D (hash entero, interpolación quíntica) para irregularidades orgánicas
# ---------------------------------------------------------------------------

def _hash3(i, j, k, seed):
    n = (i.astype(np.uint32) * np.uint32(374761393) + j.astype(np.uint32) * np.uint32(668265263)
         + k.astype(np.uint32) * np.uint32(1274126177) + np.uint32(seed * 2654435761 % 2**32))
    n = (n ^ (n >> np.uint32(13))) * np.uint32(1274126177)
    n = n ^ (n >> np.uint32(16))
    return (n & np.uint32(0xFFFF)).astype(np.float32) / np.float32(32767.5) - np.float32(1.0)


def value_noise(p, seed=0):
    """p: (..., 3) -> ruido en [-1, 1]."""
    p = np.asarray(p, np.float32)
    fl = np.floor(p)
    f = p - fl
    i = fl.astype(np.int64)
    w = f * f * f * (f * (f * 6 - 15) + 10)
    out = 0.0
    for dx in (0, 1):
        wx = w[..., 0] if dx else 1 - w[..., 0]
        for dy in (0, 1):
            wy = w[..., 1] if dy else 1 - w[..., 1]
            for dz in (0, 1):
                wz = w[..., 2] if dz else 1 - w[..., 2]
                out = out + wx * wy * wz * _hash3(i[..., 0] + dx, i[..., 1] + dy, i[..., 2] + dz, seed)
    return out


def fbm(p, octaves=4, lacunarity=2.03, gain=0.5, seed=0):
    p = np.asarray(p, np.float32)
    amp, total, norm = 1.0, 0.0, 0.0
    for o in range(octaves):
        total = total + amp * value_noise(p, seed + o * 31)
        norm += amp
        p = p * lacunarity
        amp *= gain
    return total / norm


# ---------------------------------------------------------------------------
# Rejilla y operadores booleanos suaves
# ---------------------------------------------------------------------------

class Field:
    def __init__(self, bmin, bmax, h):
        self.h = float(h)
        self.bmin = np.asarray(bmin, float)
        self.shape = tuple((np.ceil((np.asarray(bmax, float) - self.bmin) / h)).astype(int) + 1)
        self.f = np.full(self.shape, 10.0, np.float32)

    def axes(self, lo, hi):
        """Índices y coordenadas de la sub-rejilla que cubre la caja [lo, hi]."""
        i0 = np.maximum(np.floor((np.asarray(lo) - self.bmin) / self.h).astype(int), 0)
        i1 = np.minimum(np.ceil((np.asarray(hi) - self.bmin) / self.h).astype(int) + 1, self.shape)
        if np.any(i1 <= i0):
            return None
        sl = tuple(slice(a, b) for a, b in zip(i0, i1))
        xs = [(self.bmin[d] + np.arange(i0[d], i1[d]) * self.h).astype(np.float32) for d in range(3)]
        return sl, xs[0][:, None, None], xs[1][None, :, None], xs[2][None, None, :]

    def apply(self, prim, op='union', k=0.0):
        lo, hi = prim.bounds()
        m = k * 1.5 + 2 * self.h
        sub = self.axes(lo - m, hi + m)
        if sub is None:
            return
        sl, x, y, z = sub
        if hasattr(prim, 'parts'):
            # primitivas compuestas: cada tramo solo dentro de su propia caja
            d = np.full((x.shape[0], y.shape[1], z.shape[2]), 10.0, np.float32)
            off = np.array([s.start for s in sl])
            end = np.array([s.stop for s in sl])
            for part in prim.parts():
                plo, phi = part.bounds()
                i0 = np.maximum(np.floor((plo - m - self.bmin) / self.h).astype(int), off)
                i1 = np.minimum(np.ceil((phi + m - self.bmin) / self.h).astype(int) + 1, end)
                if np.any(i1 <= i0):
                    continue
                px, py, pz = [(self.bmin[k] + np.arange(i0[k], i1[k]) * self.h).astype(np.float32)
                              for k in range(3)]
                loc = tuple(slice(a - o, b - o) for a, b, o in zip(i0, i1, off))
                d[loc] = np.minimum(d[loc], part.eval(px[:, None, None], py[None, :, None],
                                                      pz[None, None, :]))
        else:
            d = prim.eval(x, y, z)
        a = self.f[sl]
        if op == 'union':
            self.f[sl] = smin(a, d, k)
        elif op == 'subtract':
            self.f[sl] = -smin(-a, d, k)
        elif op == 'intersect':
            self.f[sl] = -smin(-a, -d, k)

    def sample(self, pts):
        """Interpolación trilineal del campo en puntos (N, 3)."""
        g = (np.asarray(pts) - self.bmin) / self.h
        i = np.clip(np.floor(g).astype(int), 0, np.array(self.shape) - 2)
        t = g - i
        f = self.f
        out = 0.0
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    w = ((t[:, 0] if dx else 1 - t[:, 0]) * (t[:, 1] if dy else 1 - t[:, 1])
                         * (t[:, 2] if dz else 1 - t[:, 2]))
                    out = out + w * f[i[:, 0] + dx, i[:, 1] + dy, i[:, 2] + dz]
        return out

    def gradient(self, pts):
        e = self.h * 0.75
        g = np.empty_like(pts)
        for d in range(3):
            o = np.zeros(3)
            o[d] = e
            g[:, d] = (self.sample(pts + o) - self.sample(pts - o)) / (2 * e)
        return g


def smin(a, b, k):
    if k <= 0:
        return np.minimum(a, b)
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b + (a - b) * h - k * h * (1.0 - h)


# ---------------------------------------------------------------------------
# Primitivas
# ---------------------------------------------------------------------------

class Ellipsoid:
    def __init__(self, center, radii, forward=(1, 0, 0), up=(0, 0, 1)):
        self.c = np.asarray(center, float)
        self.r = np.asarray(radii, float)
        self.basis = np.array(frame_from(forward, up))  # filas: T, N, B

    def bounds(self):
        e = np.abs(self.basis.T) @ self.r
        return self.c - e, self.c + e

    def eval(self, x, y, z):
        qx, qy, qz = x - self.c[0], y - self.c[1], z - self.c[2]
        T, N, B = self.basis
        a = (qx * T[0] + qy * T[1] + qz * T[2]) / self.r[0]
        b = (qx * N[0] + qy * N[1] + qz * N[2]) / self.r[1]
        c = (qx * B[0] + qy * B[1] + qz * B[2]) / self.r[2]
        k0 = np.sqrt(a * a + b * b + c * c)
        k1 = np.sqrt((a / self.r[0]) ** 2 + (b / self.r[1]) ** 2 + (c / self.r[2]) ** 2) + 1e-9
        return (k0 * (k0 - 1.0) / k1).astype(np.float32)


class RoundCone:
    """Cápsula de radio variable entre a y b."""

    def __init__(self, a, b, ra, rb):
        self.a, self.b = np.asarray(a, float), np.asarray(b, float)
        self.ra, self.rb = ra, rb

    def bounds(self):
        r = max(self.ra, self.rb)
        return np.minimum(self.a, self.b) - r, np.maximum(self.a, self.b) + r

    def eval(self, x, y, z):
        ba = self.b - self.a
        l2 = float(ba @ ba)
        qx, qy, qz = x - self.a[0], y - self.a[1], z - self.a[2]
        t = np.clip((qx * ba[0] + qy * ba[1] + qz * ba[2]) / l2, 0.0, 1.0)
        dx, dy, dz = qx - ba[0] * t, qy - ba[1] * t, qz - ba[2] * t
        return (np.sqrt(dx * dx + dy * dy + dz * dz) - (self.ra + (self.rb - self.ra) * t)).astype(np.float32)


class Loft:
    """
    Barrido de secciones elípticas (o superelípticas) a lo largo de una polilínea.
    sections: lista de (centro, ry, rz) y opcionalmente up por sección.
    La sección se orienta con la tangente local; ry es el semieje lateral, rz el vertical.
    """

    def __init__(self, centers, ry, rz, power=2.0, up=(0, 0, 1), taper=0.0):
        self.taper = np.broadcast_to(np.asarray(taper, float), (len(centers),)).copy()
        self.c = np.asarray(centers, float)
        self.ry = np.asarray(ry, float)
        self.rz = np.asarray(rz, float)
        self.p = power
        n = len(self.c)
        tang = np.zeros((n, 3))
        for i in range(n):
            a, b = self.c[max(i - 1, 0)], self.c[min(i + 1, n - 1)]
            tang[i] = normalize(b - a)
        ups = np.asarray(up, float)
        if ups.ndim == 1:
            ups = np.tile(ups, (n, 1))
        self.N = np.array([frame_from(tang[i], ups[i])[1] for i in range(n)])
        self.B = np.array([frame_from(tang[i], ups[i])[2] for i in range(n)])

    def bounds(self):
        r = np.maximum(self.ry, self.rz)[:, None]
        return (self.c - r).min(0), (self.c + r).max(0)

    def segments(self):
        for i in range(len(self.c) - 1):
            yield i

    def parts(self):
        for i in self.segments():
            yield _Segment(self, i)

    def eval(self, x, y, z):
        out = None
        for i in self.segments():
            d = self._segment(i, x, y, z)
            out = d if out is None else np.minimum(out, d)
        return out

    def _segment(self, i, x, y, z):
        a, b = self.c[i], self.c[i + 1]
        ba = b - a
        L = np.linalg.norm(ba)
        T = ba / L
        qx, qy, qz = x - a[0], y - a[1], z - a[2]
        s = qx * T[0] + qy * T[1] + qz * T[2]
        t = np.clip(s / L, 0.0, 1.0)
        ax = s - t * L  # componente axial (≠0 solo fuera del segmento: tapas)
        ry = self.ry[i] + (self.ry[i + 1] - self.ry[i]) * t
        rz = self.rz[i] + (self.rz[i + 1] - self.rz[i]) * t
        N0, N1 = self.N[i], self.N[i + 1]
        B0, B1 = self.B[i], self.B[i + 1]
        nx = N0[0] + (N1[0] - N0[0]) * t
        ny = N0[1] + (N1[1] - N0[1]) * t
        nz = N0[2] + (N1[2] - N0[2]) * t
        bx = B0[0] + (B1[0] - B0[0]) * t
        by = B0[1] + (B1[1] - B0[1]) * t
        bz = B0[2] + (B1[2] - B0[2]) * t
        dx, dy, dz = qx - T[0] * t * L, qy - T[1] * t * L, qz - T[2] * t * L
        v = (dx * bx + dy * by + dz * bz) / rz
        tp = self.taper[i] + (self.taper[i + 1] - self.taper[i]) * t
        # taper > 0: sección más estrecha arriba (lomo, techo del cráneo)
        vc = np.clip(v, -1.0, 1.0)
        # tp < 0: estrecha la parte inferior (mandíbula en V)
        fac = np.where(tp >= 0, 1.0 - tp * (vc + 1.0) * 0.5, 1.0 + tp * (1.0 - vc) * 0.5)
        u = (dx * nx + dy * ny + dz * nz) / (ry * fac)
        rmin = np.minimum(ry, rz)
        if self.p == 2.0:
            lat2 = u * u + v * v
        else:
            lat2 = (np.abs(u) ** self.p + np.abs(v) ** self.p) ** (2.0 / self.p)
        n = np.sqrt(lat2 + (ax / rmin) ** 2)
        # escala por el semieje mínimo: subestima la distancia, suficiente para uniones suaves
        return ((n - 1.0) * rmin).astype(np.float32)

    def section_at(self, i, t):
        return self.c[i] + (self.c[i + 1] - self.c[i]) * t


class _Segment:
    def __init__(self, loft, i):
        self.loft, self.i = loft, i

    def bounds(self):
        l, i = self.loft, self.i
        r = max(l.ry[i], l.rz[i], l.ry[i + 1], l.rz[i + 1])
        a, b = l.c[i], l.c[i + 1]
        return np.minimum(a, b) - r, np.maximum(a, b) + r

    def eval(self, x, y, z):
        return self.loft._segment(self.i, x, y, z)


class Custom:
    """Primitiva con función arbitraria f(x, y, z) y caja dada."""

    def __init__(self, lo, hi, fn):
        self.lo, self.hi, self.fn = np.asarray(lo, float), np.asarray(hi, float), fn

    def bounds(self):
        return self.lo, self.hi

    def eval(self, x, y, z):
        return self.fn(x, y, z).astype(np.float32)


# ---------------------------------------------------------------------------
# Surface nets
# ---------------------------------------------------------------------------

_CORNERS = [(dx, dy, dz) for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)]
_EDGES = [(a, b) for a in range(8) for b in range(a + 1, 8)
          if sum(abs(np.subtract(_CORNERS[a], _CORNERS[b]))) == 1]


def surface_nets(field):
    f = field.f
    inside = f < 0
    nx, ny, nz = f.shape
    cnt = np.zeros((nx - 1, ny - 1, nz - 1), np.uint8)
    for dx, dy, dz in _CORNERS:
        cnt += inside[dx:nx - 1 + dx, dy:ny - 1 + dy, dz:nz - 1 + dz]
    active = (cnt > 0) & (cnt < 8)
    del cnt
    ci = np.nonzero(active)
    del active
    cells = np.stack(ci, 1)
    vals = np.stack([f[cells[:, 0] + dx, cells[:, 1] + dy, cells[:, 2] + dz] for dx, dy, dz in _CORNERS], 1)
    acc = np.zeros((len(cells), 3), np.float32)
    num = np.zeros(len(cells), np.float32)
    corners = np.array(_CORNERS, np.float32)
    for a, b in _EDGES:
        va, vb = vals[:, a], vals[:, b]
        m = (va < 0) != (vb < 0)
        t = np.where(m, va / np.where(m, va - vb, 1.0), 0.0)
        pos = corners[a] + (corners[b] - corners[a]) * t[:, None]
        acc += pos * m[:, None]
        num += m
    verts = field.bmin + (cells + acc / num[:, None]) * field.h
    lin = np.ravel_multi_index(ci, (nx - 1, ny - 1, nz - 1))

    def vid(i, j, k):
        q = np.ravel_multi_index((i, j, k), (nx - 1, ny - 1, nz - 1))
        return np.searchsorted(lin, q)

    faces = []
    for ax in range(3):
        u, v = (ax + 1) % 3, (ax + 2) % 3
        sl_a = [slice(1, n - 1) for n in f.shape]
        sl_b = list(sl_a)
        sl_a[ax] = slice(0, f.shape[ax] - 1)
        sl_b[ax] = slice(1, f.shape[ax])
        ia, ib = inside[tuple(sl_a)], inside[tuple(sl_b)]
        cross = ia != ib
        p = np.nonzero(cross)
        flip = ia[p]
        p = [p[d] + (0 if d == ax else 1) for d in range(3)]
        def cell(du, dv):
            q = [p[0].copy(), p[1].copy(), p[2].copy()]
            q[u] -= du
            q[v] -= dv
            return vid(*q)
        c00, c10, c11, c01 = cell(1, 1), cell(0, 1), cell(0, 0), cell(1, 0)
        quad = np.stack([c00, c10, c11, c01], 1)
        quad[~flip] = quad[~flip][:, ::-1]
        faces.append(quad)
    return verts.astype(np.float64), np.concatenate(faces)


# ---------------------------------------------------------------------------
# Post-proceso: relajación tangencial + reproyección a la isosuperficie
# ---------------------------------------------------------------------------

def vertex_neighbors(nv, faces):
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 3]], faces[:, [3, 0]]])
    e = np.concatenate([e, e[:, ::-1]])
    return e[:, 0], e[:, 1]


def relax(verts, faces, field, iterations=4, factor=0.5):
    src, dst = vertex_neighbors(len(verts), faces)
    deg = np.bincount(src, minlength=len(verts)).astype(float)
    for _ in range(iterations):
        avg = np.zeros_like(verts)
        for d in range(3):
            avg[:, d] = np.bincount(src, weights=verts[dst, d], minlength=len(verts))
        avg /= np.maximum(deg, 1)[:, None]
        verts = verts + (avg - verts) * factor
        verts = project(verts, field, steps=2)
    return verts


def project(verts, field, steps=3):
    for _ in range(steps):
        d = field.sample(verts)
        g = field.gradient(verts)
        g2 = (g * g).sum(1) + 1e-9
        verts = verts - (d / g2)[:, None] * g
    return verts
