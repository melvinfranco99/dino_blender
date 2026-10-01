"""
Anatomía de Tyrannosaurus rex (adulto, ~12 m) construida como campo de distancia.

Convenciones: metros, el animal mira hacia +X, Z arriba, +Y es su lado izquierdo.
La pose (zancada, cuello girado hacia la cámara y fauces abiertas) se esculpe
directamente en el campo para que la piel sea continua en las comisuras de la boca.

Proporciones orientativas basadas en esqueletos montados (FMNH PR 2081 "Sue"):
cráneo ~1,4 m, fémur ~1,3 m, tibia ~1,2 m, metatarso ~0,6 m, cadera a ~3,1 m del suelo.
"""

import math

import numpy as np

from . import sdf
from .sdf import Ellipsoid, Loft, RoundCone, normalize

# ---------------------------------------------------------------------------
# Tronco, cuello y cola: (x, y, z_dorso, z_vientre, semiancho)
# ---------------------------------------------------------------------------

TRUNK = [
    (-7.05, -1.30, 2.03, 1.98, 0.022),
    (-6.70, -1.12, 2.10, 1.99, 0.045),
    (-6.20, -0.86, 2.26, 2.06, 0.080),
    (-5.55, -0.56, 2.50, 2.18, 0.125),
    (-4.80, -0.30, 2.82, 2.32, 0.185),
    (-3.95, -0.12, 3.17, 2.46, 0.255),
    (-3.05, -0.03, 3.48, 2.55, 0.335),
    (-2.15, 0.00, 3.72, 2.56, 0.420),
    (-1.25, 0.00, 3.88, 2.50, 0.500),
    (-0.40, 0.00, 3.94, 2.36, 0.560),
    (0.40, 0.00, 3.93, 2.16, 0.640),
    (1.15, 0.00, 3.88, 1.99, 0.710),
    (1.85, 0.01, 3.82, 1.97, 0.720),
    (2.45, 0.03, 3.80, 2.12, 0.660),
    (2.95, 0.07, 3.86, 2.42, 0.560),
    (3.35, 0.12, 3.97, 2.88, 0.430),
    (3.68, 0.18, 4.08, 3.14, 0.370),
    (3.95, 0.25, 4.15, 3.30, 0.350),
]

# ---------------------------------------------------------------------------
# Cabeza: marco local (u adelante, v izquierda, w arriba) con origen en el cráneo
# ---------------------------------------------------------------------------

HEAD_ORIGIN = np.array([4.18, 0.31, 3.64])
HEAD_YAW = math.radians(21.0)     # gira hacia la cámara (+Y)
HEAD_PITCH = math.radians(6.0)    # hocico ligeramente alzado: rugido
HEAD_ROLL = math.radians(-5.0)    # inclinado hacia la cámara
JAW_OPEN = math.radians(27.0)
HEAD_SCALE = 1.0                  # cráneo ~1,5 m con tejidos blandos
HINGE = np.array([-0.02, 0.0, -0.27])  # articulación cuadrado-articular (local)

# Cráneo: (u, w_techo, w_labio, semiancho)
SKULL = [
    (-0.20, 0.46, -0.14, 0.36),
    (-0.05, 0.60, -0.27, 0.46),
    (0.16, 0.65, -0.31, 0.46),
    (0.34, 0.62, -0.32, 0.40),
    (0.50, 0.56, -0.31, 0.31),
    (0.68, 0.49, -0.30, 0.215),
    (0.88, 0.42, -0.28, 0.185),
    (1.06, 0.36, -0.255, 0.17),
    (1.20, 0.27, -0.23, 0.15),
    (1.31, 0.12, -0.17, 0.105),
]

# Mandíbula en reposo (cerrada): (u, w_borde, w_inferior, semiancho)
JAW = [
    (-0.16, -0.13, -0.52, 0.34),
    (0.08, -0.22, -0.74, 0.40),
    (0.33, -0.28, -0.76, 0.34),
    (0.60, -0.29, -0.68, 0.25),
    (0.88, -0.27, -0.61, 0.20),
    (1.08, -0.24, -0.55, 0.17),
    (1.21, -0.215, -0.50, 0.145),
    (1.28, -0.20, -0.42, 0.115),
]


class HeadFrame:
    def __init__(self):
        cy, sy = math.cos(HEAD_YAW), math.sin(HEAD_YAW)
        cp, sp = math.cos(HEAD_PITCH), math.sin(HEAD_PITCH)
        f = np.array([cy * cp, sy * cp, sp])
        lft = np.array([-sy, cy, 0.0])
        up = np.cross(f, lft)
        cr, sr = math.cos(HEAD_ROLL), math.sin(HEAD_ROLL)
        self.F = f
        self.L = lft * cr + up * sr
        self.U = up * cr - lft * sr
        self.O = HEAD_ORIGIN

    def p(self, u, v, w):
        k = HEAD_SCALE
        return self.O + k * (u * self.F + v * self.L + w * self.U)

    def jaw_local(self, u, v, w):
        """Transforma un punto de la mandíbula cerrada a la posición abierta (local)."""
        du, dw = u - HINGE[0], w - HINGE[2]
        c, s = math.cos(JAW_OPEN), math.sin(JAW_OPEN)
        return HINGE[0] + du * c + dw * s, v, HINGE[2] - du * s + dw * c

    def jp(self, u, v, w):
        return self.p(*self.jaw_local(u, v, w))

    def jaw_up(self):
        c, s = math.cos(JAW_OPEN), math.sin(JAW_OPEN)
        return self.U * c + self.F * s

    def jaw_fwd(self):
        c, s = math.cos(JAW_OPEN), math.sin(JAW_OPEN)
        return self.F * c - self.U * s


def profile(table, col):
    t = np.array(table)
    return lambda u: np.interp(u, t[:, 0], t[:, col])


skull_top, skull_lip, skull_hw = profile(SKULL, 1), profile(SKULL, 2), profile(SKULL, 3)
jaw_top, jaw_bot, jaw_hw = profile(JAW, 1), profile(JAW, 2), profile(JAW, 3)

# ---------------------------------------------------------------------------
# Patas traseras (zancada: izquierda adelantada y apoyada, derecha atrás despegando)
# ---------------------------------------------------------------------------

LEGS = {
    'L': dict(hip=(0.05, 0.56, 3.12), knee=(0.72, 0.80, 2.05), ankle=(0.30, 0.70, 0.74),
              ball=(0.70, 0.68, 0.13), spread=1.0, heel_lift=0.0),
    'R': dict(hip=(-0.05, -0.56, 3.10), knee=(0.18, -0.78, 1.98), ankle=(-0.62, -0.68, 0.86),
              ball=(-0.43, -0.64, 0.14), spread=0.85, heel_lift=1.0),
}

# Dedos II, III, IV: (ángulo de apertura en grados, longitud, radio base)
TOES = [(-24.0, 0.44, 0.088), (0.0, 0.54, 0.098), (22.0, 0.46, 0.088)]

ARMS = {
    s: dict(shoulder=(2.42, 0.50 * k, 2.80), elbow=(2.45, 0.70 * k, 2.42),
            wrist=(2.66, 0.67 * k, 2.32))
    for s, k in (('L', 1), ('R', -1))
}


class Dino:
    """Construye el campo y guarda puntos clave para accesorios (dientes, ojos, garras)."""

    def __init__(self):
        self.hf = HeadFrame()
        self.toe_tips = []      # (punta, dirección, radio)
        self.finger_tips = []
        self.eyes = []          # (centro, dirección, radio)
        self.mouth_parts = []   # primitivas que definen el interior de la boca
        self.lip_line = {}      # para colocar dientes

    # -- tronco ------------------------------------------------------------
    def trunk(self):
        t = sdf.catmull_rom(TRUNK, 10)
        centers = np.stack([t[:, 0], t[:, 1], (t[:, 2] + t[:, 3]) / 2], 1)
        rz = (t[:, 2] - t[:, 3]) / 2
        # lomo más estrecho que el vientre (sección en pera), menos marcado en la cola
        taper = 0.22 * sdf.smoothstep(-5.0, -1.0, t[:, 0])
        return Loft(centers, t[:, 4], rz, taper=taper)

    def dorsal_ridge(self, F):
        """Cresta de las espinas neurales a lo largo del lomo y la cola."""
        t = sdf.catmull_rom(TRUNK, 4)
        sel = (t[:, 0] > -6.4) & (t[:, 0] < 3.6)
        t = t[sel]
        centers = np.stack([t[:, 0], t[:, 1], t[:, 2] - 0.05], 1)
        r = 0.035 + 0.05 * np.clip(t[:, 4] / 0.7, 0, 1)
        F.apply(Loft(centers, r, r * 1.3), 'union', 0.09)

    # -- cabeza ------------------------------------------------------------
    def skull(self):
        hf = self.hf
        t = sdf.catmull_rom(SKULL, 4)
        us, top, lip, hw = t.T
        centers = [hf.p(u, 0, (a + b) / 2) for u, a, b in zip(us, top, lip)]
        # el hocico es alto y estrecho por arriba; la nuca es ancha y plana
        taper = 0.32 * sdf.smoothstep(0.1, 0.6, us) + 0.12
        return Loft(centers, hw * HEAD_SCALE, (top - lip) / 2 * HEAD_SCALE, power=3.0, up=hf.U, taper=taper)

    def jaw(self):
        hf = self.hf
        t = sdf.catmull_rom(JAW, 4)
        us, top, bot, hw = t.T
        centers = [hf.jp(u, 0, (a + b) / 2) for u, a, b in zip(us, top, bot)]
        # sección en V: estrecha abajo, ancha en el borde labial
        return Loft(centers, hw * HEAD_SCALE, (top - bot) / 2 * HEAD_SCALE, power=2.3, up=hf.jaw_up(), taper=-0.4)

    def surf(self, F, origin, direction, reach=1.2):
        """Punto de la superficie actual del campo a lo largo de un rayo desde el interior."""
        d = normalize(direction)
        steps = np.arange(0.0, reach, F.h * 0.25)
        pts = origin + steps[:, None] * d
        vals = F.sample(pts)
        i = int(np.argmax(vals > 0))
        return pts[i], d

    def head_details(self, F):
        hf = self.hf
        Fw, U = hf.F, hf.U
        for s in (1, -1):
            L = hf.L * s
            # musculatura temporal (aductores) que abomba el techo posterior del cráneo
            F.apply(Ellipsoid(hf.p(-0.04, 0.17 * s, 0.42), (0.24, 0.15, 0.10), Fw, U), 'union', 0.08)
            # arco yugal: cresta ósea bajo el ojo hasta la articulación de la mandíbula
            p, n = self.surf(F, hf.p(0.12, 0, -0.10), L * 0.95 - U * 0.1)
            F.apply(Ellipsoid(p - n * 0.025, (0.30, 0.04, 0.075), Fw - U * 0.12, U), 'union', 0.05)
            # cuerno lacrimal y protuberancia postorbital (rugosidades bajas)
            p, n = self.surf(F, hf.p(0.52, 0, 0.30), L * 0.55 + U * 0.85)
            F.apply(Ellipsoid(p - n * 0.025, (0.10, 0.03, 0.02), Fw + U * 0.2, n), 'union', 0.05)
            p, n = self.surf(F, hf.p(0.20, 0, 0.30), L * 0.75 + U * 0.65)
            F.apply(Ellipsoid(p - n * 0.015, (0.08, 0.04, 0.035), Fw, n), 'union', 0.04)
            # fosa nasal
            p, n = self.surf(F, hf.p(1.17, 0, 0.1), L * 0.75 + U * 0.45 + Fw * 0.25)
            F.apply(Ellipsoid(p + n * 0.006, (0.042, 0.016, 0.022), Fw + U * 0.15, n), 'subtract', 0.014)
            # ojo: orientado hacia delante (visión binocular), hundido bajo la ceja
            eye_dir = normalize(Fw * 0.55 + L * 0.82 + U * 0.08)
            p, n = self.surf(F, hf.p(0.32, 0, 0.27), eye_dir)
            r_eye = 0.07 * HEAD_SCALE
            c = p - n * 0.048 * HEAD_SCALE
            F.apply(Ellipsoid(c, (r_eye + 0.004,) * 3), 'subtract', 0.01)
            # ceja (palpebral) que sombrea el ojo y párpado inferior
            F.apply(Ellipsoid(c + U * 0.075 - Fw * 0.015, (0.12, 0.065, 0.035), Fw, U),
                    'union', 0.035)
            F.apply(Ellipsoid(c - U * 0.06 + n * 0.022, (0.09, 0.035, 0.022), Fw, U), 'union', 0.03)
            self.eyes.append((c, n, r_eye, s))

    @staticmethod
    def lip_path(upper, n_side=16, n_front=9):
        """Recorrido (u, v) del borde alveolar: lado izquierdo, vuelta por la punta, lado derecho."""
        hw = skull_hw if upper else jaw_hw
        inset, u_end = (0.035, 1.25) if upper else (0.035, 1.20)
        front = 0.1 if upper else 0.085
        us = np.linspace(0.14 if upper else 0.2, u_end, n_side)
        left = [(u, hw(u) - inset) for u in us]
        rv = hw(u_end) - inset
        arc = [(u_end + front * math.cos(a), rv * math.sin(a))
               for a in np.linspace(math.pi / 2, -math.pi / 2, n_front)[1:-1]]
        right = [(u, -v) for u, v in reversed(left)]
        return left + arc + right

    def mouth(self, F):
        """Talla el interior de la boca: paladar, suelo mandibular, garganta y lengua."""
        hf = self.hf
        K = HEAD_SCALE
        # rebordes labiales/encías: el borde de la boca es lo más bajo del cráneo
        path = self.lip_path(True)
        fade = np.array([0.35 + 0.65 * sdf.smoothstep(0.14, 0.5, u) for u, v in path])
        F.apply(Loft([hf.p(u, v, skull_lip(min(u, 1.31)) + 0.035) for u, v in path],
                     0.032 * K * fade, 0.045 * K * fade, up=hf.U), 'union', 0.07)
        us = np.linspace(0.1, 1.45, 16)
        lip, hw = skull_lip(us), skull_hw(np.minimum(us, 1.28))
        inner = np.maximum(hw - 0.075, 0.04) * HEAD_SCALE
        palate = Loft([hf.p(u, 0, l - 0.01) for u, l in zip(us, lip)], inner,
                      np.full_like(us, 0.075 * HEAD_SCALE), power=2.6, up=hf.U)
        jtop, jhw = jaw_top(us), jaw_hw(np.minimum(us, 1.25))
        jinner = np.maximum(jhw - 0.07, 0.03) * HEAD_SCALE
        floor = Loft([hf.jp(u, 0, t + 0.01) for u, t in zip(us, jtop)], jinner,
                     np.full_like(us, 0.08 * HEAD_SCALE), power=2.6, up=hf.jaw_up())
        # garganta: hueco oscuro entre comisuras
        mid_a = (hf.p(0.45, 0, -0.33) + hf.jp(0.45, 0, -0.24)) / 2
        mid_b = (hf.p(-0.05, 0, -0.25) + hf.jp(-0.05, 0, -0.30)) / 2
        throat = RoundCone(mid_a, mid_b, 0.13 * HEAD_SCALE, 0.09 * HEAD_SCALE)
        # cuña central que abre el hueco entre las mandíbulas hasta la comisura
        for part in (palate, floor, throat):
            F.apply(part, 'subtract', 0.03)
            self.mouth_parts.append(part)
        # tejido de la comisura (cierra el rincón entre cráneo y mandíbula)
        for s in (1, -1):
            u0 = 0.1
            w0 = 0.5 * (skull_lip(u0) + hf.jaw_local(u0, 0, jaw_top(u0))[2])
            F.apply(Ellipsoid(hf.p(u0, s * (skull_hw(u0) - 0.07), w0), (0.13 * K, 0.06 * K, 0.08 * K),
                              hf.F, hf.U), 'union', 0.06)
        # lengua
        tu = np.linspace(0.05, 0.95, 10)
        tongue = Loft([hf.jp(u, 0, jaw_top(u) - 0.035 - 0.03 * (1 - u)) for u in tu],
                      (0.12 - 0.05 * tu) * HEAD_SCALE, (0.045 + 0.02 * (1 - tu)) * HEAD_SCALE, up=hf.jaw_up())
        F.apply(tongue, 'union', 0.04)
        self.mouth_parts.append(tongue)
        # piel de la garganta (gular) que une la mandíbula abierta con el cuello
        F.apply(RoundCone(hf.jp(0.45, 0, -0.5), (3.5, 0.16, 3.2), 0.12, 0.22), 'union', 0.1)

    # -- patas -------------------------------------------------------------
    def leg(self, F, side):
        g = LEGS[side]
        s = 1 if side == 'L' else -1
        hip, knee, ankle, ball = (np.array(g[k]) for k in ('hip', 'knee', 'ankle', 'ball'))
        fem = knee - hip
        # muslo: masa muscular enorme (iliotibial, femorotibial)
        F.apply(Ellipsoid(hip + fem * 0.42 + np.array([0.0, 0.14 * s, 0.0]), (0.95, 0.45, 0.62),
                          fem), 'union', 0.16)
        # caudofemoral: del muslo a la base de la cola
        F.apply(RoundCone(hip + fem * 0.3 + np.array([-0.25, -0.05 * s, 0.05]),
                          np.array([-2.2, 0.22 * s, 2.95]), 0.40, 0.22), 'union', 0.25)
        F.apply(RoundCone(hip, knee, 0.32, 0.25), 'union', 0.15)
        # rodilla y tibia con gemelos
        tib = ankle - knee
        F.apply(Ellipsoid(knee + tib * 0.06, (0.24, 0.24, 0.24)), 'union', 0.12)
        F.apply(RoundCone(knee, ankle, 0.27, 0.14), 'union', 0.10)
        back = normalize(np.cross(tib, np.array([0, 1.0, 0])))  # hacia atrás de la pierna
        F.apply(Ellipsoid(knee + tib * 0.30 + back * 0.10, (0.48, 0.19, 0.22), tib,
                          back), 'union', 0.12)
        # tobillo y metatarso
        F.apply(Ellipsoid(ankle, (0.15, 0.13, 0.15)), 'union', 0.06)
        F.apply(RoundCone(ankle, ball, 0.125, 0.11), 'union', 0.06)
        # almohadilla bajo el metatarso
        F.apply(Ellipsoid(ball + np.array([-0.05, 0, -0.01]), (0.19, 0.15, 0.11), ball - ankle), 'union', 0.05)
        # dedos
        meta = normalize(np.array([ball[0] - ankle[0], ball[1] - ankle[1], 0.0]))
        for i, (ang, length, r0) in enumerate(TOES):
            a = math.radians(ang * s * g['spread'])
            d = np.array([meta[0] * math.cos(a) - meta[1] * math.sin(a),
                          meta[0] * math.sin(a) + meta[1] * math.cos(a), 0.0])
            lat = np.array([-d[1], d[0], 0.0])
            start = ball + lat * (i - 1) * 0.07 + d * 0.06
            start[2] = 0.11
            n_ph = 3
            seg = length / n_ph
            p = start.copy()
            r = r0
            for k in range(n_ph):
                q = p + d * seg
                q[2] = max(r * 0.75, 0.035)
                F.apply(RoundCone(p, q, r, r * 0.84), 'union', 0.035)
                # almohadilla digital
                F.apply(Ellipsoid(p + d * seg * 0.5 + np.array([0, 0, -r * 0.25]),
                                  (seg * 0.55, r * 1.05, r * 0.7), d), 'union', 0.03)
                p, r = q, r * 0.84
            self.toe_tips.append((p - d * 0.01, normalize(d + np.array([0, 0, -0.35])), r))
        # dedo I (espolón) en la cara interna del metatarso
        dew = ankle + (ball - ankle) * 0.62 + np.array([0.0, -0.11 * s, 0.0])
        dew_tip = dew + np.array([0.06, -0.05 * s, -0.10])
        F.apply(RoundCone(dew, dew_tip, 0.05, 0.035), 'union', 0.03)
        self.toe_tips.append((dew_tip, normalize(dew_tip - dew), 0.035))

    def arm(self, F, side):
        g = ARMS[side]
        sh, el, wr = (np.array(g[k]) for k in ('shoulder', 'elbow', 'wrist'))
        s = 1 if side == 'L' else -1
        F.apply(Ellipsoid((sh + el) / 2, (0.24, 0.10, 0.11), el - sh), 'union', 0.08)
        F.apply(RoundCone(el, wr, 0.075, 0.055), 'union', 0.04)
        for k, off in enumerate((-0.035, 0.035)):
            base = wr + np.array([0.02, off * s, 0.0])
            d = normalize(np.array([0.85, 0.12 * s * (1 if k else -0.4), -0.55]))
            mid = base + d * 0.10
            tip = mid + normalize(d + np.array([0, 0, -0.5])) * 0.08
            F.apply(RoundCone(base, mid, 0.04, 0.032), 'union', 0.02)
            F.apply(RoundCone(mid, tip, 0.032, 0.026), 'union', 0.015)
            self.finger_tips.append((tip, normalize(tip - mid), 0.026))

    # -- ensamblado --------------------------------------------------------
    def build(self, h=0.02):
        bmin = np.array([-7.2, -1.55, -0.12])
        bmax = np.array([5.95, 1.35, 4.95])
        F = sdf.Field(bmin, bmax, h)
        F.apply(self.trunk(), 'union')
        self.dorsal_ridge(F)
        F.apply(self.skull(), 'union', 0.2)
        F.apply(self.jaw(), 'union', 0.05)
        self.head_details(F)
        for s in ('L', 'R'):
            self.leg(F, s)
            self.arm(F, s)
        self.mouth(F)
        self.surface_noise(F)
        # los pies apoyan: recorta ligeramente bajo el suelo para un contacto plano
        F.f[:, :, : int(round((0.0 - bmin[2]) / h))] = np.maximum(
            F.f[:, :, : int(round((0.0 - bmin[2]) / h))], 0.02)
        self.field = F
        return F

    def surface_noise(self, F):
        """Abultamientos orgánicos de baja frecuencia (músculo, grasa, rugosidad del hocico)."""
        band = np.abs(F.f) < 0.06
        idx = np.nonzero(band)
        p = F.bmin + np.stack(idx, 1) * F.h
        n = sdf.fbm(p * 2.2, octaves=3, seed=3) * 0.012
        # rugosidad nasal y facial más marcada
        hf = self.hf
        rel = p - hf.O
        u = rel @ hf.F
        w = rel @ hf.U
        face = (sdf.smoothstep(-0.3, 0.1, u) * sdf.smoothstep(1.6, 1.4, u)
                * sdf.smoothstep(-0.1, 0.2, w)) * (np.linalg.norm(rel, axis=1) < 1.8)
        n += face * sdf.fbm(p * 9.0, octaves=3, seed=11) * 0.012
        F.f[idx] += n.astype(np.float32)

    # -- regiones (atributos para los materiales) --------------------------
    def regions(self, verts):
        """Atributos por vértice para el sombreado (todos en [0, 1])."""
        F, hf = self.field, self.hf
        x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
        g = F.gradient(verts)
        nrm = g / (np.linalg.norm(g, axis=1, keepdims=True) + 1e-9)
        nz = nrm[:, 2]

        # boca: superficies talladas por el interior (paladar, suelo, garganta, lengua)
        d_mouth = np.full(len(verts), 10.0)
        for part in self.mouth_parts[:-1]:
            d_mouth = np.minimum(d_mouth, part.eval(x, y, z))
        d_tongue = self.mouth_parts[-1].eval(x, y, z)
        boca = (np.minimum(d_mouth, d_tongue) < 0.012).astype(float)
        lengua = (d_tongue < 0.02).astype(float)
        labio = (1 - sdf.smoothstep(0.02, 0.09, d_mouth)) * (1 - boca)

        rel = verts - hf.O
        u = rel @ hf.F
        cabeza = sdf.smoothstep(-0.55, -0.2, u) * (np.linalg.norm(rel, axis=1) < 2.2)

        # contrasombreado: altura relativa dentro de la sección del tronco
        t = sdf.catmull_rom(TRUNK, 6)
        zt, zb = np.interp(x, t[:, 0], t[:, 2]), np.interp(x, t[:, 0], t[:, 3])
        yc, hw = np.interp(x, t[:, 0], t[:, 1]), np.interp(x, t[:, 0], t[:, 4])
        hrel = np.clip((z - zb) / np.maximum(zt - zb, 0.05), 0, 1)
        trunk_d = 0.7 * hrel + 0.3 * (0.5 + 0.5 * nz)
        limb = np.clip(np.maximum((zb - 0.05 - z) / 0.4, (np.abs(y - yc) - hw * 1.1) / 0.25), 0, 1)
        limb_d = 0.42 + 0.3 * nz
        head_d = np.clip(0.38 + 0.5 * nz, 0, 1)
        dorso = trunk_d * (1 - limb) + limb_d * limb
        dorso = dorso * (1 - cabeza) + head_d * cabeza

        # pliegues del cuello (más marcados en el lado hacia el que gira) y la garganta
        neck = sdf.smoothstep(2.55, 3.05, x) * (1 - sdf.smoothstep(-0.45, -0.1, u))
        side = 0.55 + 0.45 * np.clip(nrm[:, 1] * 1.5, -1, 1)
        throat = sdf.smoothstep(0.1, -0.6, nz) * sdf.smoothstep(2.9, 3.4, x) * (1 - sdf.smoothstep(0.2, 0.6, u))
        body = sdf.smoothstep(-3.5, -1.5, x) * (1 - neck) * (1 - sdf.smoothstep(0.2, 0.8, nz)) * 0.12
        pl_cuello = np.clip(neck * side * (1 - sdf.smoothstep(0.4, 0.95, nz)) + throat + body, 0, 1)

        # pliegues en articulaciones: rodillas, tobillos, base de los dedos, codos y axilas
        joints = []
        for leg in LEGS.values():
            joints += [(leg['knee'], 0.45), (leg['ankle'], 0.32), (leg['ball'], 0.30)]
        for arm in ARMS.values():
            joints += [(arm['shoulder'], 0.3), (arm['elbow'], 0.18)]
        pl_pata = np.zeros(len(verts))
        for p, r in joints:
            dist = np.linalg.norm(verts - np.asarray(p), axis=1)
            pl_pata = np.maximum(pl_pata, 1 - sdf.smoothstep(r * 0.5, r, dist))
        pl_pata = np.maximum(pl_pata, sdf.smoothstep(0.35, 0.1, z) * 0.8)  # dedos

        # oclusión ambiental aproximada por muestreo del campo a lo largo de la normal
        occ = np.zeros(len(verts))
        for i, dd in enumerate((0.03, 0.08, 0.16, 0.3)):
            occ += (dd - np.minimum(F.sample(verts + nrm * dd), dd)) / dd * (0.5 ** i)
        suciedad = np.clip(1.0 - occ * 0.6, 0, 1)

        return dict(dorso=dorso, cabeza=cabeza, labio=labio, lengua=lengua,
                    pliegue_cuello=pl_cuello, pliegue_pata=pl_pata, suciedad=suciedad,
                    boca=boca), nrm

    # -- dientes -----------------------------------------------------------
    def surf_in(self, origin, direction, reach=1.0):
        """Primer punto sólido marchando desde fuera hacia `origin`."""
        F = self.field
        d = normalize(direction)
        steps = np.arange(reach, 0.0, -F.h * 0.25)
        pts = origin + steps[:, None] * d
        vals = F.sample(pts)
        i = int(np.argmax(vals < 0))
        return pts[i]

    def first_solid(self, start, direction, reach=0.6):
        """Primer punto sólido marchando desde `start` (en el aire) a lo largo de `direction`."""
        F = self.field
        d = normalize(direction)
        steps = np.arange(0.0, reach, F.h * 0.2)
        vals = F.sample(start + steps[:, None] * d)
        hit = vals < 0
        if not hit.any():
            return None
        return start + steps[int(np.argmax(hit))] * d

    def teeth_layout(self, rng):
        """Lista de dientes: (base, dirección, atrás, longitud, radio, mandíbula_inferior).
        La base se busca lanzando un rayo vertical hacia el borde del labio, de modo que el
        diente siempre nace de la encía, sea cual sea la forma final del cráneo."""
        hf, K = self.hf, HEAD_SCALE
        ju, jf = hf.jaw_up(), hf.jaw_fwd()
        teeth = []
        for s in (1, -1):
            L = hf.L * s
            upper = []
            path = [(u, v) for u, v in self.lip_path(True) if v * s > 0]
            path.sort(key=lambda p: -p[0])
            # premaxilares: pequeños, en D, en la curva de la punta del hocico
            for k, (u, v) in enumerate([p for p in path if p[0] > 1.25][:4]):
                upper.append((u, v, 0.055 + 0.01 * k, -hf.U - hf.F * 0.05))
            # maxilares: los mayores hacia el tercio anterior
            side = [p for p in path if p[0] <= 1.25]
            for k, (u, v) in enumerate(side[::max(len(side) // 12, 1)][:12]):
                length = 0.06 + 0.078 * math.exp(-((k - 3) / 3.3) ** 2)
                d = -hf.U - hf.F * rng.uniform(0.08, 0.22) + L * rng.uniform(-0.06, 0.04)
                upper.append((u, v, length, d))
            for k, (u, v, length, d) in enumerate(upper):
                hit = self.first_solid(hf.p(u, v, skull_lip(min(u, 1.31)) - 0.25), hf.U)
                if hit is None:
                    continue
                r = rng.random()
                if k >= 4 and r < 0.07:
                    continue                    # diente caído
                if k >= 4 and r < 0.15:
                    length *= 0.55              # diente de reemplazo o roto
                teeth.append((hit + hf.U * 0.015, normalize(d), -hf.F,
                              K * length * rng.uniform(0.9, 1.08), 0.33, False))
            # dentarios (mandíbula)
            path = [(u, v) for u, v in self.lip_path(False, n_side=13) if v * s > 0]
            path.sort(key=lambda p: -p[0])
            for k, (u, v) in enumerate(path[:13]):
                length = 0.05 + 0.058 * math.exp(-((k - 2.5) / 3.6) ** 2)
                hit = self.first_solid(hf.jp(u, v, jaw_top(min(u, 1.28)) + 0.25), -ju)
                if hit is None:
                    continue
                r = rng.random()
                if r < 0.06:
                    continue
                if r < 0.13:
                    length *= 0.6
                d = normalize(ju - jf * rng.uniform(0.05, 0.2) + L * rng.uniform(-0.05, 0.03))
                teeth.append((hit - ju * 0.015, d, -jf, K * length * rng.uniform(0.9, 1.08), 0.32, True))
        return teeth
