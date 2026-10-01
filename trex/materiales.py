"""
Materiales procedurales (Cycles) del T. rex y del entorno.

La piel combina varias capas: contrasombreado dorso/vientre, manchas y bandas, escamas de
tamaño variable por región, escamas de ornamento dispersas, pliegues en cuello y
articulaciones, barro seco en las patas, suciedad en cavidades y labios húmedos.
El relieve se aplica como desplazamiento real (con subdivisión adaptativa) más bump fino.
"""

import bpy

HAZE_COLOR = (0.62, 0.66, 0.72)
HAZE_STRENGTH = 1.0
HAZE_DISTANCE = 260.0


class NB:
    """Ayudante mínimo para construir árboles de nodos."""

    def __init__(self, mat):
        mat.use_nodes = True
        self.nt = mat.node_tree
        self.nt.nodes.clear()
        self.mat = mat

    def n(self, kind, x=0, y=0, **props):
        node = self.nt.nodes.new(kind)
        node.location = (x, y)
        for k, v in props.items():
            if k.startswith('i_'):
                key = k[2:].replace('_', ' ')
                node.inputs[key].default_value = v
            else:
                setattr(node, k, v)
        return node

    def l(self, a, b):
        self.nt.links.new(a, b)

    # ayudantes de alto nivel ------------------------------------------------
    def attr(self, name, x=-2000, y=0):
        return self.n('ShaderNodeAttribute', x, y, attribute_name=name).outputs['Fac']

    def math(self, op, a, b=None, x=0, y=0, clamp=False):
        m = self.n('ShaderNodeMath', x, y, operation=op, use_clamp=clamp)
        for i, v in enumerate((a, b)):
            if v is None:
                continue
            if isinstance(v, (int, float)):
                m.inputs[i].default_value = v
            else:
                self.l(v, m.inputs[i])
        return m.outputs[0]

    def mad(self, a, b, c, x=0, y=0):
        """a * b + c"""
        m = self.n('ShaderNodeMath', x, y, operation='MULTIPLY_ADD')
        for i, v in enumerate((a, b, c)):
            if isinstance(v, (int, float)):
                m.inputs[i].default_value = v
            else:
                self.l(v, m.inputs[i])
        return m.outputs[0]

    def mix(self, fac, a, b, blend='MIX', x=0, y=0):
        m = self.n('ShaderNodeMix', x, y, data_type='RGBA', blend_type=blend, clamp_result=True)
        for sock, v in ((m.inputs[0], fac), (m.inputs[6], a), (m.inputs[7], b)):
            if isinstance(v, (int, float)):
                sock.default_value = v
            elif isinstance(v, tuple):
                sock.default_value = v if len(v) == 4 else (*v, 1.0)
            else:
                self.l(v, sock)
        return m.outputs[2]

    def mixf(self, fac, a, b, x=0, y=0):
        m = self.n('ShaderNodeMix', x, y, data_type='FLOAT', clamp_factor=True)
        for sock, v in ((m.inputs[0], fac), (m.inputs[2], a), (m.inputs[3], b)):
            if isinstance(v, (int, float)):
                sock.default_value = v
            else:
                self.l(v, sock)
        return m.outputs[0]

    def ramp(self, fac, stops, x=0, y=0, interp='LINEAR'):
        r = self.n('ShaderNodeValToRGB', x, y)
        r.color_ramp.interpolation = interp
        els = r.color_ramp.elements
        while len(els) > len(stops):
            els.remove(els[-1])
        while len(els) < len(stops):
            els.new(0.5)
        for el, (pos, col) in zip(els, stops):
            el.position = pos
            if isinstance(col, (int, float)):
                col = (col, col, col)
            el.color = col if len(col) == 4 else (*col, 1.0)
        self.l(fac, r.inputs['Fac'])
        return r.outputs['Color']

    def maprange(self, v, a, b, c=0.0, d=1.0, x=0, y=0, smooth=False):
        m = self.n('ShaderNodeMapRange', x, y, interpolation_type='SMOOTHSTEP' if smooth else 'LINEAR')
        m.inputs['From Min'].default_value = a
        m.inputs['From Max'].default_value = b
        m.inputs['To Min'].default_value = c
        m.inputs['To Max'].default_value = d
        self.l(v, m.inputs['Value'])
        return m.outputs['Result']

    def noise(self, vec, scale, detail=4.0, rough=0.5, distortion=0.0, x=0, y=0, dims='3D', w=0.0):
        n = self.n('ShaderNodeTexNoise', x, y, noise_dimensions=dims)
        n.inputs['Scale'].default_value = scale
        n.inputs['Detail'].default_value = detail
        n.inputs['Roughness'].default_value = rough
        n.inputs['Distortion'].default_value = distortion
        if dims == '4D':
            n.inputs['W'].default_value = w
        if vec is not None:
            self.l(vec, n.inputs['Vector'])
        return n

    def voronoi(self, vec, scale, feature='F1', rnd=1.0, x=0, y=0, metric='EUCLIDEAN', detail=0.0):
        v = self.n('ShaderNodeTexVoronoi', x, y, feature=feature, distance=metric)
        v.inputs['Scale'].default_value = scale
        v.inputs['Randomness'].default_value = rnd
        if 'Detail' in v.inputs:
            v.inputs['Detail'].default_value = detail
        if vec is not None:
            self.l(vec, v.inputs['Vector'])
        return v

    def output(self, surface, displacement=None, volume=None, x=1400, y=0):
        o = self.n('ShaderNodeOutputMaterial', x, y)
        self.l(surface, o.inputs['Surface'])
        if displacement is not None:
            self.l(displacement, o.inputs['Displacement'])
        return o


def principled(nb, x=800, y=0, **inputs):
    p = nb.n('ShaderNodeBsdfPrincipled', x, y, subsurface_method='BURLEY')
    for k, v in inputs.items():
        key = k.replace('_', ' ')
        if isinstance(v, (int, float, tuple)):
            if isinstance(v, tuple) and len(v) == 3 and p.inputs[key].type == 'RGBA':
                v = (*v, 1.0)
            p.inputs[key].default_value = v
        else:
            nb.l(v, p.inputs[key])
    return p


def with_haze(nb, shader_out, x=1100, y=0):
    """Perspectiva aérea: mezcla con el color del cielo según la distancia a la cámara."""
    cam = nb.n('ShaderNodeCameraData', x - 400, y - 250)
    k = nb.math('DIVIDE', cam.outputs['View Distance'], HAZE_DISTANCE, x - 250, y - 250)
    e = nb.math('EXPONENT', nb.math('MULTIPLY', k, -1.0, x - 150, y - 250), None, x - 50, y - 250)
    fac = nb.math('SUBTRACT', 1.0, e, x + 50, y - 250, clamp=True)
    em = nb.n('ShaderNodeEmission', x, y - 400)
    em.inputs['Color'].default_value = (*HAZE_COLOR, 1.0)
    em.inputs['Strength'].default_value = HAZE_STRENGTH
    # la niebla no debe iluminar la escena: solo se ve en rayos de cámara
    lp = nb.n('ShaderNodeLightPath', x - 400, y - 550)
    fac = nb.math('MULTIPLY', fac, lp.outputs['Is Camera Ray'], x + 150, y - 300)
    mix = nb.n('ShaderNodeMixShader', x + 250, y)
    nb.l(fac, mix.inputs[0])
    nb.l(shader_out, mix.inputs[1])
    nb.l(em.outputs[0], mix.inputs[2])
    return mix.outputs[0]


# ---------------------------------------------------------------------------
# Piel
# ---------------------------------------------------------------------------

def skin():
    mat = bpy.data.materials.new("TRex_Piel")
    mat.displacement_method = 'BOTH'
    nb = NB(mat)
    tc = nb.n('ShaderNodeTexCoord', -2600, 0)
    P = tc.outputs['Object']
    dorso = nb.attr('dorso', -2400, 600)
    cabeza = nb.attr('cabeza', -2400, 450)
    labio = nb.attr('labio', -2400, 300)
    pl_cuello = nb.attr('pliegue_cuello', -2400, 150)
    pl_pata = nb.attr('pliegue_pata', -2400, 0)
    suciedad = nb.attr('suciedad', -2400, -150)
    sep = nb.n('ShaderNodeSeparateXYZ', -2400, -300)
    nb.l(P, sep.inputs[0])
    zpos = sep.outputs['Z']

    # --- escamas -----------------------------------------------------------
    warp = nb.noise(P, 6.0, 2.0, x=-2200, y=-700)
    pw = nb.n('ShaderNodeVectorMath', -2000, -700, operation='MULTIPLY_ADD')
    pw.inputs[1].default_value = (0.012, 0.012, 0.012)
    nb.l(warp.outputs['Color'], pw.inputs[0])
    nb.l(P, pw.inputs[2])
    Pw = pw.outputs[0]
    # escamas pequeñas (flancos, patas) y grandes (lomo, cabeza)
    v_small = nb.voronoi(Pw, 62.0, 'DISTANCE_TO_EDGE', 0.95, -1800, -600)
    v_small_c = nb.voronoi(Pw, 62.0, 'F1', 0.95, -1800, -800)
    v_big = nb.voronoi(Pw, 21.0, 'DISTANCE_TO_EDGE', 0.9, -1800, -1000)
    v_big_c = nb.voronoi(Pw, 21.0, 'F1', 0.9, -1800, -1200)
    # cabeza: placas irregulares tipo cocodrilo
    v_head = nb.voronoi(Pw, 11.0, 'DISTANCE_TO_EDGE', 1.0, -1800, -1400)
    h_small = nb.maprange(v_small.outputs['Distance'], 0.0, 0.16, smooth=True, x=-1600, y=-600)
    h_big = nb.maprange(v_big.outputs['Distance'], 0.0, 0.10, smooth=True, x=-1600, y=-1000)
    h_head = nb.maprange(v_head.outputs['Distance'], 0.0, 0.05, smooth=True, x=-1600, y=-1400)
    # máscara: escamas grandes en el lomo (y algo de ruido para bordes irregulares)
    big_mask_n = nb.noise(P, 3.0, 3.0, x=-1800, y=-1600)
    big_mask = nb.math('ADD', dorso, nb.mad(big_mask_n.outputs['Fac'], 0.5, -0.25,
                                             -1600, -1600), -1450, -1600)
    big_mask = nb.maprange(big_mask, 0.62, 0.78, smooth=True, x=-1300, y=-1600)
    scales = nb.mixf(big_mask, h_small, h_big, -1300, -800)
    scales = nb.mixf(cabeza, scales, nb.math('MULTIPLY', h_head, h_small, -1450, -1400), -1150, -900)
    # ligera bóveda dentro de cada escama según su color aleatorio
    bw_small = nb.n('ShaderNodeRGBToBW', -1600, -850)
    nb.l(v_small_c.outputs['Color'], bw_small.inputs[0])
    bw_big = nb.n('ShaderNodeRGBToBW', -1600, -1200)
    nb.l(v_big_c.outputs['Color'], bw_big.inputs[0])
    cell_rand = nb.mixf(big_mask, bw_small.outputs[0], bw_big.outputs[0], -1300, -1150)

    # escamas de ornamento: botones dispersos más grandes
    v_feat = nb.voronoi(Pw, 7.0, 'F1', 1.0, -1800, -1800)
    bw_feat = nb.n('ShaderNodeRGBToBW', -1600, -1900)
    nb.l(v_feat.outputs['Color'], bw_feat.inputs[0])
    feat_on = nb.maprange(bw_feat.outputs[0], 0.55, 0.62, smooth=True, x=-1450, y=-1900)
    feat_dome = nb.maprange(v_feat.outputs['Distance'], 0.32, 0.12, smooth=True, x=-1450, y=-1800)
    feat = nb.math('MULTIPLY', feat_on, feat_dome, -1250, -1850)
    feat = nb.math('MULTIPLY', feat, nb.math('SUBTRACT', 1.0, cabeza, -1400, -2000), -1100, -1850)

    # pliegues: anillos en el cuello (bandas en X) y en articulaciones (bandas en Z)
    wave_n = nb.n('ShaderNodeTexWave', -1800, -2200, wave_type='BANDS', bands_direction='X',
                  wave_profile='SIN')
    wave_n.inputs['Scale'].default_value = 1.6
    wave_n.inputs['Distortion'].default_value = 5.0
    wave_n.inputs['Detail'].default_value = 3.0
    wave_n.inputs['Detail Scale'].default_value = 1.5
    nb.l(Pw, wave_n.inputs['Vector'])
    wave_l = nb.n('ShaderNodeTexWave', -1800, -2450, wave_type='BANDS', bands_direction='Z',
                  wave_profile='SIN')
    wave_l.inputs['Scale'].default_value = 2.4
    wave_l.inputs['Distortion'].default_value = 5.0
    wave_l.inputs['Detail'].default_value = 3.0
    nb.l(P, wave_l.inputs['Vector'])
    folds_n = nb.math('MULTIPLY', nb.math('POWER', wave_n.outputs['Fac'], 3.0, -1600, -2200), pl_cuello,
                      -1450, -2200)
    folds_l = nb.math('MULTIPLY', nb.math('POWER', wave_l.outputs['Fac'], 3.0, -1600, -2450), pl_pata,
                      -1450, -2450)
    folds = nb.math('ADD', folds_n, folds_l, -1300, -2300)

    # arrugas finas entre escamas
    fine = nb.noise(P, 140.0, 6.0, 0.6, x=-1800, y=-2700)
    lump = nb.noise(P, 4.0, 4.0, 0.55, x=-1800, y=-2900)

    # relieve: lo grande (ornamentos, pliegues, bultos) desplaza la malla; las escamas y
    # arrugas finas, más pequeñas que la separación entre vértices, van como bump
    Hs = nb.math('MULTIPLY', scales, nb.mad(cell_rand, 0.5, 0.75, -1100, -1000), -950, -900)
    Hs = nb.math('MULTIPLY', Hs, nb.mixf(big_mask, 0.0025, 0.005, -950, -1100), -800, -950)
    Hs = nb.mad(fine.outputs['Fac'], 0.0012, Hs, -650, -950)
    Hd = nb.math('MULTIPLY', feat, 0.006, -800, -1500)
    Hd = nb.mad(folds, -0.018, Hd, -650, -1500)
    Hd = nb.mad(lump.outputs['Fac'], 0.006, Hd, -350, -1500)
    disp = nb.n('ShaderNodeDisplacement', 1100, -600, space='OBJECT')
    disp.inputs['Midlevel'].default_value = 0.0
    disp.inputs['Scale'].default_value = 1.0
    nb.l(Hd, disp.inputs['Height'])
    bump_s = nb.n('ShaderNodeBump', 300, -900)
    bump_s.inputs['Strength'].default_value = 1.0
    bump_s.inputs['Distance'].default_value = 1.0
    nb.l(Hs, bump_s.inputs['Height'])
    bump = nb.n('ShaderNodeBump', 500, -900)
    bump.inputs['Strength'].default_value = 0.2
    bump.inputs['Distance'].default_value = 0.002
    micro = nb.noise(P, 420.0, 3.0, 0.6, x=300, y=-1100)
    nb.l(micro.outputs['Fac'], bump.inputs['Height'])
    nb.l(bump_s.outputs['Normal'], bump.inputs['Normal'])

    # --- color -------------------------------------------------------------
    warp_c = nb.noise(P, 1.2, 5.0, 0.6, x=-1800, y=900)
    shade = nb.mad(warp_c.outputs['Fac'], 0.30, nb.math('SUBTRACT', dorso, 0.15, -1600, 750),
                    -1450, 800)
    base = nb.ramp(shade, [(0.16, (0.25, 0.20, 0.135)), (0.36, (0.135, 0.092, 0.055)),
                           (0.58, (0.066, 0.047, 0.029)), (0.84, (0.026, 0.023, 0.017))], -1300, 800)
    # bandas transversales tenues en lomo y cola (camuflaje disruptivo)
    bands = nb.n('ShaderNodeTexWave', -1800, 1200, wave_type='BANDS', bands_direction='X',
                 wave_profile='SIN')
    bands.inputs['Scale'].default_value = 0.42
    bands.inputs['Distortion'].default_value = 9.0
    bands.inputs['Detail'].default_value = 4.0
    bands.inputs['Detail Roughness'].default_value = 0.6
    nb.l(P, bands.inputs['Vector'])
    band_mask = nb.math('MULTIPLY', nb.maprange(bands.outputs['Fac'], 0.5, 0.8, smooth=True, x=-1600, y=1200),
                        nb.maprange(dorso, 0.38, 0.72, smooth=True, x=-1600, y=1050), -1450, 1150)
    band_mask = nb.math('MULTIPLY', band_mask, nb.math('SUBTRACT', 1.0, cabeza, -1600, 1350), -1300, 1150)
    col = nb.mix(nb.math('MULTIPLY', band_mask, 0.7, -1150, 1150), base, (0.018, 0.015, 0.012), x=-1100, y=900)
    # manchas pequeñas moteadas
    mott = nb.noise(P, 9.0, 3.0, 0.5, x=-1800, y=1450)
    mott_m = nb.maprange(mott.outputs['Fac'], 0.55, 0.7, smooth=True, x=-1600, y=1450)
    col = nb.mix(nb.math('MULTIPLY', mott_m, 0.5, -1300, 1450), col, (0.018, 0.014, 0.011), x=-950, y=1000)
    # manchas claras irregulares en flancos (rompen la silueta)
    blot = nb.noise(P, 1.6, 4.0, 0.6, distortion=1.5, x=-1800, y=1650)
    blot_m = nb.maprange(blot.outputs['Fac'], 0.6, 0.68, smooth=True, x=-1600, y=1650)
    blot_m = nb.math('MULTIPLY', blot_m, nb.maprange(dorso, 0.25, 0.55, smooth=True, x=-1600, y=1800), -1450, 1700)
    col = nb.mix(nb.math('MULTIPLY', blot_m, 0.35, -1300, 1700), col, (0.16, 0.12, 0.08), x=-900, y=1100)
    # cabeza: tono algo más gris y cálido en el hocico; labios claros
    head_col = nb.mix(0.3, col, (0.075, 0.052, 0.038), x=-950, y=600)
    col = nb.mix(cabeza, col, head_col, x=-800, y=800)
    col = nb.mix(nb.math('MULTIPLY', labio, 0.6, -950, 450), col, (0.17, 0.135, 0.10), x=-650, y=800)
    # variación por escama
    var = nb.mad(cell_rand, 0.35, 0.82, -800, 300)
    comb = nb.n('ShaderNodeCombineColor', -650, 300)
    for i in range(3):
        nb.l(var, comb.inputs[i])
    col = nb.mix(1.0, col, comb.outputs[0], 'MULTIPLY', -500, 800)
    # ornamentos algo más claros
    col = nb.mix(nb.math('MULTIPLY', feat, 0.12, -500, 500), col, (0.2, 0.16, 0.12), x=-350, y=800)
    # surcos entre escamas y cavidades más oscuros (polvo y suciedad incrustados)
    groove = nb.math('SUBTRACT', 1.0, scales, -950, 100)
    cav = nb.math('MAXIMUM', nb.math('MULTIPLY', groove, 0.45, -800, 100),
                  nb.math('MULTIPLY', folds, 0.5, -800, 0), -650, 100)
    cav = nb.math('ADD', cav, nb.maprange(suciedad, 0.7, 0.1, 0.0, 0.35, smooth=True, x=-800, y=-100), -500, 100)
    col = nb.mix(nb.math('MINIMUM', cav, 0.7, -350, 100), col, (0.016, 0.012, 0.009), x=-200, y=800)
    # barro seco en patas (sube con salpicaduras) y polvo claro
    mud_n = nb.noise(P, 2.5, 6.0, 0.65, x=-800, y=-300)
    mud_h = nb.mad(mud_n.outputs['Fac'], 0.9, zpos, -650, -300)
    mud = nb.maprange(mud_h, 1.75, 1.05, smooth=True, x=-500, y=-300)
    mud_col = nb.ramp(nb.noise(P, 14.0, 4.0, x=-650, y=-500).outputs['Fac'],
                      [(0.35, (0.10, 0.075, 0.05)), (0.65, (0.23, 0.19, 0.14))], -500, -500)
    col = nb.mix(mud, col, mud_col, x=-50, y=800)

    # cicatrices de peleas en cabeza y cuello: líneas finas claras y algo hundidas
    scar_v = nb.voronoi(Pw, 1.3, 'DISTANCE_TO_EDGE', 1.0, -800, -700)
    scar_l = nb.maprange(scar_v.outputs['Distance'], 0.012, 0.0, smooth=True, x=-650, y=-700)
    scar_n = nb.noise(P, 0.9, 2.0, x=-800, y=-850)
    scar_m = nb.maprange(scar_n.outputs['Fac'], 0.58, 0.66, smooth=True, x=-650, y=-850)
    head_neck = nb.math('MAXIMUM', cabeza, pl_cuello, -650, -1000)
    scar = nb.math('MULTIPLY', nb.math('MULTIPLY', scar_l, scar_m, -500, -750), head_neck, -350, -750)
    col = nb.mix(nb.math('MULTIPLY', scar, 0.7, -250, -750), col, (0.2, 0.14, 0.11), x=0, y=800)
    Hs2 = nb.mad(scar, -0.002, Hs, -250, -950)
    nb.l(Hs2, bump_s.inputs['Height'])

    # --- rugosidad, capa húmeda -------------------------------------------
    rough = nb.mad(scales, -0.1, 0.74, -300, -100)
    rough = nb.mixf(mud, rough, 0.9, -150, -100)
    rough = nb.mixf(labio, rough, 0.32, 0, -100)
    coat = nb.math('MULTIPLY', labio, 0.6, 0, -250)

    p = principled(nb, 600, 400, Base_Color=col, Roughness=rough, Coat_Weight=coat,
                   Coat_Roughness=0.15, Subsurface_Weight=0.05, Subsurface_Scale=0.03,
                   Subsurface_Radius=(1.0, 0.35, 0.18), Specular_IOR_Level=0.45)
    p.inputs['Subsurface Anisotropy'].default_value = 0.4
    nb.l(bump.outputs['Normal'], p.inputs['Normal'])
    nb.l(bump.outputs['Normal'], p.inputs['Coat Normal'])
    nb.output(p.outputs[0], disp.outputs[0])
    return mat


def mouth():
    mat = bpy.data.materials.new("TRex_Boca")
    mat.displacement_method = 'BOTH'
    nb = NB(mat)
    P = nb.n('ShaderNodeTexCoord', -1600, 0).outputs['Object']
    lengua = nb.attr('lengua', -1400, 300)
    ao = nb.n('ShaderNodeAmbientOcclusion', -1200, 500, samples=8, only_local=True)
    ao.inputs['Distance'].default_value = 0.35
    m = nb.noise(P, 18.0, 4.0, 0.6, x=-1400, y=0)
    base = nb.ramp(m.outputs['Fac'], [(0.35, (0.20, 0.035, 0.035)), (0.65, (0.36, 0.085, 0.07))], -1200, 0)
    tongue = nb.ramp(m.outputs['Fac'], [(0.3, (0.32, 0.09, 0.08)), (0.7, (0.46, 0.17, 0.14))], -1200, -200)
    col = nb.mix(lengua, base, tongue, x=-1000, y=0)
    # manchas oscuras (pigmentación) en paladar
    sp = nb.voronoi(P, 30.0, 'F1', 1.0, -1400, -400)
    spm = nb.maprange(sp.outputs['Distance'], 0.25, 0.1, 0.0, 0.5, smooth=True, x=-1200, y=-400)
    col = nb.mix(nb.math('MULTIPLY', spm, nb.math('SUBTRACT', 1.0, lengua, -1200, -550), -1050, -450),
                 col, (0.08, 0.02, 0.03), x=-850, y=0)
    occl = nb.maprange(ao.outputs['AO'], 0.15, 1.0, 0.03, 1.0, x=-1000, y=500)
    col = nb.mix(1.0, col, occl, 'MULTIPLY', -650, 0)
    # rugosidades transversales del paladar y papilas de la lengua
    rug = nb.n('ShaderNodeTexWave', -1400, -700, wave_type='BANDS', bands_direction='X')
    rug.inputs['Scale'].default_value = 9.0
    rug.inputs['Distortion'].default_value = 3.0
    nb.l(P, rug.inputs['Vector'])
    pap = nb.noise(P, 160.0, 2.0, x=-1400, y=-900)
    hgt = nb.mixf(lengua, rug.outputs['Fac'], pap.outputs['Fac'], -1100, -800)
    disp = nb.n('ShaderNodeDisplacement', 300, -500, space='OBJECT')
    disp.inputs['Midlevel'].default_value = 0.5
    disp.inputs['Scale'].default_value = 0.004
    nb.l(hgt, disp.inputs['Height'])
    p = principled(nb, 300, 200, Base_Color=col, Roughness=0.32, Coat_Weight=0.85, Coat_Roughness=0.04,
                   Subsurface_Weight=0.25, Subsurface_Scale=0.02, Subsurface_Radius=(1.0, 0.25, 0.15))
    nb.output(p.outputs[0], disp.outputs[0])
    return mat


def teeth():
    mat = bpy.data.materials.new("TRex_Dientes")
    nb = NB(mat)
    t = nb.attr('largo', -1400, 200)
    pieza = nb.attr('pieza', -1400, 0)
    P = nb.n('ShaderNodeTexCoord', -1400, -300).outputs['Object']
    crown = nb.ramp(t, [(0.0, (0.10, 0.06, 0.03)), (0.18, (0.27, 0.19, 0.10)), (0.45, (0.46, 0.38, 0.25)),
                        (1.0, (0.62, 0.57, 0.46))], -1100, 200)
    stain = nb.ramp(pieza, [(0.0, (1.0, 0.92, 0.8)), (0.5, (1, 1, 1)), (1.0, (0.85, 0.75, 0.55))], -1100, 0)
    col = nb.mix(1.0, crown, stain, 'MULTIPLY', -850, 100)
    streak = nb.noise(P, 90.0, 5.0, 0.6, x=-1100, y=-300)
    col = nb.mix(nb.math('MULTIPLY', streak.outputs['Fac'], 0.35, -900, -300), col, (0.3, 0.22, 0.12),
                 x=-650, y=100)
    bump = nb.n('ShaderNodeBump', -300, -300)
    bump.inputs['Strength'].default_value = 0.15
    bump.inputs['Distance'].default_value = 0.001
    nb.l(streak.outputs['Fac'], bump.inputs['Height'])
    p = principled(nb, 0, 100, Base_Color=col, Roughness=0.3, Coat_Weight=0.7, Coat_Roughness=0.05,
                   Subsurface_Weight=0.2, Subsurface_Scale=0.008, Subsurface_Radius=(1.0, 0.8, 0.5))
    nb.l(bump.outputs['Normal'], p.inputs['Normal'])
    nb.output(p.outputs[0], x=300)
    return mat


def claws():
    mat = bpy.data.materials.new("TRex_Garras")
    nb = NB(mat)
    t = nb.attr('largo', -1200, 200)
    P = nb.n('ShaderNodeTexCoord', -1200, -200).outputs['Object']
    col = nb.ramp(t, [(0.0, (0.035, 0.028, 0.022)), (0.6, (0.06, 0.05, 0.04)), (1.0, (0.16, 0.14, 0.11))],
                  -900, 200)
    dust = nb.noise(P, 20.0, 5.0, x=-900, y=-100)
    dm = nb.maprange(dust.outputs['Fac'], 0.5, 0.75, 0.0, 0.7, smooth=True, x=-700, y=-100)
    col = nb.mix(dm, col, (0.22, 0.18, 0.13), x=-500, y=100)
    grooves = nb.noise(P, 60.0, 4.0, x=-900, y=-400)
    bump = nb.n('ShaderNodeBump', -400, -300)
    bump.inputs['Strength'].default_value = 0.3
    bump.inputs['Distance'].default_value = 0.001
    nb.l(grooves.outputs['Fac'], bump.inputs['Height'])
    rough = nb.mixf(dm, 0.32, 0.8, -500, -150)
    p = principled(nb, 0, 100, Base_Color=col, Roughness=rough, Coat_Weight=0.3, Coat_Roughness=0.2)
    nb.l(bump.outputs['Normal'], p.inputs['Normal'])
    nb.output(p.outputs[0], x=300)
    return mat


def eye():
    """Ojo: iris ámbar fibroso con anillo límbico oscuro y pupila redondeada (vertical)."""
    mat = bpy.data.materials.new("TRex_Ojo")
    nb = NB(mat)
    P = nb.n('ShaderNodeTexCoord', -1600, 0).outputs['Object']
    sep = nb.n('ShaderNodeSeparateXYZ', -1400, 0)
    nb.l(P, sep.inputs[0])
    x, y, z = sep.outputs
    # coordenadas polares en la cara frontal (+Z del objeto = dirección de la mirada)
    r = nb.n('ShaderNodeVectorMath', -1200, 150, operation='LENGTH')
    cmb = nb.n('ShaderNodeCombineXYZ', -1300, 150)
    nb.l(x, cmb.inputs[0])
    nb.l(y, cmb.inputs[1])
    nb.l(cmb.outputs[0], r.inputs[0])
    rad = r.outputs['Value']
    ang = nb.math('ARCTAN2', y, x, -1200, -50)
    pol = nb.n('ShaderNodeCombineXYZ', -1050, 0)
    nb.l(nb.math('MULTIPLY', ang, 6.0, -1100, -50), pol.inputs[0])
    nb.l(nb.math('MULTIPLY', rad, 3.0, -1100, 100), pol.inputs[1])
    fib = nb.noise(pol.outputs[0], 6.0, 6.0, 0.7, x=-900, y=0)
    iris_t = nb.mad(fib.outputs['Fac'], 0.35, rad, -700, 50)
    iris = nb.ramp(iris_t, [(0.25, (0.95, 0.66, 0.16)), (0.55, (0.75, 0.38, 0.05)), (0.8, (0.25, 0.09, 0.015)),
                            (0.95, (0.02, 0.01, 0.005))], -500, 50)
    # pupila: elipse vertical
    px = nb.math('DIVIDE', x, 0.22, -1100, 400)
    py = nb.math('DIVIDE', y, 0.42, -1100, 300)
    pr = nb.math('ADD', nb.math('MULTIPLY', px, px, -950, 400), nb.math('MULTIPLY', py, py, -950, 300), -800, 350)
    pupil = nb.maprange(pr, 0.85, 1.05, 0.0, 1.0, smooth=True, x=-650, y=350)
    comb = nb.n('ShaderNodeCombineColor', -500, 350)
    for i in range(3):
        nb.l(pupil, comb.inputs[i])
    col = nb.mix(1.0, iris, comb.outputs[0], 'MULTIPLY', -300, 150)
    back = nb.maprange(z, -0.2, 0.3, 0.0, 1.0, x=-500, y=-200)
    col = nb.mix(back, (0.01, 0.005, 0.003), col, x=-150, y=150)
    p = principled(nb, 100, 100, Base_Color=col, Roughness=0.45)
    nb.output(p.outputs[0], x=400)
    return mat


def cornea():
    mat = bpy.data.materials.new("TRex_Cornea")
    nb = NB(mat)
    p = principled(nb, 0, 0, Base_Color=(1.0, 1.0, 1.0), Roughness=0.0, IOR=1.376,
                   Transmission_Weight=1.0)
    nb.output(p.outputs[0], x=300)
    return mat


def saliva():
    mat = bpy.data.materials.new("TRex_Saliva")
    nb = NB(mat)
    p = principled(nb, 0, 0, Base_Color=(0.95, 0.92, 0.85), Roughness=0.03, IOR=1.34,
                   Transmission_Weight=0.92)
    nb.output(p.outputs[0], x=300)
    return mat


# ---------------------------------------------------------------------------
# Entorno
# ---------------------------------------------------------------------------

def ground():
    mat = bpy.data.materials.new("Suelo")
    mat.displacement_method = 'BOTH'
    nb = NB(mat)
    P = nb.n('ShaderNodeTexCoord', -2000, 0).outputs['Object']
    huella = nb.attr('huella', -2000, 400)
    hum = nb.attr('humedad', -2000, 550)
    big = nb.noise(P, 0.08, 6.0, 0.6, x=-1800, y=300)
    mid = nb.noise(P, 0.7, 6.0, 0.6, x=-1800, y=100)
    # barro agrietado en zonas húmedas que se secaron
    crk_mask = nb.maprange(big.outputs['Fac'], 0.56, 0.63, smooth=True, x=-1600, y=300)
    crk_mask = nb.math('MAXIMUM', crk_mask, nb.maprange(hum, 0.2, 0.6, smooth=True, x=-1600, y=550), -1450, 400)
    crk = nb.voronoi(P, 2.2, 'DISTANCE_TO_EDGE', 1.0, -1600, -100)
    crk2 = nb.voronoi(P, 7.0, 'DISTANCE_TO_EDGE', 1.0, -1600, -300)
    cracks = nb.math('MINIMUM', crk.outputs['Distance'], nb.math('MULTIPLY', crk2.outputs['Distance'], 2.5, -1450, -300),
                     -1300, -200)
    crack_l = nb.maprange(cracks, 0.0, 0.04, 1.0, 0.0, smooth=True, x=-1150, y=-200)
    crack_l = nb.math('MULTIPLY', crack_l, crk_mask, -1000, -200)
    # guijarros
    peb = nb.voronoi(P, 22.0, 'F1', 1.0, -1600, -550)
    bwp = nb.n('ShaderNodeRGBToBW', -1450, -650)
    nb.l(peb.outputs['Color'], bwp.inputs[0])
    peb_on = nb.maprange(bwp.outputs[0], 0.86, 0.88, x=-1300, y=-650)
    peb_d = nb.maprange(peb.outputs['Distance'], 0.38, 0.15, smooth=True, x=-1300, y=-550)
    pebbles = nb.math('MULTIPLY', peb_on, peb_d, -1150, -600)
    pebbles = nb.math('MULTIPLY', pebbles, nb.math('SUBTRACT', 1.0, crk_mask, -1300, -800), -1000, -600)
    # color
    dirt = nb.ramp(mid.outputs['Fac'], [(0.3, (0.045, 0.034, 0.024)), (0.55, (0.085, 0.064, 0.044)),
                                        (0.75, (0.13, 0.1, 0.07))], -1300, 600)
    mudc = nb.ramp(mid.outputs['Fac'], [(0.3, (0.11, 0.09, 0.07)), (0.7, (0.16, 0.135, 0.105))], -1300, 800)
    col = nb.mix(crk_mask, dirt, mudc, x=-1000, y=600)
    col = nb.mix(nb.maprange(hum, 0.3, 0.9, 0.0, 0.7, x=-1150, y=750), col, (0.032, 0.026, 0.02), x=-850, y=600)
    green_n = nb.noise(P, 0.35, 5.0, 0.6, x=-1300, y=1000)
    green = nb.maprange(green_n.outputs['Fac'], 0.48, 0.64, 0.0, 0.9, smooth=True, x=-1150, y=1000)
    green = nb.math('MULTIPLY', green, nb.math('SUBTRACT', 1.0, crk_mask, -1150, 1150), -1000, 1000)
    gcol = nb.ramp(nb.noise(P, 6.0, 4.0, x=-1150, y=1250).outputs['Fac'],
                   [(0.3, (0.025, 0.032, 0.012)), (0.7, (0.07, 0.072, 0.03))], -1000, 1250)
    col = nb.mix(green, col, gcol, x=-700, y=600)
    col = nb.mix(crack_l, col, (0.03, 0.022, 0.016), x=-550, y=600)
    pcol = nb.ramp(bwp.outputs[0], [(0.86, (0.09, 0.085, 0.075)), (1.0, (0.2, 0.18, 0.15))], -1000, 400)
    col = nb.mix(pebbles, col, pcol, x=-400, y=600)
    col = nb.mix(nb.math('MULTIPLY', huella, 0.5, -700, 300), col, (0.06, 0.045, 0.03), x=-250, y=600)
    # relieve
    H = nb.math('MULTIPLY', crack_l, -0.012, -800, -200)
    H = nb.mad(pebbles, 0.02, H, -650, -300)
    fine = nb.noise(P, 40.0, 8.0, 0.65, x=-900, y=-900)
    H = nb.mad(fine.outputs['Fac'], 0.012, H, -500, -400)
    H = nb.mad(mid.outputs['Fac'], 0.05, H, -350, -400)
    disp = nb.n('ShaderNodeDisplacement', 600, -500, space='OBJECT')
    disp.inputs['Midlevel'].default_value = 0.0
    nb.l(H, disp.inputs['Height'])
    bump = nb.n('ShaderNodeBump', 100, -700)
    bump.inputs['Strength'].default_value = 0.4
    bump.inputs['Distance'].default_value = 0.003
    grain = nb.noise(P, 250.0, 3.0, x=-100, y=-800)
    nb.l(grain.outputs['Fac'], bump.inputs['Height'])
    rough = nb.mixf(crk_mask, 0.92, 0.8, -400, 0)
    rough = nb.mixf(hum, rough, 0.55, -250, 0)
    p = principled(nb, 300, 400, Base_Color=col, Roughness=rough, Specular_IOR_Level=0.35)
    nb.l(bump.outputs['Normal'], p.inputs['Normal'])
    nb.output(with_haze(nb, p.outputs[0], 800, 400), disp.outputs[0], x=1400)
    return mat


def rock():
    mat = bpy.data.materials.new("Roca")
    mat.displacement_method = 'BOTH'
    nb = NB(mat)
    tc = nb.n('ShaderNodeTexCoord', -1600, 0)
    P = tc.outputs['Object']
    n = nb.noise(P, 2.5, 8.0, 0.65, x=-1400, y=200)
    strata = nb.n('ShaderNodeTexWave', -1400, 0, wave_type='BANDS', bands_direction='Z')
    strata.inputs['Scale'].default_value = 3.0
    strata.inputs['Distortion'].default_value = 14.0
    strata.inputs['Detail'].default_value = 6.0
    nb.l(P, strata.inputs['Vector'])
    col = nb.ramp(nb.mad(strata.outputs['Fac'], 0.12, n.outputs['Fac'], -1200, 100),
                  [(0.35, (0.035, 0.032, 0.028)), (0.6, (0.095, 0.088, 0.075)), (0.85, (0.15, 0.14, 0.12))],
                  -1000, 100)
    # líquenes
    li = nb.noise(P, 6.0, 4.0, x=-1400, y=-300)
    lim = nb.maprange(li.outputs['Fac'], 0.62, 0.7, 0.0, 0.8, smooth=True, x=-1200, y=-300)
    col = nb.mix(lim, col, (0.11, 0.11, 0.06), x=-800, y=100)
    # tierra en la base
    sep = nb.n('ShaderNodeSeparateXYZ', -1400, -500)
    nb.l(tc.outputs['Generated'], sep.inputs[0])
    soil = nb.maprange(sep.outputs['Z'], 0.38, 0.15, smooth=True, x=-1200, y=-500)
    col = nb.mix(soil, col, (0.05, 0.04, 0.03), x=-600, y=100)
    vor = nb.voronoi(P, 4.0, 'DISTANCE_TO_EDGE', 1.0, -1400, -700)
    cr = nb.maprange(vor.outputs['Distance'], 0.0, 0.03, -1.0, 0.0, x=-1200, y=-700)
    H = nb.mad(n.outputs['Fac'], 0.06, nb.math('MULTIPLY', cr, 0.02, -1000, -700), -800, -600)
    disp = nb.n('ShaderNodeDisplacement', 300, -400, space='OBJECT')
    disp.inputs['Midlevel'].default_value = 0.0
    nb.l(H, disp.inputs['Height'])
    p = principled(nb, 100, 200, Base_Color=col, Roughness=0.82)
    nb.output(with_haze(nb, p.outputs[0], 500, 200), disp.outputs[0], x=1100)
    return mat


def foliage(name, c1, c2, translucency=0.35):
    mat = bpy.data.materials.new(name)
    nb = NB(mat)
    info = nb.n('ShaderNodeObjectInfo', -1200, 300)
    P = nb.n('ShaderNodeTexCoord', -1200, 0).outputs['Object']
    n = nb.noise(P, 8.0, 3.0, x=-1000, y=0)
    t = nb.mad(info.outputs['Random'], 0.6, n.outputs['Fac'], -800, 100)
    col = nb.ramp(nb.math('MULTIPLY', t, 0.6, -700, 100), [(0.2, c1), (0.8, c2)], -500, 100)
    p = principled(nb, -200, 200, Base_Color=col, Roughness=0.55, Specular_IOR_Level=0.35)
    tr = nb.n('ShaderNodeBsdfTranslucent', -200, -200)
    nb.l(col, tr.inputs['Color'])
    mix = nb.n('ShaderNodeMixShader', 100, 100)
    mix.inputs[0].default_value = translucency
    nb.l(p.outputs[0], mix.inputs[1])
    nb.l(tr.outputs[0], mix.inputs[2])
    nb.output(with_haze(nb, mix.outputs[0], 500, 100), x=1000)
    return mat


def bark():
    mat = bpy.data.materials.new("Corteza")
    nb = NB(mat)
    P = nb.n('ShaderNodeTexCoord', -1000, 0).outputs['Object']
    w = nb.n('ShaderNodeTexWave', -800, 0, wave_type='BANDS', bands_direction='X')
    w.inputs['Scale'].default_value = 6.0
    w.inputs['Distortion'].default_value = 10.0
    nb.l(P, w.inputs['Vector'])
    col = nb.ramp(w.outputs['Fac'], [(0.2, (0.03, 0.022, 0.016)), (0.8, (0.13, 0.095, 0.07))], -600, 0)
    p = principled(nb, -300, 0, Base_Color=col, Roughness=0.9)
    nb.output(with_haze(nb, p.outputs[0], 100, 0), x=600)
    return mat


def hills():
    """Colinas lejanas: vegetación oscura casi disuelta en la niebla."""
    mat = bpy.data.materials.new("Colinas")
    nb = NB(mat)
    P = nb.n('ShaderNodeTexCoord', -1000, 0).outputs['Object']
    n = nb.noise(P, 0.02, 6.0, 0.6, x=-800, y=0)
    col = nb.ramp(n.outputs['Fac'], [(0.35, (0.02, 0.028, 0.014)), (0.65, (0.06, 0.06, 0.035))], -600, 0)
    p = principled(nb, -300, 0, Base_Color=col, Roughness=0.9)
    nb.output(with_haze(nb, p.outputs[0], 100, 0), x=600)
    return mat
