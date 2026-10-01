# T. rex procedural (Blender 5.x + Cycles)

```bash
blender -b --python generar_dinosaurio.py                                   # solo genera dinosaurio.blend
blender -b --python generar_dinosaurio.py -- --render salida.png --pct 50 --samples 32
#   --camara principal|retrato|lateral   --voxel 0.016 (más fino = más detalle y más RAM)
#   --sin-entorno   --no-guardar
```

Estructura:

| Archivo | Contenido |
|---|---|
| `generar_dinosaurio.py` | Orquestación y argumentos |
| `trex/sdf.py` | Campos de distancia en numpy, surface nets, relajación |
| `trex/anatomia.py` | Proporciones, pose, cráneo y mandíbula, boca, dientes, regiones de la piel |
| `trex/materiales.py` | Piel, boca, dientes, garras, ojo, suelo, rocas, vegetación |
| `trex/escenario.py` | Terreno con huellas, vegetación (Geometry Nodes), bosque, colinas, cielo, luz |
| `trex/rig.py` | Esqueleto con pesos calculados por distancia |
| `version_anterior/` | Proyecto original |

Memoria: con 8 GB de RAM, el cuerpo usa desplazamiento por vértice (sin subdivisión adaptativa).
Un render a 1080p ocupa unos 2,5–3 GB.
