print("Cargando bibliotecas...")
import numpy as np
from math import log
from PIL import Image
from matplotlib import colormaps
from time import time, sleep
import sys

t_ini = time()

'''
Número máximo de elementos de la secuencia calculados para probar la convergencia de un candidato.
MÁS ALTO: Más preciso, más lento.
'''
CONVERGENCE_STEPS = 512
'''
Umbral para considerar que cualquier candidato es divergente.
Cuando una secuencia alcanza o supera este valor, su candidato asociado
se considera divergente.
'''
CONVERGENCE_THRESHOLD = 1000

'''
Calcula el siguiente valor de la secuencia de Julia.
Cuando el candidato es 0, la secuencia de Julia es la secuencia de Mandelbrot para
el parámetro.
'''
def julia_sequence(parameter, candidate=0):
    while True:
        yield candidate
        candidate = candidate**2 + parameter

mandelbrot_sequence = lambda z: julia_sequence(z) # Atajo Lambda para Mandelbrot

'''
Prueba si un valor de la secuencia es estable.
'''
def stable(candidate, iters=CONVERGENCE_STEPS):
    if type(candidate) == complex:
        z = 0
    elif type(candidate) == np.ndarray:
        z = np.zeros_like(candidate)
    seq = julia_sequence(candidate, z)
    for _ in range(iters):
        res = next(seq)
    return abs(res) <= 2

'''
Devuelve la estabilidad de un valor en la secuencia.
La estabilidad es igual al número de iteraciones (candidatos calculados)
hasta que se alcanza la divergencia.
'''
def stability(candidate, iters=CONVERGENCE_STEPS, smooth=True):
    z = 0
    seq = julia_sequence(candidate, z)
    for iter in range(iters):
        res = next(seq)
        if abs(res) > CONVERGENCE_THRESHOLD:
            if smooth: # Suaviza los bordes
                return (iter + 1 - log(log(abs(res))) / log(2)) / iters
            return iter / iters
    return 1
        
'''
Genera una matriz 2D de números complejos equidistantes en el rectángulo especificado.
'''
def complex_matrix(elements, center=(0, 0), scale=4):
    xcenter, ycenter = center
    yrang = xrang = scale / 2
    re = np.linspace(xcenter - xrang, xcenter + xrang, elements)
    im = np.linspace(ycenter - yrang, ycenter + yrang, elements)
    return re[:, np.newaxis] + im[np.newaxis, :] * 1j

# Paleta de colores para la imagen
cm_name = "ocean"
palette = [tuple(int(round(channel * 255)) for channel in color) for color in [colormaps[cm_name](c) for c in np.linspace(0, 1, CONVERGENCE_STEPS)]][::-1]

# Ancho y alto de la imagen
width = height = 480

print("Bibliotecas cargadas\n\nGenerando imagen...")


# 1. GENERACIÓN DE LOS VALORES A CALCULAR

t1 = time()
mandelbrot_matrix = complex_matrix(width, center=(-0.7435, 0.1314), scale=0.002)
t2 = time()
time_generation = t2 - t1


# 2. CÁLCULO DE LOS VALORES

# Generar la forma de la imagen:
pixels = np.empty((*mandelbrot_matrix.transpose().shape, 4), dtype=np.uint8)

# Inicializar variables del bucle de cálculo:
last_progress = 0
print("Progreso:   |" + "·" * 100 + "|   0%", end="")
# Bucle de cálculo:
for y in range(height):
    for x in range(width):
        pixels[y, x] = palette[int((len(palette) - 1) * stability(mandelbrot_matrix[x, -((y + 1) % (height))]))]

    # Actualizar progreso
    if int(y / height * 100) > last_progress:
        last_progress = int(y / height * 100)
        prog = last_progress
        spaces = 2 if prog < 10 else 1
        print("\rProgreso:   |" + "█" * prog + "·" * (100 - prog) + f"| {' ' * spaces}{prog}%", end="")
        sys.stdout.flush()
print("\rProgreso:   |" + "█" * 100 + "| 100%")
sys.stdout.flush()
t3 = time()
time_computation = t3 - t2


# 4. ENSAMBLAR Y MOSTRAR LA IMAGEN FINAL

imagen = Image.fromarray(pixels)

t_end = time()

sleep(1) # Para efecto dramático
print(f"\nTIEMPO TOTAL DE EJECUCIÓN: {t_end - t_ini}s\n")



# Print execution time statistics:
print(f"Generation step:\n\ttime:\t{time_generation}s")
print(f"Computation step:\n\ttime:\t{time_computation}s")

imagen.show()
#image.save(f"{cm_name}{width}.png")

