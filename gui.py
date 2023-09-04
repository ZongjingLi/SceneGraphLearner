from dataclasses import dataclass
from config import *
from models import *
from datasets import *
import matplotlib.pyplot as plt

device = "cuda:0" if torch.cuda.is_available() else "cpu"

import taichi as ti
ti.init(arch = ti.cpu)


# [Claim some Basic Types]
vec3d = ti.types.vector(3, ti.f64)
vec4d = ti.types.vector(4, ti.f64)
mat4x3i = ti.types.matrix(4,3,dtype = ti.f64)

@ti.func
def inv_square(x):return 1.0 / (x * x)


@ti.func
def norm(w: vec4d):
    return w.norm()

@ti.kernel
def test(v: vec4d) -> ti.types.f64:
    outputs = norm(v)
    return outputs

Sphere = ti.types.struct(center= vec3d, radius = float)


@ti.func
def sphere_volume(s: Sphere):
    return s.radius  ** 3

@ti.kernel
def evaluate_run(s: Sphere) -> float:
    return sphere_volume(s)

v = vec4d(1,2,0,3)
v_norm = test(v)
print("v_norm:",v_norm)

center = vec3d(1,0,1)
radius = 2.0
sphere1 = Sphere(center, radius)
print(sphere1)
print(evaluate_run(sphere1))

W, H = (640,640)
random_image = (torch.randn(W,H,3) ** 2).clamp(0.0,1.0).numpy()


main_gui = ti.GUI("Main GUI", (W,H) )
while main_gui.running:
    main_gui.set_image(random_image)
    main_gui.show()