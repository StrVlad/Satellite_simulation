from matplotlib.widgets import CheckButtons, TextBox, Button
import matplotlib.pyplot as plt
import scipy
import numpy as np
from math import cos, sin
import PIL
#Ввод переменных
G = 6.673 * 10 ** (-11)
M = 5.972 * 10 ** 24
radius = 6371000
N = float(input("North coordinate: "))
E = float(input("East coordinate: "))
ISS_height = float(input("Height above the Earth: "))
ISS_velocity = float(input("Velocity: "))
AmRot = int(input("Amount of rotations: "))


def rotz(g):
    return [[cos(g), -sin(g), 0], [sin(g), cos(g), 0], [0, 0, 1]]
#Высчитываем нормаль
def get_orbit_n(r, solution_one):
    phi = 90 * np.pi / 180

    p1 = -r[1] / r[0]
    p2 = -cos(phi) * r[2] / r[0]

    a = p1 ** 2 + 1
    b = 2 * p1 * p2
    c = p2 ** 2 - sin(phi) * sin(phi)

    y1 = (-b + (b ** 2 - 4 * a * c) ** 0.5) / (2 * a)
    y2 = (-b - (b ** 2 - 4 * a * c) ** 0.5) / (2 * a)

    x1 = p1 * y1 + p2
    x2 = p1 * y2 + p2

    z = cos(phi)

    n1 = [x1, y1, z]
    n2 = [x2, y2, z]

    if solution_one:
        return n1
    else:
        return n2

#Высчитываем траекторию спутника
ISS_time = 90 * 60
North = N * np.pi / 180
East = E * np.pi / 180
m = 420000

init_pos = np.array([cos(North) * cos(East), cos(North) * sin(East), sin(North)])
orbit_norm = np.array(get_orbit_n(init_pos, False))
tau = np.cross(orbit_norm, init_pos)

r0 = init_pos * (radius + ISS_height)
v0 = tau * ISS_velocity
x0 = np.concatenate([r0, v0])

tspan = np.linspace(0, AmRot * ISS_time, 10 ** 5)


def odefun(x, t):
    return np.concatenate([x[3:6], (-G * M * x[:3]) / ((np.linalg.norm(x[:3])) ** 3)])
x = scipy.integrate.odeint(odefun, x0, tspan, rtol=1e-13, atol=1e-14)
trajectory = x[:, :3]
velocity = x[:, 3:6]

kinetic_energy = np.zeros(tspan.shape[0])
potential_energy = np.zeros(tspan.shape[0])

trajectory_corrected1 = (np.zeros(len(trajectory)))
trajectory_corrected2 = (np.zeros(len(trajectory)))
trajectory_corrected3 = (np.zeros(len(trajectory)))

for i in range(len(x)):
    current_time = tspan[i]
    angle_erth_rotation = -2 * np.pi * (current_time / (24 * 60 * 60))
    current_point = np.transpose(trajectory[i, :])
    current_point_corrected = current_point.dot(rotz(angle_erth_rotation))

    trajectory_corrected1[i] = list(current_point_corrected)[0]
    trajectory_corrected2[i] = list(current_point_corrected)[1]
    trajectory_corrected3[i] = list(current_point_corrected)[2]

    kinetic_energy[i] = 0.5 * m * np.dot(velocity[i, :], velocity[i, :])
    potential_energy[i] = (-G * M * m) / np.linalg.norm(current_point)

total_energy = potential_energy + kinetic_energy

#Построение земли и траектории спутника
fig = plt.figure("Orbit_Simulation")

ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.plot3D(trajectory_corrected1[:], trajectory_corrected2[:], trajectory_corrected3[:], color='violet', linewidth=2)

pic = PIL.Image.open('earthmap.jpg')
pic = np.array(pic.resize([int(d) for d in pic.size])) / 256.
u = np.linspace(-180, 180, pic.shape[1]) * np.pi / 180
v = np.linspace(-90, 90, pic.shape[0])[::-1] * np.pi / 180
x = 0.7 * radius * np.outer(np.cos(u), np.cos(v)).T
y = 0.7 * radius * np.outer(np.sin(u), np.cos(v)).T
z = 0.7 * radius * np.outer(np.ones(np.size(u)), np.sin(v)).T

Earth = ax.plot_surface(x, y, z, rstride=4, cstride=4, facecolors=pic, visible=True)

ax_checkbox = plt.axes([0.4, 0.7, 0.15, 0.15])
check = CheckButtons(ax_checkbox, ["Earth"], [1])

def Earth_alpha(label):
    Earth.set_visible(not Earth.get_visible())
    plt.draw
check.on_clicked(Earth_alpha)

#Вывод текста с проверками

first_space_velocity = np.sqrt((G * M) / (radius + ISS_height))
if ISS_velocity >= first_space_velocity:
    s_fall = "Satellite doesn't fall"
else:
    s_fall = 'Satellite falls!!'

satellite_dist_from_Earth = np.sqrt(
    trajectory_corrected1[:] ** 2 + trajectory_corrected2[:] ** 2 + trajectory_corrected3[:] ** 2) - radius
count = 0

for i in range(len(satellite_dist_from_Earth)):
    if satellite_dist_from_Earth[i] <= 10000000:
        count += 0
    else:
        count += 1
if count == 0:
    s_atm = "Satellite is in the Earth's atmosphere"
else:
    s_atm = "Satellite is NOT in the Earth's atmosphere"

second_space_velocity = first_space_velocity * 2 ** 0.5

if ISS_velocity >= second_space_velocity:
    s_out = "Satellite leaves the Earth's orbit"
else:
    s_out = "Satellite doesn't leave the Earth's orbit"
plt.figtext(0.4, 0.1, f"{s_fall}\n{s_atm}\n{s_out}")

#Графики энергий
ax.set_zlim(-8e6, 8e6)
ax.set_xlim(-8e6, 8e6)
ax.set_ylim(-8e6, 8e6)
ax = fig.add_subplot(3, 3, 1)
ax1 = fig.add_subplot(3, 3, 4)
ax2 = fig.add_subplot(3, 3, 7)
ax.set_title('Potential')
ax1.set_title('Kinetic')
ax2.set_title('Total')
ax.plot(tspan, potential_energy)
ax1.plot(tspan, kinetic_energy)
ax2.plot(tspan, total_energy)
ax2.set_ylim(-3e13, 3e13)

plt.show()
