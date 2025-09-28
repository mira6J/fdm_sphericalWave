import numpy as np
import matplotlib.pyplot as plt

gamma = 1.4
R = 1.0
N = 500
r_edges = np.linspace(0, R, N+1)
r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
dr = r_edges[1:] - r_edges[:-1]
A = 4 * np.pi * r_edges**2
V = (4/3) * np.pi * (r_edges[1:]**3 - r_edges[:-1]**3)

def initial_conditions(r):
    p = np.where(r < 0.05, 1e6, 1e5)
    rho = np.where(r < 0.05, 5.0, 1.0)
    u = np.zeros_like(r)
    return rho, u, p

def get_primitive(U):
    rho = U[:,0]
    u = U[:,1] / np.clip(rho, 1e-8, None)
    p = (gamma-1)*(U[:,2] - 0.5*rho*u**2)
    return rho, u, p

def spherical_source(U, r):
    rho, u, p = get_primitive(U)
    S = np.zeros_like(U)
    S[:,1] = np.where(r > 1e-8, 2*p/r, 0)
    return S

def apply_boundary(U):
    U[0,1] = 0.0
    U[-1,:] = U[-2,:]
    return U

def minmod(a, b):
    result = np.zeros_like(a)
    mask = (np.sign(a) == np.sign(b))
    result[mask] = np.where(np.abs(a[mask]) < np.abs(b[mask]), a[mask], b[mask])
    return result

def hllc_flux(UL, UR):
    rhoL, uL, pL = get_primitive(UL)
    rhoR, uR, pR = get_primitive(UR)
    EL = UL[:,2]
    ER = UR[:,2]
    cL = np.sqrt(np.maximum(gamma * pL / np.clip(rhoL, 1e-8, None), 1e-8))
    cR = np.sqrt(np.maximum(gamma * pR / np.clip(rhoR, 1e-8, None), 1e-8))
    SL = np.minimum(uL - cL, uR - cR)
    SR = np.maximum(uL + cL, uR + cR)
    S_star = (pR - pL + rhoL*uL*(SL-uL) - rhoR*uR*(SR-uR)) / (rhoL*(SL-uL) - rhoR*(SR-uR) + 1e-8)

    FL = np.zeros_like(UL)
    FR = np.zeros_like(UR)
    FL[:,0] = rhoL*uL
    FL[:,1] = rhoL*uL**2 + pL
    FL[:,2] = uL*(EL + pL)
    FR[:,0] = rhoR*uR
    FR[:,1] = rhoR*uR**2 + pR
    FR[:,2] = uR*(ER + pR)

    flux = np.zeros_like(UL)
    for i in range(len(UL)):
        if SL[i] >= 0:
            flux[i] = FL[i]
        elif S_star[i] >= 0:
            rho_star = rhoL[i] * (SL[i]-uL[i]) / (SL[i]-S_star[i]+1e-8)
            mom_star = rho_star * S_star[i]
            E_star = rho_star * (EL[i]/rhoL[i] + (S_star[i]-uL[i])*(S_star[i]+pL[i]/(rhoL[i]*(SL[i]-uL[i])+1e-8)))
            U_star_L = np.array([rho_star, mom_star, E_star])
            flux[i] = FL[i] + SL[i]*(U_star_L - UL[i])
        elif SR[i] > 0:
            rho_star = rhoR[i] * (SR[i]-uR[i]) / (SR[i]-S_star[i]+1e-8)
            mom_star = rho_star * S_star[i]
            E_star = rho_star * (ER[i]/rhoR[i] + (S_star[i]-uR[i])*(S_star[i]+pR[i]/(rhoR[i]*(SR[i]-uR[i])+1e-8)))
            U_star_R = np.array([rho_star, mom_star, E_star])
            flux[i] = FR[i] + SR[i]*(U_star_R - UR[i])
        else:
            flux[i] = FR[i]
    return flux

rho, u, p = initial_conditions(r_centers)
e = p / ((gamma - 1) * rho)
U = np.zeros((N, 3))
U[:,0] = rho
U[:,1] = rho * u
U[:,2] = rho * (e + 0.5 * u**2)

output_radius = [0.1, 0.5, 0.9]
pressure_history = {radius: [] for radius in output_radius}

T_end = 0.001
CFL = 0.8
t = 0
dt_list = []
while t < T_end:
    dU = U[1:] - U[:-1]
    dU_left = np.vstack((np.zeros((1,3)), dU))
    dU_right = np.vstack((dU, np.zeros((1,3))))
    slope = minmod(dU_left, dU_right)
    UL = U - 0.5*slope
    UR = U + 0.5*slope

    UL_face = UL[:-1]
    UR_face = UR[1:]
    F_edge = hllc_flux(UL_face, UR_face)
    F_edge = np.vstack((F_edge[0], F_edge, F_edge[-1]))

    rho, u, p = get_primitive(U)
    c = np.sqrt(np.maximum(gamma * p / np.clip(rho, 1e-8, None), 1e-8))
    max_speed = np.max(np.abs(u) + c)
    dt = CFL * np.min(dr) / (max_speed + 1e-8)
    if t + dt > T_end:
        dt = T_end - t
    dt_list.append(dt)

    U_new = U.copy()
    for i in range(N):
        F_R = F_edge[i+1]
        F_L = F_edge[i]
        U_new[i] += -dt / V[i] * (A[i+1]*F_R - A[i]*F_L) + spherical_source(U[i:i+1], r_centers[i])[0] * dt
    U_new = apply_boundary(U_new)
    U_new[:,0] = np.clip(U_new[:,0], 1e-8, 1e8)
    U_new[:,2] = np.clip(U_new[:,2], 1e-8, 1e12)
    U = U_new

    for radius in output_radius:
        idx = np.argmin(np.abs(r_centers - radius))
        p_hist = (gamma - 1) * (U[idx, 2] - 0.5 * U[idx, 1]**2 / U[idx, 0])
        pressure_history[radius].append(p_hist)
    t += dt

rho, u, p = get_primitive(U)

# Plot pressure (in MPa) vs radius
plt.figure()
plt.plot(r_centers, p/1e6)
plt.xlabel('Radius [m]')
plt.ylabel('Pressure [MPa]')
plt.title('Pressure Distribution at t = {:.3f} ms'.format(t*1e3))
plt.grid()
plt.tight_layout()
plt.show()

# Plot density vs radius
plt.figure()
plt.plot(r_centers, rho)
plt.xlabel('Radius [m]')
plt.ylabel('Density [kg/m³]')
plt.title('Density Distribution at t = {:.3f} ms'.format(t*1e3))
plt.grid()
plt.tight_layout()
plt.show()

# Plot pressure history (in MPa) at selected radii, time in ms
plt.figure()
for radius in output_radius:
    time_ms = np.cumsum(dt_list) * 1e3
    plt.plot(time_ms, np.array(pressure_history[radius])/1e6, label=f'r={radius:.2f}m')
plt.xlabel('Time [ms]')
plt.ylabel('Pressure [MPa]')
plt.legend()
plt.title('Pressure History at Selected Radii')
plt.grid()
plt.tight_layout()
plt.show()
