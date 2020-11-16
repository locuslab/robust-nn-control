
import numpy as np
import cvxpy as cp
import scipy.optimize as sco
import scipy.linalg as sla
import itertools
import torch


def quadrotor_jacobian(x):
    px, pz, phi, vx, vz, phidot = x
    g = 9.81

    jac = np.array([
            [0, 0, -vx*np.sin(phi)-vz*np.cos(phi), np.cos(phi), -np.sin(phi), 0],
            [0, 0, vx*np.cos(phi)-vz*np.sin(phi), np.sin(phi), np.cos(phi), 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, -g*np.cos(phi), 0, phidot, vz],
            [0, 0, g*np.sin(phi), -phidot, 0, -vx],
            [0, 0, 0, 0, 0, 0]
        ])

    return jac

def calc_max_jac(x_min, x_max):
    px_max, pz_max, phi_max, vx_max, vz_max, phidot_max = x_max
    px_min, pz_min, phi_min, vx_min, vz_min, phidot_min = x_min
    (sinphi_max, cosphi_max), (sinphi_min, cosphi_min) = get_sinusoid_extrema(phi_min, phi_max)

    g = 9.81

    jac = np.array([
            [0, 0, get_max_value(lambda x: -x[3]*np.sin(x[2])-x[4]*np.cos(x[2]), x_min, x_max), cosphi_max, -sinphi_min, 0],
            [0, 0, get_max_value(lambda x: x[3]*np.cos(x[2])-x[4]*np.sin(x[2]), x_min, x_max), sinphi_max, cosphi_max, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, -g*cosphi_min, 0, phidot_max, vz_max],
            [0, 0, g*sinphi_max, -phidot_min, 0, -vx_min],
            [0, 0, 0, 0, 0, 0]
        ])    

    return jac

def calc_min_jac(x_min, x_max):
    px_max, pz_max, phi_max, vx_max, vz_max, phidot_max = x_max
    px_min, pz_min, phi_min, vx_min, vz_min, phidot_min = x_min
    (sinphi_max, cosphi_max), (sinphi_min, cosphi_min) = get_sinusoid_extrema(phi_min, phi_max)

    g = 9.81

    jac = np.array([
            [0, 0, get_min_value(lambda x: -x[3]*np.sin(x[2])-x[4]*np.cos(x[2]), x_min, x_max), cosphi_min, -sinphi_max, 0],
            [0, 0, get_min_value(lambda x: x[3]*np.cos(x[2])-x[4]*np.sin(x[2]), x_min, x_max), sinphi_min, cosphi_min, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, -g*cosphi_max, 0, phidot_min, vz_min],
            [0, 0, g*sinphi_min, -phidot_max, 0, -vx_max],
            [0, 0, 0, 0, 0, 0]
    ]) 

    return jac


def is_in(val, v_min, v_max):
    return (val >= v_min) and (val <= v_max)

def get_max_value(fun, x_min, x_max):
    res = sco.minimize(lambda x: -fun(x), (x_max + x_min)/2, 
        constraints=(
            {'type': 'ineq', 'fun': lambda x: x - x_min},
            {'type': 'ineq', 'fun': lambda x: -x + x_max}))
    return -res['fun']

def get_min_value(fun, x_min, x_max):
    res = sco.minimize(fun, (x_max + x_min)/2, 
        constraints=(
            {'type': 'ineq', 'fun': lambda x: x - x_min},
            {'type': 'ineq', 'fun': lambda x: -x + x_max}))
    return res['fun']

def get_sinusoid_extrema(phi_min, phi_max):
    cosphi_max = 1 if is_in(0, phi_min, phi_max) \
        else max(np.cos(phi_min), np.cos(phi_max))
    cosphi_min = -1 if is_in(np.pi, phi_min, phi_max) or  is_in(-np.pi, phi_min, phi_max) \
        else min(np.cos(phi_min), np.cos(phi_max))
    sinphi_max = 1 if is_in(np.pi/2, phi_min, phi_max) \
        else max(np.sin(phi_min), np.sin(phi_max))
    sinphi_min = -1 if is_in(-np.pi/2, phi_min, phi_max) \
        else min(np.sin(phi_min), np.sin(phi_max))

    return (sinphi_max, cosphi_max), (sinphi_min, cosphi_min)


def xdot_uncontrolled(x):
    px, pz, phi, vx, vz, phidot = [x[:,i] for i in range(x.shape[1])]
    g = 9.81

    x_part = torch.stack([
        vx*torch.cos(phi) - vz*torch.sin(phi), 
        vx*torch.sin(phi) + vz*torch.cos(phi),
        phidot,
        vz*phidot - g*torch.sin(phi),
        -vx*phidot - g*torch.cos(phi) + g,
        torch.zeros(x.shape[0])
    ]).T

    return x_part.numpy()


def main():
    n = 6
    x_max = np.array([6, 6, np.pi/16, 0.25, 0.25, np.pi/32])
    x_min = np.array([-6, -6, -np.pi/16, -0.25, -0.25, -np.pi/32])

    max_jac = calc_max_jac(x_min, x_max)
    min_jac = calc_min_jac(x_min, x_max)

    print('constructing polytope')
    # construct polytope
    non_const = (max_jac != min_jac)
    # Aks = iter([np.array(p) for p in itertools.product(*zip(max_jac[non_const],min_jac[non_const]))])
    Aks_nonconst = [np.array(p) for p in itertools.product(*zip(max_jac[non_const],min_jac[non_const]))]
    Aks = [max_jac.copy() for i in range(len(Aks_nonconst))]
    for i in range(len(Aks)):
        np.putmask(Aks[i], non_const, Aks_nonconst[i])

    print('constructing problem')
    V = cp.Variable((n,n), symmetric=True)
    W = cp.Variable((n,n), symmetric=True)
    A = quadrotor_jacobian(np.zeros(n))

    obj = cp.trace(V) + cp.trace(W)
    cons = [cp.bmat([[V, (Ak-A).T], 
                      [Ak-A, W]]) >> 0 \
            for Ak in Aks]
    cons += [W >> 0]

    prob = cp.Problem(cp.Minimize(obj), cons)

    print('solving SDP')
    prob.solve(solver=cp.MOSEK, verbose=True)

    # TODO: figure out best way to do this
    C = np.linalg.cholesky((V.value).T)
    G = np.linalg.cholesky(W.value)
    # C = sla.sqrtm(V.value)
    # G = sla.sqrtm(W.value)

    # Check correctness
    prop = np.random.random((200, n))
    rand_xs = x_max*prop + x_min*(1-prop)
    fx = xdot_uncontrolled(torch.Tensor(rand_xs))
    # print(np.linalg.norm((fx - rand_xs@A.T)@np.linalg.inv(G).T, axis=1) <= np.linalg.norm(rand_xs@C.T, axis=1))
    print((np.linalg.norm((fx - rand_xs@A.T)@np.linalg.inv(G).T, axis=1) <= np.linalg.norm(rand_xs@C.T, axis=1)).all())

    ratio = np.linalg.norm(rand_xs@C.T, axis=1)/np.linalg.norm((fx - rand_xs@A.T)@np.linalg.inv(G).T, axis=1)
    print(ratio.max())
    print(ratio.mean())
    print(np.median(ratio))

    # Save matrices
    np.save('A.npy', A)
    np.save('G.npy', G)
    np.save('C.npy', C)

if __name__ == '__main__':
    main()

