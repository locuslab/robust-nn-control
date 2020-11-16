import numpy as np
import cvxpy as cp
import scipy.linalg as la
from constants import *

import warnings

class RobustMPC():
    def __init__(self):
        pass

    def get_action(self, x):
        Ks_new = np.apply_along_axis(self.controller_gain_fn, 1, x.cpu().detach().numpy())
        nan_mask = np.isnan(Ks_new).sum(axis=1).sum(axis=1) > 0
        
        # Accommodate any solver errors by falling back on robust LQR
        Ks_t = torch.tensor(Ks_new, device=self.device, dtype=TORCH_DTYPE)
        Ks_t[nan_mask] = self.K_init

        u = Ks_t.bmm(x.unsqueeze(-1)).squeeze(-1)
        return u

class RobustNLDIMPC(RobustMPC):
    def __init__(self, A, B, G, C, D, Q, R, K_init, device='cpu'):
        super().__init__()
        Q_sqrt = la.sqrtm(Q.cpu().detach().numpy())
        R_sqrt = la.sqrtm(R.cpu().detach().numpy())
        self.controller_gain_fn = lambda x: get_nldi_controller_gain(
            x, *[v.cpu().detach().numpy() for v in (A, B, G, C, D)], Q_sqrt, R_sqrt)

        self.K_init = K_init
        self.device = device

class RobustPLDIMPC(RobustMPC):
    def __init__(self, As, Bs, Q, R, K_init, device='cpu'):
        super().__init__()
        Q_sqrt = la.sqrtm(Q.cpu().detach().numpy())
        R_sqrt = la.sqrtm(R.cpu().detach().numpy())
        self.controller_gain_fn = lambda x: get_pldi_controller_gain(
            x, As.cpu().detach().numpy(), Bs.cpu().detach().numpy(), Q_sqrt, R_sqrt)

        self.K_init = K_init
        self.device = device

def get_nldi_controller_gain(x_in, A, B, G, C, D, Q_sqrt, R_sqrt):
    n = A.shape[1]
    m = B.shape[1]
    w = G.shape[1]
    wq = C.shape[0]
    assert (w <= wq), "wp must equal wq to use this method"
    if w < wq:
        G = np.concatenate([G, np.zeros([G.shape[0], wq - w])], axis=1)
        w = wq
    x = np.expand_dims(x_in, 1)

    S = cp.Variable((n,n), symmetric=True)
    Y = cp.Variable((m,n))
    gam = cp.Variable()
    lam = cp.Variable(w)

    m1 = cp.bmat((
        (np.expand_dims(np.array([1]),0), x.T),
        (x, S)
    ))

    m2 = cp.bmat((
        (S, Y.T@R_sqrt, S@Q_sqrt, S@C.T + Y.T@D.T, S@A.T + Y.T@B.T),
        (R_sqrt@Y, gam*np.eye(m), np.zeros((m,n)), np.zeros((m,w)), np.zeros((m,n))),
        (Q_sqrt@S, np.zeros((n,m)), gam*np.eye(n), np.zeros((n,w)), np.zeros((n,n))),
        (C@S + D@Y, np.zeros((w,m)), np.zeros((w,n)), cp.diag(lam), np.zeros((w,n))),
        (A@S + B@Y, np.zeros((n,m)), np.zeros((n,n)), np.zeros((n,w)), S - G@cp.diag(lam)@G.T)
    ))

    cons = [S >> 0, lam >= 1e-2, m1 >> 0, m2 >> 0]

    try:
        prob = cp.Problem(cp.Minimize(gam), cons)
        prob.solve(solver=cp.MOSEK)
        if prob.status in ["infeasible", "unbounded"]:
            warnings.warn('Infeasible or unbounded SDP for some x. Falling back to K_init for that x.')
            K = np.nan * np.ones((m,n))
        else:
            K = np.linalg.solve(S.value, Y.value.T).T
    except cp.SolverError:
        warnings.warn('Solver error for some x. Falling back to K_init for that x.')
        K = np.nan * np.ones((m,n))

    return K


def get_pldi_controller_gain(x_in, As, Bs, Q_sqrt, R_sqrt):
    n = As.shape[2]
    m = Bs.shape[2]
    L = As.shape[0]

    x = np.expand_dims(x_in, 1)

    S = cp.Variable((n,n), symmetric=True)
    Y = cp.Variable((m,n))
    gam = cp.Variable()

    m1 = cp.bmat((
        (np.expand_dims(np.array([1]),0), x.T),
        (x, S)
    ))

    m2s = [cp.bmat((
            (S, S@As[i].T + Y.T@Bs[i].T, S@Q_sqrt, Y.T@R_sqrt),
            (As[i]@S + Bs[i]@Y, S, np.zeros((n,n)), np.zeros((n,m))),
            (Q_sqrt@S, np.zeros((n,n)), gam*np.eye(n), np.zeros((n,m))),
            (R_sqrt@Y, np.zeros((m,n)), np.zeros((m,n)), gam*np.eye(m))
            )) for i in range(L)]

    cons = [S >> 0, m1 >> 0] + [m2 >> 0 for m2 in m2s]

    try:
        prob = cp.Problem(cp.Minimize(gam), cons)
        prob.solve(solver=cp.MOSEK)
        if prob.status in ["infeasible", "unbounded"]:
            warnings.warn('Infeasible or unbounded SDP for some x. Falling back to K_init for that x.')
            K = np.nan * np.ones((m,n))
        else:
            K = np.linalg.solve(S.value, Y.value.T).T
    except cp.SolverError:
        warnings.warn('Solver error for some x. Falling back to K_init for that x.')
        K = np.nan * np.ones((m,n))

    return K