import torch
import torch.nn as nn
from torch.autograd import Function
import operator
from functools import reduce
import cvxpy as cp
from qpth.qp import QPFunction
from cvxpylayers.torch import CvxpyLayer
from sqrtm import sqrtm
import warnings

from constants import *


class MBPPolicy(nn.Module):
    def __init__(self, K, n, m):
        super().__init__()
        self.K = K

        layer_sizes = [n, 200, 200]
        layers = reduce(operator.add,
                        [[nn.Linear(a, b), nn.ReLU()]
                         for a, b in zip(layer_sizes[0:-1], layer_sizes[1:])])
        layers += [nn.Linear(layer_sizes[-1], m)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return x @ self.K.T + self.net(x)


class StableNLDIProjection:
    def __init__(self, P, A, B, G, C, D, alpha, isD0=False):
        self.P = P
        self.A = A
        self.B = B
        self.G = G
        self.C = C
        self.D = D
        self.alpha = alpha
        self.isD0 = isD0
        self.proj_layer = None

        if self.isD0:
            epsilon = 10e-5
            self.proj_layer = lambda u, g, h: u - nn.ReLU()(torch.div((u * g).sum(-1) - h, (g * g).sum(-1))).unsqueeze(
                1) * g - epsilon * torch.sign(g)
        else:
            self.proj_layer = SOCProjFast()

    def project_action(self, u, x):
        if self.isD0:
            Px = x @ self.P
            g = 2 * Px @ self.B
            neg_h = self.alpha * (Px * x).sum(-1) + \
                    2 * torch.norm(Px @ self.G, dim=1) * torch.norm(x @ self.C.T, dim=1) + \
                    2 * (Px @ self.A * x).sum(-1)
            u = self.proj_layer(u, g, -neg_h)
        else:
            Px = x @ self.P
            const = torch.norm(x @ self.P @ self.G, dim=1)
            A = self.D.expand(x.shape[0], self.D.shape[0], self.D.shape[1])
            b = x @ self.C.T
            c = (1 / const).unsqueeze(1) * (-Px @ self.B)
            d = -((2 * Px @ self.A + self.alpha * Px) * x).sum(-1) / (2 * const)

            u = self.proj_layer(u, A, b, c, d)

        return u

    def __getstate__(self):
        state = [self.P, self.A, self.B, self.G, self.C, self.D, self.alpha, self.isD0]
        return state

    def __setstate__(self, state):
        self.__init__(*state)


class StablePLDIProjection:
    def __init__(self, P, A, B):
        self.P = P
        self.A = A
        self.B = B

        self.e = torch.DoubleTensor().to(device=A.device)

    def project_action(self, u, x):
        Px = x @ self.P
        G = 2 * Px.expand(self.B.shape[0], Px.shape[0], Px.shape[1]).bmm(self.B).transpose(0, 1)
        h = (Px * x).sum(-1).unsqueeze(1) + \
            2 * Px.expand(self.B.shape[0], Px.shape[0], Px.shape[1]).bmm(self.A).transpose(0, 1).matmul(
            x.unsqueeze(2)).squeeze(2)

        Q = torch.eye(u.shape[-1], device=x.device).unsqueeze(0).expand(u.shape[0], u.shape[-1], u.shape[-1])
        res = QPFunction(verbose=-1)(Q.double(), -u.double(), G.double(), -h.double(), self.e, self.e)
        return res.type(TORCH_DTYPE) 

    def __getstate__(self):
        state = [self.P, self.A, self.B]
        return state

    def __setstate__(self, state):
        self.__init__(*state)


class StableHinfProjection:
    def __init__(self, P, A, B, G, Q, R, alpha, gamma, sigma):
        self.P = P
        self.A = A
        self.B = B
        self.G = G
        self.Q = Q
        self.R = R
        self.alpha = alpha
        self.gamma = gamma
        self.sigma = sigma

    def project_action(self, u, xin):
        x = xin.unsqueeze(-1)

        Atilde = sqrtm(self.sigma*self.R)/torch.sqrt(x.transpose(1,2)@(
            self.P@self.B@torch.inverse(self.R)@self.B.T@self.P/self.sigma - \
            self.P@self.A - self.A.T@self.P - self.alpha*self.P - self.sigma * self.Q - \
            self.P@self.G@self.G.T@self.P/(self.sigma*(self.gamma**2)))@x)
        btilde = Atilde@torch.inverse(self.R)@self.B.T@self.P@x/self.sigma

        ctilde = torch.zeros(Atilde.shape[0], Atilde.shape[2], device=xin.device, dtype=TORCH_DTYPE)
        dtilde = torch.ones(btilde.shape[0], device=xin.device, dtype=TORCH_DTYPE)

        u = SOCProjFast(momentum=False)(u, Atilde, btilde.squeeze(-1), ctilde, dtilde)

        return u

    def __getstate__(self):
        state = [self.P, self.A, self.B, self.G, self.Q, self.R, self.alpha, self.gamma]
        return state

    def __setstate__(self, state):
        self.__init__(*state)


class StablePolicy(nn.Module):
    def __init__(self, pi, stable_projection):
        super().__init__()
        self.pi = pi
        self.stable_projection = stable_projection

    def forward(self, x):
        u = self.pi(x)
        u = self.stable_projection.project_action(u, x)
        return u


# From https://github.com/locuslab/qpth/blob/master/qpth/util.py
def bger(x, y):
    """Batch outer product"""
    return x.unsqueeze(2).bmm(y.unsqueeze(1))


def SOCProj(tol=1e-5, max_iters=1000000, rho=10):
    """Projection onto a second order cone constraint"""

    class SOCProjFn(Function):

        @staticmethod
        def forward(ctx, pi, A, b, c, d):
            G = torch.cat([A, c.unsqueeze(1)], dim=1)
            h = torch.cat([b, d.unsqueeze(-1)], dim=1)

            xkm1 = pi
            zkm1 = G.bmm(xkm1.unsqueeze(-1)).squeeze() + h
            ukm1 = torch.zeros_like(zkm1, device=zkm1.device, dtype=TORCH_DTYPE)

            # precompute inversion matrix for x update
            inv_mat = torch.inverse(
                torch.eye(pi.shape[-1], device=zkm1.device, dtype=TORCH_DTYPE).unsqueeze(0).expand(pi.shape[0], pi.shape[-1], pi.shape[-1]) + \
                rho * G.transpose(1, 2).bmm(G))

            for i in range(max_iters):
                xk = inv_mat.bmm(
                    (pi.unsqueeze(-1) - G.transpose(1, 2).bmm((ukm1 - rho * zkm1 + rho * h).unsqueeze(-1)))).squeeze(-1)
                zk = SOCProjFn.proj_normcone(G.bmm(xk.unsqueeze(-1)).squeeze(-1) + h + ukm1 / rho)
                uk = ukm1 + rho * (G.bmm(xk.unsqueeze(-1)).squeeze(-1) - zk + h)

                if i % 10 == 0 and (torch.norm(xkm1 - xk, dim=1) < tol).all() and \
                        (torch.norm(zkm1 - zk, dim=1) < tol).all() and (torch.norm(ukm1 - uk, dim=1) < tol).all():
                    ctx.save_for_backward(xk, zk, uk, G, h)
                    print(i)
                    return xk

                xkm1 = xk
                zkm1 = zk
                ukm1 = uk

            warnings.warn('Max iterations reached')
            ctx.save_for_backward(xk, zk, uk, G, h)
            return xk

        @staticmethod
        def backward(ctx, dl_dx):
            x, z, u, G, h = ctx.saved_tensors
            m = x.shape[-1]
            w = z.shape[-1]  # also equals u.shape[-1]
            loss_vec = torch.cat([dl_dx,
                                  torch.zeros(dl_dx.shape[0], w, device=dl_dx.device, dtype=TORCH_DTYPE),
                                  torch.zeros(dl_dx.shape[0], w, device=dl_dx.device, dtype=TORCH_DTYPE)],
                                 dim=1)

            dsoc = SOCProjFn.dproj_normcone(u / rho + G.bmm(x.unsqueeze(-1)).squeeze(-1) + h)
            mat = torch.cat([
                torch.cat([
                    torch.eye(m).unsqueeze(0) + rho * G.transpose(1, 2).bmm(G),
                    -rho * G.transpose(1, 2),
                    G.transpose(1, 2)], dim=2),
                torch.cat([
                    -dsoc.bmm(G),
                    torch.eye(w, device=dl_dx.device, dtype=TORCH_DTYPE).unsqueeze(0).expand(x.shape[0], w, w),
                    -dsoc / rho], dim=2),
                torch.cat([
                    G,
                    -torch.eye(w, device=dl_dx.device, dtype=TORCH_DTYPE).unsqueeze(0).expand(x.shape[0], w, w),
                    torch.zeros(x.shape[0], w, w, device=dl_dx.device, dtype=TORCH_DTYPE)], dim=2)],
                dim=1)
            res = torch.inverse(mat.transpose(1, 2)).bmm(loss_vec.unsqueeze(-1)).squeeze(-1)
            d_x = res[:, :m]
            d_z = res[:, m:m + w]
            d_u = res[:, -w:]

            dldy = d_x
            dldh = -rho * G.bmm(d_x.unsqueeze(-1)).squeeze(-1) + dsoc.bmm(d_z.unsqueeze(-1)).squeeze(-1) - d_u
            dldG = bger(-rho * G.bmm(x.unsqueeze(-1)).squeeze(-1) - u + rho * z - rho * h, d_x) - \
                   bger(rho * G.bmm(d_x.unsqueeze(-1)).squeeze(-1), x) + \
                   bger(dsoc.bmm(d_z.unsqueeze(-1)).squeeze(-1), x) - \
                   bger(d_u, x)

            dldA = dldG[:, :-1, :]
            dldb = dldh[:, :-1]
            dldc = dldG[:, -1, :]
            dldd = dldh[:, -1]

            return dldy, dldA, dldb, dldc, dldd

        @staticmethod
        def proj_normcone(z_in):
            '''Deals with 3 cases of projections: in cone (case 1), in "negative" cone (case 2), other (case 3)'''
            z = z_in[:, :-1]
            t = z_in[:, -1]
            z_norm = torch.norm(z, dim=1)
            case1m = (z_norm <= t)
            case2m = (z_norm <= -t)
            case3v = (z_norm + t).unsqueeze(-1) / 2 * \
                     torch.cat([z / z_norm.unsqueeze(-1), torch.ones(t.shape[0], 1, device=z_in.device, dtype=TORCH_DTYPE)], dim=1)
            return case1m.unsqueeze(-1).expand_as(z_in) * z_in + \
                   ~(case1m | case2m).unsqueeze(-1).expand_as(z_in) * case3v

        @staticmethod
        def dproj_normcone(z_in):
            '''Deals with 3 cases of projections: in cone (case 1), in "negative" cone (case 2), other (case 3)'''
            z = z_in[:, :-1]
            t = z_in[:, -1]

            z_norm = torch.norm(z, dim=1)
            d1dz = (bger(z, z) + \
                    (z_norm + t).unsqueeze(1).unsqueeze(2) * (
                            z_norm.unsqueeze(1).unsqueeze(2) * torch.eye(z.shape[1], device=z_in.device, dtype=TORCH_DTYPE).unsqueeze(0).expand(
                        z.shape[0], z.shape[1], z.shape[1])
                            - bger(z, z) / z_norm.unsqueeze(1).unsqueeze(2))) / (
                           2 * z_norm.unsqueeze(1).unsqueeze(2) ** 2)
            d1dr = (z.T / (2 * z_norm)).T
            case3v = torch.cat([
                torch.cat([d1dz, d1dr.unsqueeze(2)], dim=2),
                torch.cat([d1dr.unsqueeze(1), 0.5 * torch.ones(d1dr.shape[0], 1, 1, device=z_in.device, dtype=TORCH_DTYPE)], dim=2)],
                dim=1)

            case1m = (z_norm <= t)
            case2m = (z_norm <= -t)

            return case1m.unsqueeze(1).unsqueeze(2).expand_as(case3v) * torch.eye(z_in.shape[1], device=z_in.device, dtype=TORCH_DTYPE) + \
                   ~(case1m | case2m).unsqueeze(-1).unsqueeze(2).expand_as(case3v) * case3v

    return SOCProjFn.apply


def SOCProjFast(tol=1e-5, max_iters=10000, momentum=True):
    """Projection onto a second order cone constraint"""

    class SOCProjFastFn(Function):

        @staticmethod
        def forward(ctx, pi, A, b, c, d):
            G = torch.cat([A, c.unsqueeze(1)], dim=1)
            h = torch.cat([b, d.unsqueeze(-1)], dim=1)

            H = G.bmm(G.transpose(1,2))
            eig_H = torch.symeig(H,eigenvectors=False).eigenvalues

            mh = torch.min(eig_H,1)[0]
            Lh = torch.max(eig_H,1)[0]

            ## to avoid extremely small but negative mh
            threshold = 1e-5
            mh = (mh>threshold)*mh

            momentum_param = lambda iter: (momentum)*((mh>0)*((torch.sqrt(Lh)-torch.sqrt(mh))/(torch.sqrt(Lh)+torch.sqrt(mh))) + (mh==0)*((iter)/(iter+3))).unsqueeze(-1)

            step_size = (1/Lh).unsqueeze(-1)


            ## initial condition
            lamk = torch.zeros_like(h)
            lamkm1 = lamk
            xkm1 = pi

            for i in range(max_iters):

                vk = lamk + momentum_param(i) * (lamk-lamkm1)
                lamkp1 = SOCProjFastFn.proj_normcone(vk - step_size * (H.bmm(vk.unsqueeze(-1)) + G.bmm(pi.unsqueeze(-1))+h.unsqueeze(-1)).squeeze(-1))

                lamkm1 = lamk
                lamk = lamkp1
                # print(torch.max(rd))

                xk = pi + G.transpose(1, 2).bmm(lamk.unsqueeze(-1)).squeeze(-1)
                if torch.norm(xkm1 - xk, dim=1).max() < tol:
                    ctx.save_for_backward(xk, -lamk, G, h)
                    return xk
                xkm1 = xk

            warnings.warn('Max iterations reached')
            xk = pi + G.transpose(1,2).bmm(lamk.unsqueeze(-1)).squeeze(-1)
            ctx.save_for_backward(xk, -lamk, G, h)
            return xk

        @staticmethod
        def backward(ctx, dl_dx):
            x, u, G, h = ctx.saved_tensors
            z = G.bmm(x.unsqueeze(-1)).squeeze(-1) + h
            m = x.shape[-1]
            w = z.shape[-1]  # also equals u.shape[-1]
            loss_vec = torch.cat([dl_dx,
                                  torch.zeros(dl_dx.shape[0], w, device=dl_dx.device, dtype=TORCH_DTYPE),
                                  torch.zeros(dl_dx.shape[0], w, device=dl_dx.device, dtype=TORCH_DTYPE)],
                                 dim=1)

            dsoc = SOCProjFastFn.dproj_normcone(u + G.bmm(x.unsqueeze(-1)).squeeze(-1) + h)
            mat = torch.cat([
                torch.cat([
                    torch.eye(m, device=dl_dx.device, dtype=TORCH_DTYPE).unsqueeze(0) + G.transpose(1, 2).bmm(G),
                    -G.transpose(1, 2),
                    G.transpose(1, 2)], dim=2),
                torch.cat([
                    -dsoc.bmm(G),
                    torch.eye(w, device=dl_dx.device, dtype=TORCH_DTYPE).unsqueeze(0).expand(x.shape[0], w, w),
                    -dsoc], dim=2),
                torch.cat([
                    G,
                    -torch.eye(w, device=dl_dx.device, dtype=TORCH_DTYPE).unsqueeze(0).expand(x.shape[0], w, w),
                    torch.zeros(x.shape[0], w, w, device=dl_dx.device, dtype=TORCH_DTYPE)], dim=2)],
                dim=1)

            try:
                res = torch.inverse(mat.transpose(1, 2)).bmm(loss_vec.unsqueeze(-1)).squeeze(-1)
            except RuntimeError:
            # if (torch.det(mat.transpose(1, 2)) == 0).any():
                # print(dsoc[i, :, :])
                warnings.warn('Singular matrix in backwards pass')
                res = torch.zeros(mat.shape[0], loss_vec.shape[-1], device=dl_dx.device, dtype=TORCH_DTYPE)  # could also do 1e-5 * torch.ones
                i = np.argwhere((torch.det(mat.transpose(1, 2)) != 0).cpu()).reshape(-1).to(device=dl_dx.device)
                if i.shape[0] > 0:
                    res0 = torch.inverse(mat[i].transpose(1, 2)).bmm(loss_vec[i].unsqueeze(-1)).squeeze(-1)
                    res[i] = res0
                
            d_x = res[:, :m]
            d_z = res[:, m:m + w]
            d_u = res[:, -w:]

            dldy = d_x
            dldh = -G.bmm(d_x.unsqueeze(-1)).squeeze(-1) + dsoc.bmm(d_z.unsqueeze(-1)).squeeze(-1) - d_u
            dldG = bger(-G.bmm(x.unsqueeze(-1)).squeeze(-1) - u + z - h, d_x) - \
                   bger(G.bmm(d_x.unsqueeze(-1)).squeeze(-1), x) + \
                   bger(dsoc.bmm(d_z.unsqueeze(-1)).squeeze(-1), x) - \
                   bger(d_u, x)

            dldA = dldG[:, :-1, :]
            dldb = dldh[:, :-1]
            dldc = dldG[:, -1, :]
            dldd = dldh[:, -1]

            return dldy, dldA, dldb, dldc, dldd

        @staticmethod
        def proj_normcone(z_in):
            '''Deals with 3 cases of projections: in cone (case 1), in "negative" cone (case 2), other (case 3)'''
            z = z_in[:, :-1]
            t = z_in[:, -1]
            z_norm = torch.norm(z, dim=1)
            case1m = (z_norm <= t)
            case2m = (z_norm <= -t)
            case3v = (z_norm + t).unsqueeze(-1) / 2 * \
                     torch.cat([z / z_norm.unsqueeze(-1), torch.ones(t.shape[0], 1, device=z_in.device, dtype=TORCH_DTYPE)], dim=1)
            return case1m.unsqueeze(-1).expand_as(z_in) * z_in + \
                   ~(case1m | case2m).unsqueeze(-1).expand_as(z_in) * case3v

        @staticmethod
        def dproj_normcone(z_in):
            '''Deals with 3 cases of projections: in cone (case 1), in "negative" cone (case 2), other (case 3)'''
            z = z_in[:, :-1]
            t = z_in[:, -1]

            z_norm = torch.norm(z, dim=1)
            d1dz = (bger(z, z) +
                    (z_norm + t).unsqueeze(1).unsqueeze(2) * (
                            z_norm.unsqueeze(1).unsqueeze(2) * torch.eye(z.shape[1], device=z_in.device, dtype=TORCH_DTYPE).unsqueeze(0).expand(
                        z.shape[0], z.shape[1], z.shape[1])
                            - bger(z, z) / z_norm.unsqueeze(1).unsqueeze(2))) / (
                           2 * z_norm.unsqueeze(1).unsqueeze(2) ** 2)
            d1dr = (z.T / (2 * z_norm)).T
            case3v = torch.cat([
                torch.cat([d1dz, d1dr.unsqueeze(2)], dim=2),
                torch.cat([d1dr.unsqueeze(1), 0.5 * torch.ones(d1dr.shape[0], 1, 1, device=z_in.device, dtype=TORCH_DTYPE)], dim=2)],
                dim=1)

            case1m = (z_norm <= t)
            case2m = (z_norm <= -t)

            return case1m.unsqueeze(1).unsqueeze(2).expand_as(case3v) * torch.eye(z_in.shape[1], device=z_in.device, dtype=TORCH_DTYPE) + \
                   ~(case1m | case2m).unsqueeze(-1).unsqueeze(2).expand_as(case3v) * case3v

    return SOCProjFastFn.apply