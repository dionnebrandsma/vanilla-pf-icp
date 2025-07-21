import utils
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial import KDTree

class VanillaPFRegistration:
    def __init__(self, num_particles=50, max_iter=30, conv_thres=0.005):
        self.np = num_particles
        self.max_iter = max_iter
        self.conv_thres = conv_thres 

        self.particles = []
        self.TPara = None
        self.TParticles = []
        self.res = 0.5

    def register(self, source: np.ndarray, target: np.ndarray):
        assert source.shape[0] == 3 and target.shape[0] == 3
        self.Mov = source.copy()
        self.Ref = target.copy()
        self.nDim = self.Mov.shape[0]

        nbrs = NearestNeighbors(n_neighbors=2).fit(self.Ref.T)
        distances, _ = nbrs.kneighbors(self.Ref.T)
        scale = 10.0 * np.median(distances[:, 1])

        self.Ref /= scale
        self.Mov /= scale

        mDiff = np.mean(self.Ref, axis=1) - np.mean(self.Mov, axis=1)
        self.Mov += mDiff[:, np.newaxis]

        xRange = [np.min(self.Ref[0]), np.max(self.Ref[0])]
        yRange = [np.min(self.Ref[1]), np.max(self.Ref[1])]
        zRange = [np.min(self.Ref[2]), np.max(self.Ref[2])] if self.nDim == 3 else None

        x0 = np.zeros(6 if self.nDim == 3 else 3)
        R0, T0 = self._state2RTfun(x0)

        self.Aft = self._local2global(self.Mov, R0.T, T0)
        self.Mov = self.Aft

        self.Md = NearestNeighbors(n_neighbors=1).fit(self.Ref.T)

        self.particles = [{'w': 1.0 / self.np} for _ in range(self.np)]
        self.TPara = []

        for it in tqdm(range(self.max_iter)):
            self._motion_model(xRange, yRange, zRange)
            self._update_weight()
            xOpt = self._get_best_particle()

            if len(self.TPara) >= 2:
                if np.linalg.norm(self.TPara[-1] - self.TPara[-2]) <= self.conv_thres:
                    break

            optR, optT = self._state2RTfun(xOpt)
            self.Aft = self._local2global(self.Mov, optR.T, optT)

            self._resampling()

        corT = scale * (optR @ mDiff + optT)
        return scale*self.Aft, utils.homogeneous_transformation(optR, corT)

    def _motion_model(self, xRange, yRange, zRange):
        if 'S' in self.particles[0]:
            # Use Gaussian Mixture sampling
            Tmp = np.column_stack([p['xv'] for p in self.particles])
            Mu = Tmp.T
            W = np.array([p['w'] for p in self.particles])

            Sigma_list = [p['S'] for p in self.particles]
            Sigma = np.stack(Sigma_list)

            gm = GaussianMixture(n_components=len(self.particles), covariance_type='full')
            gm.weights_ = W / np.sum(W)
            gm.means_ = Mu
            gm.covariances_ = Sigma
            gm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(Sigma))

            ParaNew = gm.sample(len(self.particles))[0]
            for idx, p in enumerate(self.particles):
                p['xv'] = ParaNew[idx]
        else:
            for p in self.particles:
                para = [
                    np.random.uniform(*xRange),
                    np.random.uniform(*yRange),
                ]
                if self.nDim == 3:
                    para.append(np.random.uniform(*zRange))
                    para += list(np.random.uniform(-np.pi, np.pi, 3))
                else:
                    para.append(np.random.uniform(-np.pi, np.pi))
                p['xv'] = np.array(para)

        for p in self.particles:
            para0 = p['xv']
            R0, T0 = self._state2RTfun(para0)

            R_new, T_new = self._icp_update(self.Md, self.Ref, self.Mov, np.hstack((R0, T0.reshape(-1, 1))), 5.0 * self.res, 7)
            paraNew = self._RT2statefun(R_new, T_new)

            e = para0 - paraNew
            if self.nDim == 3:
                e[3:6] = (e[3:6] + np.pi) % (2 * np.pi) - np.pi
            else:
                e[2] = (e[2] + np.pi) % (2 * np.pi) - np.pi

            S = np.diag(np.square(e))
            p['S'] = S
            p['xv'] = paraNew

    def _update_weight(self):
        W = []
        for p in self.particles:
            R, T = self._state2RTfun(p['xv'])
            Aft = self._local2global(self.Mov, R.T, T)

            DD, _ = self.Md.kneighbors(Aft.T)
            DD = DD.flatten()

            W.append(-np.sum(DD ** 0.5))

        W = np.array(W)
        W -= np.max(W)
        W = np.exp(W)
        W /= np.sum(W)

        for i, p in enumerate(self.particles):
            p['w'] = W[i]

    def _get_best_particle(self):
        W = np.array([p['w'] for p in self.particles])
        max_id = np.argmax(W)
        xOpt = self.particles[max_id]['xv']
        self.TParticles.append({'data': self.particles.copy()})
        self.TPara.append(xOpt)
        return xOpt

    def _resampling(self):
        w = np.array([p['w'] for p in self.particles])
        Neff = 1.0 / np.sum(w**2)
        N = len(self.particles)
        threshold = N * 0.5
        if Neff > threshold:
            return

        positions = (np.random.rand(N) + np.arange(N)) / N
        indexes = np.zeros(N, dtype=int)
        cumulative_sum = np.cumsum(w)
        i = 0
        for j in range(N):
            while positions[j] > cumulative_sum[i]:
                i += 1
            indexes[j] = i

        new_particles = []
        for i in indexes:
            p = self.particles[i].copy()
            p['w'] = 1.0 / N
            new_particles.append(p)

        self.particles = new_particles

    def _local2global(self, points, R, T):
        return R.T @ points + T[:, np.newaxis]
    
    def _state2RTfun(self, para):
        para = np.asarray(para)
        if len(para) == 3:  # 2D
            T = para[0:2]
            ang = para[2]
            R = np.array([[np.cos(ang), -np.sin(ang)],
                        [np.sin(ang),  np.cos(ang)]])
        elif len(para) == 6:  # 3D
            T = para[0:3]
            euler_angles = para[3:]
            R = Rot.from_euler('xyz', euler_angles).as_matrix()
        return R, T

    def _RT2statefun(self, R, T):
        T = np.asarray(T)
        if len(T) == 2:  # 2D
            # Promote 2D R to 3x3 to use rotm2eul
            R_3x3 = np.eye(3)
            R_3x3[:2, :2] = R
            ang = Rot.from_matrix(R_3x3).as_euler('xyz')[0]
            para = np.concatenate([T, [ang]])
        elif len(T) == 3:  # 3D
            ang = Rot.from_matrix(R).as_euler('xyz')
            para = np.concatenate([T, ang])
        return para

    def _reg_fun(self, A, B):
        # A, B are both 3xN
        A = np.squeeze(A)
        centroid_A = np.mean(A, axis=1, keepdims=True)
        centroid_B = np.mean(B, axis=1, keepdims=True)
        H = (B - centroid_B) @ (A - centroid_A).T
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        T = centroid_A.squeeze() - R @ centroid_B.squeeze()
        return R, T

    def _icp_update(self, Md, Ref, Mov, Tf0, DistThr, L):
        max_iter = L
        dim = Ref.shape[0]
        R = Tf0[:dim, :dim]
        T = Tf0[:dim, -1]

        for _ in range(max_iter):
            # Transform Mov points using current estimate
            AftData = self._local2global(Mov, R.T, T)

            # Find nearest neighbors in Ref
            DD, NNIdx = Md.kneighbors(AftData.T)
            idx = np.where(DD < DistThr)[0]
            tmpDist = DistThr

            # Ensure enough correspondences
            while len(idx) < 10:
                tmpDist += 0.5 * DistThr
                idx = np.where(DD < tmpDist)[0]

            MovIdx = idx
            RefIdx = NNIdx[idx]

            # Compute optimal R, T between matched pairs
            dR, dT = self._reg_fun(Ref[:, RefIdx], AftData[:, MovIdx])

            # Update global transformation
            R = dR @ R
            T = dR @ T + dT

            # Check convergence
            err = max(np.linalg.norm(dR - np.eye(dim)), np.linalg.norm(dT))
            if err <= 1e-6:
                break
        return R, T

