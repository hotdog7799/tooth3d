import os
import time
from scipy import io
from numpy import linalg as LA

import torch
import numpy as np
import torch.fft as fft
import torch.nn.functional as f

import cv2
import matplotlib.pyplot as plt
import copy
from mpl_toolkits.axes_grid1 import make_axes_locatable

class ADMM3D():
    def __init__(self, config_list):
        # DataStamp
        self.dtstamp = time.strftime('%m%d_%H%M_python', time.localtime(time.time()))
        # Load config variables into self
        for con, val in config_list.items():
            try:
                exec("self.{} = val".format(con))
                print("{} = {}".format(con, val))
            except Exception as e:
                print("Parameter: {} setup error: {}".format(con, e))
                
        # GPU setting
        if torch.cuda.is_available():
            print("Device:{} selected".format(self.numGPU))
            self.device = torch.device('cuda:{}'.format(self.numGPU))
        else:
            print("GPU not available")
            self.device = torch.device('cpu')
        print(self.device)
        
        # Path settings
        self.cwd = os.getcwd()
        self.homed = os.path.expanduser('~')
        if self.path_ref:
            self.psf_file = self.homed + self.psf_file
            self.img_file = self.homed + self.img_file
        if not os.path.isdir("{}/{}".format(self.save_dir, self.dtstamp)):
            os.makedirs("{}/{}".format(self.save_dir, self.dtstamp))
        
        # Initialize data
        self.init_data()
        self.setup_folder()
        
        # Initialize metric lists for logging
        self.objective_history = []
        self.data_fidelity_history = []
        self.regularizer_history = []
        self.primal_resid_s_history = []
        self.dual_resid_s_history = []
        self.mu1_history = []
        self.mu2_history = []
        self.mu3_history = []

    def init_data(self):
        # Load PSF from .mat file
        psf = io.loadmat(self.psf_file, mat_dtype=True)['psf_stack']
        # Optionally rotate if required by your data
        psf = np.rot90(psf, 2)
        psf = psf.astype('float32')
        psf = psf - self.psf_bias
        psf[psf < 0] = 0
        psf = psf / LA.norm(psf)  # Norm 2 normalization
        psf = self.img_resize(psf, "lateral", self.lateral_downsample)
        psf = self.img_resize(psf, "axial", self.axial_downsample)

        # Load raw image
        raw = cv2.imread(self.img_file, flags=cv2.IMREAD_UNCHANGED)
        raw = np.rot90(raw, 2)
        raw = np.array(raw, dtype='float32')
        if len(raw.shape) == 3:
            if self.color_to_process.lower() == 'mono':
                raw = cv2.cvtColor(raw, cv2.COLOR_RGB2GRAY)
            else:
                color_list = {'red': 0, 'green': 1, 'blue': 2}
                raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
                raw = raw[:, :, color_list[self.color_to_process.lower()]]
        raw = raw - self.raw_bias
        raw[raw < 0] = 0
        raw = self.img_resize(raw, "lateral", self.lateral_downsample)
        raw = raw / np.max(raw)
        
        # Get dimensions from psf
        [self.Nx, self.Ny, self.Nz] = psf.shape
        
        self.psf = torch.from_numpy(psf).to(self.device)
        self.raw = torch.from_numpy(raw).to(self.device)
        
        # Set the reconstruction domain shape (will be updated if padded)
        self.x_shape = (self.Nx, self.Ny, self.Nz)
        return psf, raw

    def setup_folder(self):
        if self.save_dir.endswith('/'):
            self.save_dir = self.save_dir[:-1]
        save_path = os.path.join(self.save_dir, self.dtstamp)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        return

    def img_resize(self, X, flag, f):
        if flag == "lateral":
            num = int(np.log2(f))
            if num != 0:
                for i in range(num):
                    X = 0.25 * (X[::2, ::2, ...] + X[1::2, ::2, ...] +
                                X[::2, 1::2, ...] + X[1::2, 1::2, ...])
        elif flag == "axial":
            num = int(np.log2(f))
            if num != 0:
                for i in range(num):
                    X = 0.5 * (X[:, :, ::2, ...] + X[:, :, 1::2, ...])
        return X

    def pad2d(self, x):
        v_pad = int(np.floor(self.Nx/2))
        h_pad = int(np.floor(self.Ny/2))
        if x.ndim == 3:
            return f.pad(x, (0, 0, h_pad, h_pad, v_pad, v_pad))
        else:
            tmp = torch.unsqueeze(x, dim=2)
            return f.pad(tmp, (0, self.Nz-1, h_pad, h_pad, v_pad, v_pad), "constant", 0)

    def crop2d(self, x):
        v_crop = int(np.floor(self.Nx/2))
        h_crop = int(np.floor(self.Ny/2))
        v, h = x.shape[:2]
        return x[v_crop:v - v_crop, h_crop:h - h_crop]

    def crop3d(self, x):
        # return self.crop2d(x[:, :, 0])
        return x[:, :, 0]

    def l2norm(self, X):
        return torch.linalg.norm(X.ravel(), ord=2)

    def norm_8bit(self, X):
        max_X = X.max()
        if max_X == 0:
            max_X = 1
        A = X / max_X * 255
        return A.astype('uint8')

    def Hadj(self, x):
        return torch.real(fft.ifftn(self.Hs_conj * fft.fftn(x)))

    def Psi(self, x):
        # Compute first-order finite differences along each dimension
        return -torch.diff(x, n=1, dim=0), -torch.diff(x, n=1, dim=1), -torch.diff(x, n=1, dim=2)

    def PsiT(self, P1, P2, P3):
        # Reconstruct the divergence operator that is the negative adjoint of Psi.
        A = torch.cat([torch.unsqueeze(P1[0, :, :], dim=0), torch.diff(P1, 1, 0), -torch.unsqueeze(P1[-1, :, :], dim=0)], dim=0)
        B = torch.cat([torch.unsqueeze(P2[:, 0, :], dim=1), torch.diff(P2, 1, 1), -torch.unsqueeze(P2[:, -1, :], dim=1)], dim=1)
        C = torch.cat([torch.unsqueeze(P3[:, :, 0], dim=2), torch.diff(P3, 1, 2), -torch.unsqueeze(P3[:, :, -1], dim=2)], dim=2)
        return A + B + C

    def generate_laplacian(self, vk):
        laplacian = vk.clone().detach()
        laplacian[0, 0, 0] = 6
        laplacian[0, 1, 0] = -1
        laplacian[1, 0, 0] = -1
        laplacian[0, 0, 1] = -1
        laplacian[0, -1, 0] = -1
        laplacian[-1, 0, 0] = -1
        laplacian[0, 0, -1] = -1
        return torch.abs(fft.fftn(laplacian))

    def soft_threshold(self, X, tau_c):
        return torch.sign(X) * torch.maximum(torch.zeros_like(X), torch.abs(X) - tau_c)

    def soft_threshold_3d_z(self, v, h, d, tau, tau_z, arg=None):
        def size(x, i):
            return x.shape[i - 1]
        vararg_out = []
        if size(v, 1) != 0:
            mag_xy = torch.sqrt(torch.cat([v, torch.zeros(1, size(v, 2), size(v, 3)).to(self.device)], dim=0) ** 2 +
                                torch.cat([h, torch.zeros(size(h, 1), 1, size(h, 3)).to(self.device)], dim=1) ** 2)
            magtmag_xy = self.soft_threshold(mag_xy, tau)
            mmult_xy = magtmag_xy / (mag_xy + torch.finfo(torch.float32).eps)
            mmult_xy = mmult_xy * (mag_xy > 0)
            mmult_xy = torch.nan_to_num(mmult_xy)
            mag_z = torch.sqrt(torch.cat([d, torch.zeros((size(d, 1), size(d, 2), 1)).to(self.device)], dim=2) ** 2)
            magtmag_z = self.soft_threshold(mag_z, tau_z)
            mmult_z = magtmag_z / (mag_z + torch.finfo(torch.float32).eps)
            mmult_z = mmult_z * (mag_z > 0)
            mmult_z = torch.nan_to_num(mmult_z)
            vararg_out.append(v * mmult_xy[:-1, :, :])
            vararg_out.append(h * mmult_xy[:, :-1, :])
            vararg_out.append(d * mmult_z[:, :, :-1])
        else:
            vararg_out.append(self.soft_threshold(arg[0], tau))
        return vararg_out

    def soft_threshold_3d_cubic(self, vx, vy, vz, tau, beta):
        """
        Cubic TV soft-threshold.
        vx, vy, vz: 3D tensors (finite differences along x, y, and z)
          - vx: shape (Nx-1, Ny, Nz)
          - vy: shape (Nx, Ny-1, Nz)
          - vz: shape (Nx, Ny, Nz-1)
        beta: weighting factor for the z-difference.
        
        The function pads each gradient to the full image size,
        multiplies the z component by beta, computes the vector magnitude,
        applies soft-thresholding, and crops back to the original difference sizes.
        """
        # Compute full sizes
        Nx = vx.shape[0] + 1
        Ny = vy.shape[1] + 1
        Nz = vz.shape[2] + 1
        vx_full = torch.zeros((Nx, Ny, Nz), device=vx.device, dtype=vx.dtype)
        vy_full = torch.zeros((Nx, Ny, Nz), device=vy.device, dtype=vy.dtype)
        vz_full = torch.zeros((Nx, Ny, Nz), device=vz.device, dtype=vz.dtype)
        vx_full[:-1, :, :] = vx
        vy_full[:, :-1, :] = vy
        vz_full[:, :, :-1] = vz
        # Multiply the z component by beta
        v3d_z = beta * vz_full
        mag_3d = torch.sqrt(vx_full**2 + vy_full**2 + v3d_z**2 + 1e-12)
        shrink = torch.relu(mag_3d - tau) / (mag_3d + 1e-12)
        vx_new_full = vx_full * shrink
        vy_new_full = vy_full * shrink
        vz_new_full = (1.0 / beta) * (v3d_z * shrink)
        # Crop back to original difference sizes
        return vx_new_full[:-1, :, :], vy_new_full[:, :-1, :], vz_new_full[:, :, :-1]

    def update_param(self, mu, r, s):
        if r > self.resid_tol * s:
            mu_out = mu * self.mu_inc
            mu_update = 1
        elif r * self.resid_tol < s:
            mu_out = mu / self.mu_dec
            mu_update = -1
        else:
            mu_out = mu
            mu_update = 0
        return mu_out, mu_update

    def draw_fig(self, vk, n):
        vk = vk.astype(np.float32)
        global_min = vk.min()
        global_max = vk.max()
        if global_max - global_min != 0:
            norm_v = (vk - global_min) / (global_max - global_min)
        else:
            norm_v = vk
        H, W, D = norm_v.shape
        # half_crop = 60
        half_crop = self.half_crop
        print("half_crop = ",half_crop)
        crop_center_x = W // 2
        crop_center_y = H // 2

        def crop_img(img):
            return img[crop_center_y - half_crop : crop_center_y + half_crop,
                       crop_center_x - half_crop : crop_center_x + half_crop]

        cropped_stack = np.empty((half_crop*2, half_crop*2, D), dtype=norm_v.dtype)
        for i in range(D):
            cropped_stack[:, :, i] = crop_img(norm_v[:, :, i])
        print(f"crop_center_x = {crop_center_x}") # 4608
        print(f"crop_center_y = {crop_center_y}") # 2592
        print(f"cropped_stack shape = {cropped_stack.shape}")

        n_slices = cropped_stack.shape[2]
        n_cols = int(np.ceil(np.sqrt(n_slices)))
        n_rows = int(np.ceil(n_slices / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1 or n_cols == 1:
            axes = np.atleast_2d(axes)
        for i in range(n_slices):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].imshow(cropped_stack[:, :, i], cmap='gray', vmin=0, vmax=1)
            axes[row, col].set_title("Slice {}".format(i))
            axes[row, col].axis('on')
        total_subplots = n_rows * n_cols
        for j in range(n_slices, total_subplots):
            row = j // n_cols
            col = j % n_cols
            axes[row, col].axis('off')
        plt.suptitle("Iteration {}".format(n))
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if self.save_fig:
            iter_folder = os.path.join(self.save_dir, self.dtstamp, "iter_{}".format(n))
            if not os.path.exists(iter_folder):
                os.makedirs(iter_folder)
            overall_filename = os.path.join(iter_folder, "iter_{}_all_slices.png".format(n))
            plt.savefig(overall_filename)
        plt.show()

    def projection_plot(self, data, label=0, show=True):
        dataxy = np.sum(data, axis=2)
        dataxz = np.sum(data, axis=1)
        datayz = np.sum(data, axis=0)

        fig, axes = plt.subplots(figsize=(17, 4), nrows=1, ncols=3)
        ax0 = axes[0].imshow(dataxy, cmap='gray')
        fig.colorbar(ax0, cax=make_axes_locatable(axes[0]).append_axes('right', size='5%', pad=0.05),
                     orientation='vertical')
        axes[0].set_title("xy projection")
        axes[0].set_ylabel("y")
        axes[0].set_xlabel("x")
        ax1 = axes[1].imshow(dataxz, cmap='gray')
        fig.colorbar(ax1, cax=make_axes_locatable(axes[1]).append_axes('right', size='5%', pad=0.05),
                     orientation='vertical')
        axes[1].set_title("xz projection")
        axes[1].set_ylabel("x")
        axes[1].set_xlabel("z")
        ax2 = axes[2].imshow(datayz, cmap='gray')
        fig.colorbar(ax2, cax=make_axes_locatable(axes[2]).append_axes('right', size='5%', pad=0.05),
                     orientation='vertical')
        axes[2].set_title("yz projection")
        axes[2].set_ylabel("y")
        axes[2].set_xlabel("z")
        for ax in axes:
            ax.set_aspect(aspect='auto')
        plt.suptitle("data:{}".format(label))
        plt.tight_layout()
        if show:
            plt.show()
        return fig

    def crop_center(self, img, crop_size=150):
        h, w = img.shape
        startx = (w - crop_size) // 2
        starty = (h - crop_size) // 2
        return img[starty:starty+crop_size, startx:startx+crop_size]

    def plot_iteration_metrics(self):
        iterations = np.arange(len(self.objective_history))
        plt.figure(figsize=(12, 10))
        plt.subplot(2,2,1)
        plt.plot(iterations, self.objective_history, label="Objective")
        plt.xlabel("Iteration")
        plt.ylabel("Objective")
        plt.title("Objective vs Iterations")
        plt.legend()
        plt.subplot(2,2,2)
        plt.plot(iterations, self.data_fidelity_history, label="Data Fidelity")
        plt.plot(iterations, self.regularizer_history, label="Regularizer")
        plt.xlabel("Iteration")
        plt.ylabel("Cost")
        plt.title("Fidelity & Regularizer vs Iterations")
        plt.legend()
        plt.subplot(2,2,3)
        plt.plot(iterations, self.primal_resid_s_history, label="Primal Residual")
        plt.plot(iterations, self.dual_resid_s_history, label="Dual Residual")
        plt.xlabel("Iteration")
        plt.ylabel("Residual")
        plt.title("Residuals vs Iterations")
        plt.legend()
        plt.subplot(2,2,4)
        plt.plot(iterations, self.mu1_history, label="mu1")
        plt.plot(iterations, self.mu2_history, label="mu2")
        plt.plot(iterations, self.mu3_history, label="mu3")
        plt.xlabel("Iteration")
        plt.ylabel("Parameter value")
        plt.title("Parameters vs Iterations")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def admm_solver(self, psf, b): # 지금 쓰는 버전
        # Standard 3D TV solver (your current implementation)
        psf = psf / torch.linalg.norm(psf.ravel())
        psf = torch.roll(torch.flip(psf, dims=[2]), int(np.ceil(self.Nz/2)+1), dims=2)
        # self.Hs = fft.fftn(fft.ifftshift(self.pad2d(psf))) # Padding
        self.Hs = fft.fftn(fft.ifftshift(psf))
        self.Hs_conj = torch.conj(self.Hs)
        self.HtH = torch.abs(self.Hs * self.Hs_conj)
        vk = torch.zeros_like(self.Hs, dtype=torch.float32)
        xi = vk.clone().detach()
        rho = vk.clone().detach()
        # Dtb = self.pad2d(b) # Padding
        Dtb = b
        Dtb = Dtb.unsqueeze(-1) # 추가
        PsiTPsi = self.generate_laplacian(vk)
        eta_1 = vk[:-1, :, :].clone().detach()
        eta_2 = vk[:, :-1, :].clone().detach()
        eta_3 = vk[:, :, :-1].clone().detach()
        [uk1, uk2, uk3] = self.Psi(vk)
        Lvk1 = uk1.clone().detach()
        Lvk2 = uk2.clone().detach()
        Lvk3 = uk3.clone().detach()
        v_mult = 1 / (self.mu1 * self.HtH + self.mu2 * PsiTPsi + self.mu3)
        # DtD = self.pad2d(torch.ones_like(b)) # Padding
        DtD = torch.ones_like(b)
        nu_mult = 1 / (DtD + self.mu1)
        # nu_mult = nu_mult.unsqueeze(-1)
        
        Hvkp = torch.zeros_like(vk)
        vkp = torch.zeros_like(vk)
        dual_resid_s = np.zeros(self.max_iter)
        primal_resid_s = np.zeros(self.max_iter)
        dual_resid_u = np.zeros(self.max_iter)
        primal_resid_u = np.zeros(self.max_iter)
        dual_resid_w = np.zeros(self.max_iter)
        primal_resid_w = np.zeros(self.max_iter)
        objective = np.zeros(self.max_iter)
        data_fidelity = np.zeros(self.max_iter)
        regularizer_penalty = np.zeros(self.max_iter)
        n = 0
        while n < self.max_iter:
            Hvk = Hvkp.clone().detach()
            print("n: ",n)
            print("xi shape:", xi.shape) #torch.Size([2592, 4608, 4])
            print("Hvk shape:", Hvk.shape)#torch.Size([2592, 4608, 4])
            print("Dtb shape:", Dtb.shape)#torch.Size([2592, 4608])
            print("nu_mult shape: ",nu_mult.shape)
            if nu_mult.dim() == 2:   # [Nx, Ny]
                nu_mult = nu_mult.unsqueeze(-1)  # [Nx, Ny, 1]
                print("unsqueezed nu_mult shape: ",nu_mult.shape)
            nukp = nu_mult * (self.mu1 * (xi / self.mu1 + Hvk) + Dtb) #error
            wkp = torch.maximum(rho / self.mu3 + vk, torch.zeros_like(vk))
            [uk1, uk2, uk3] = self.soft_threshold_3d_z(Lvk1 + eta_1/self.mu2,
                                                        Lvk2 + eta_2/self.mu2,
                                                        Lvk3 + eta_3/self.mu2,
                                                        self.tau/self.mu2, self.tau_z/self.mu2)
            vkp_numerator = (self.mu3 * (wkp - rho/self.mu3) +
                             self.mu2 * self.PsiT(uk1 - eta_1/self.mu2,
                                                   uk2 - eta_2/self.mu2,
                                                   uk3 - eta_3/self.mu2) +
                             self.mu1 * self.Hadj(nukp - xi/self.mu1))
            vkp = torch.real(fft.ifftn(v_mult * fft.fftn(vkp_numerator)))
            Hvkp = torch.real(fft.ifftn(self.Hs * fft.fftn(vkp)))
            r_sv = Hvkp - nukp
            xi = xi + self.mu1 * r_sv
            dual_resid_s[n] = self.mu1 * self.l2norm(Hvk - Hvkp)
            print("Hvkp[:, :, 0].shape = ",Hvkp[:, :, 0].shape)
            primal_resid_s[n] = self.l2norm(r_sv)
            print("crop3d(Hvkp):", (self.crop3d(Hvkp)).shape)
            print("self.raw:", (self.raw).shape)
            data_fidelity[n] = 0.5 * self.l2norm(self.crop3d(Hvkp) - self.raw)**2
            Lvk1_ = copy.deepcopy(Lvk1)
            Lvk2_ = copy.deepcopy(Lvk2)
            Lvk3_ = copy.deepcopy(Lvk3)
            [Lvk1, Lvk2, Lvk3] = self.Psi(vkp)
            r_su_1 = Lvk1 - uk1
            r_su_2 = Lvk2 - uk2
            r_su_3 = Lvk3 - uk3
            eta_1 = eta_1 + self.mu2 * r_su_1
            eta_2 = eta_2 + self.mu2 * r_su_2
            eta_3 = eta_3 + self.mu2 * r_su_3
            dual_resid_u[n] = self.mu2 * torch.sqrt(self.l2norm(Lvk1_ - Lvk1)**2 +
                                                     self.l2norm(Lvk2_ - Lvk2)**2 +
                                                     self.l2norm(Lvk3_ - Lvk3)**2)
            primal_resid_u[n] = torch.sqrt(self.l2norm(Lvk1 - uk1)**2 +
                                            self.l2norm(Lvk2 - uk2)**2 +
                                            self.l2norm(Lvk3 - uk3)**2)
            regularizer_penalty[n] = self.tau*(torch.sum(torch.abs(Lvk1)) +
                                               torch.sum(torch.abs(Lvk2)) +
                                               torch.sum(torch.abs(Lvk3)))
            objective[n] = data_fidelity[n] + regularizer_penalty[n]
            r_sw = vkp - wkp
            rho = rho + self.mu3 * r_sw
            dual_resid_w[n] = self.mu3 * self.l2norm(vk - vkp)
            primal_resid_w[n] = self.l2norm(vkp - wkp)
            if self.autotune:
                [self.mu1, mu1_update] = self.update_param(self.mu1, primal_resid_s[n], dual_resid_s[n])
                [self.mu2, mu2_update] = self.update_param(self.mu2, primal_resid_u[n], dual_resid_u[n])
                [self.mu3, mu3_update] = self.update_param(self.mu3, primal_resid_w[n], dual_resid_w[n])
                if mu1_update or mu2_update or mu3_update:
                    v_mult = 1 / (self.mu1 * self.HtH + self.mu2 * PsiTPsi + self.mu3)
                    nu_mult = 1 / (DtD + self.mu1)
                    # nu_mult = nu_mult.unsqueeze(-1) # 패딩 안함
            vk = vkp
            self.objective_history.append(objective[n])
            self.data_fidelity_history.append(data_fidelity[n])
            self.regularizer_history.append(regularizer_penalty[n])
            self.primal_resid_s_history.append(primal_resid_s[n])
            self.dual_resid_s_history.append(dual_resid_s[n])
            self.mu1_history.append(self.mu1)
            self.mu2_history.append(self.mu2)
            self.mu3_history.append(self.mu3)
            if self.print_interval and ((n+1) % self.print_interval == 0 or n==0):
                print('iter:{} time:{:.3f} cost:{:.3f} data_fidelity:{:.3f} regularizer_penalty:{:.3f}'.format(
                      n, np.round(time.time() - self.st,2), objective[n],
                      data_fidelity[n], regularizer_penalty[n]))
                print('Primal/Dual v:{:.5f} Primal/Dual u:{:.5f} Primal/Dual w:{:.5f} mu1:{:.3f} mu2:{:.3f} mu3:{:.3f}'.format(
                      (primal_resid_s[n]/(dual_resid_s[n]+1e-9)),
                      (primal_resid_u[n]/(dual_resid_u[n]+1e-9)),
                      (primal_resid_w[n]/(dual_resid_w[n]+1e-9)),
                      self.mu1, self.mu2, self.mu3))
            if self.disp_figs and ((n+1)%self.disp_figs==0 or n==0):
                if self.useGPU:
                    out = vkp.to('cpu').numpy()
                    proj = vkp.to('cpu').numpy()
                    self.draw_fig(out, n)
                    self.projection_plot(proj, label=n)
            else:
                pass
            if self.useGPU:
                vkp = vkp.to('cpu').numpy()
            n = n + 1
        return vkp

    def admm_solver_cubicTV(self, psf, b):
        # Similar pre-processing as in admm_solver
        psf = psf / torch.linalg.norm(psf.ravel())
        shift = int(self.Nz // 2)
        psf = torch.roll(torch.flip(psf, dims=[2]), shift, dims=2)
        padded_psf = self.pad2d(psf)  # get padded PSF
        self.x_shape = padded_psf.shape   # update domain size accordingly
        self.Hs = fft.fftn(fft.ifftshift(padded_psf))
        self.Hs_conj = torch.conj(self.Hs)
        self.HtH = torch.abs(self.Hs * self.Hs_conj)
        vk = torch.zeros_like(self.Hs, dtype=torch.float32)
        xi = vk.clone().detach()
        rho = vk.clone().detach()
        Dtb = self.pad2d(b)
        # Use the same laplacian generation and difference variables:
        PsiTPsi = self.generate_laplacian(vk)
        eta_1 = vk[:-1, :, :].clone().detach()
        eta_2 = vk[:, :-1, :].clone().detach()
        eta_3 = vk[:, :, :-1].clone().detach()
        [uk1, uk2, uk3] = self.Psi(vk)
        Lvk1 = uk1.clone().detach()
        Lvk2 = uk2.clone().detach()
        Lvk3 = uk3.clone().detach()
        v_mult = 1 / (self.mu1 * self.HtH + self.mu2 * PsiTPsi + self.mu3)
        DtD = self.pad2d(torch.ones_like(b))
        nu_mult = 1 / (DtD + self.mu1)
        Hvkp = torch.zeros_like(vk)
        vkp = torch.zeros_like(vk)
        dual_resid_s = np.zeros(self.max_iter)
        primal_resid_s = np.zeros(self.max_iter)
        dual_resid_u = np.zeros(self.max_iter)
        primal_resid_u = np.zeros(self.max_iter)
        dual_resid_w = np.zeros(self.max_iter)
        primal_resid_w = np.zeros(self.max_iter)
        objective = np.zeros(self.max_iter)
        data_fidelity = np.zeros(self.max_iter)
        regularizer_penalty = np.zeros(self.max_iter)
        n = 0
        while n < self.max_iter:
            Hvk = Hvkp.clone().detach()
            nukp = nu_mult * (self.mu1 * (xi / self.mu1 + Hvk) + Dtb)
            wkp = torch.maximum(rho / self.mu3 + vk, torch.zeros_like(vk))
            # Here, call the cubic TV soft-threshold function instead:
            [uk1, uk2, uk3] = self.soft_threshold_3d_cubic(
                Lvk1 + eta_1 / self.mu2,
                Lvk2 + eta_2 / self.mu2,
                Lvk3 + eta_3 / self.mu2,
                tau=self.tau/self.mu2,
                beta=self.beta_z
            )
            vkp_numerator = (self.mu3 * (wkp - rho / self.mu3) +
                             self.mu2 * self.PsiT(uk1 - eta_1 / self.mu2,
                                                   uk2 - eta_2 / self.mu2,
                                                   uk3 - eta_3 / self.mu2) +
                             self.mu1 * self.Hadj(nukp - xi / self.mu1))
            vkp = torch.real(fft.ifftn(v_mult * fft.fftn(vkp_numerator)))
            Hvkp = torch.real(fft.ifftn(self.Hs * fft.fftn(vkp)))
            r_sv = Hvkp - nukp
            xi = xi + self.mu1 * r_sv
            dual_resid_s[n] = self.mu1 * self.l2norm(Hvk - Hvkp)
            primal_resid_s[n] = self.l2norm(r_sv)
            data_fidelity[n] = 0.5 * self.l2norm(self.crop3d(Hvkp) - b)**2
            Lvk1_ = copy.deepcopy(Lvk1)
            Lvk2_ = copy.deepcopy(Lvk2)
            Lvk3_ = copy.deepcopy(Lvk3)
            [Lvk1, Lvk2, Lvk3] = self.Psi(vkp)
            r_su_1 = Lvk1 - uk1
            r_su_2 = Lvk2 - uk2
            r_su_3 = Lvk3 - uk3
            eta_1 = eta_1 + self.mu2 * r_su_1
            eta_2 = eta_2 + self.mu2 * r_su_2
            eta_3 = eta_3 + self.mu2 * r_su_3
            dual_resid_u[n] = self.mu2 * torch.sqrt(self.l2norm(Lvk1_ - Lvk1)**2 +
                                                     self.l2norm(Lvk2_ - Lvk2)**2 +
                                                     self.l2norm(Lvk3_ - Lvk3)**2)
            primal_resid_u[n] = torch.sqrt(self.l2norm(Lvk1 - uk1)**2 +
                                            self.l2norm(Lvk2 - uk2)**2 +
                                            self.l2norm(Lvk3 - uk3)**2)
            regularizer_penalty[n] = self.tau*(torch.sum(torch.abs(Lvk1)) +
                                               torch.sum(torch.abs(Lvk2)) +
                                               torch.sum(torch.abs(Lvk3)))
            objective[n] = data_fidelity[n] + regularizer_penalty[n]
            r_sw = vkp - wkp
            rho = rho + self.mu3 * r_sw
            dual_resid_w[n] = self.mu3 * self.l2norm(vk - vkp)
            primal_resid_w[n] = self.l2norm(vkp - wkp)
            if self.autotune:
                [self.mu1, mu1_update] = self.update_param(self.mu1, primal_resid_s[n], dual_resid_s[n])
                [self.mu2, mu2_update] = self.update_param(self.mu2, primal_resid_u[n], dual_resid_u[n])
                [self.mu3, mu3_update] = self.update_param(self.mu3, primal_resid_w[n], dual_resid_w[n])
                if mu1_update or mu2_update or mu3_update:
                    v_mult = 1 / (self.mu1 * self.HtH + self.mu2 * PsiTPsi + self.mu3)
                    nu_mult = 1 / (DtD + self.mu1)
            vk = vkp
            self.objective_history.append(objective[n])
            self.data_fidelity_history.append(data_fidelity[n])
            self.regularizer_history.append(regularizer_penalty[n])
            self.primal_resid_s_history.append(primal_resid_s[n])
            self.dual_resid_s_history.append(dual_resid_s[n])
            self.mu1_history.append(self.mu1)
            self.mu2_history.append(self.mu2)
            self.mu3_history.append(self.mu3)
            if self.print_interval and ((n+1)%self.print_interval == 0 or n==0):
                print('iter:{} time:{:.3f} cost:{:.3f} data_fidelity:{:.3f} regularizer_penalty:{:.3f}'.format(
                      n, np.round(time.time()-self.st,2), objective[n],
                      data_fidelity[n], regularizer_penalty[n]))
                print('Primal/Dual v:{:.5f} Primal/Dual u:{:.5f} Primal/Dual w:{:.5f} mu1:{:.3f} mu2:{:.3f} mu3:{:.3f}'.format(
                      (primal_resid_s[n]/(dual_resid_s[n]+1e-9)),
                      (primal_resid_u[n]/(dual_resid_u[n]+1e-9)),
                      (primal_resid_w[n]/(dual_resid_w[n]+1e-9)),
                      self.mu1, self.mu2, self.mu3))
            if self.disp_figs and ((n+1)%self.disp_figs==0 or n==0):
                if self.useGPU:
                    out = vkp.to('cpu').numpy()
                    proj = vkp.to('cpu').numpy()
                    self.draw_fig(out, n)
                    self.projection_plot(proj, label=n)
            else:
                pass
            if self.useGPU:
                vkp = vkp.to('cpu').numpy()
            n = n + 1
        return vkp

    def admm(self):
        self.st = time.time()
        if self.regularizer.lower() == '3dtvz_cubic':
            final_im = self.admm_solver_cubicTV(self.psf, self.raw)
        else:
            final_im = self.admm_solver(self.psf, self.raw)
        print('Total elapsed Time: {}'.format(np.round(time.time() - self.st, 2)))
        print('Reconstruction Finished!')
        # io.savemat("{}/{}/final_data_uncropped.mat".format(self.save_dir, self.dtstamp),
        #            {"final": final_im}, format='5', do_compression=True)
        return final_im

if __name__ == "__main__":
    config = {}
    A = ADMM3D(config)
    final_im = A.admm()
    A.draw_fig(final_im, A.max_iter)
