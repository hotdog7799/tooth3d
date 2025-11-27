# -*- coding:utf-8 -*-
# Package Import
# %matplotlib inline
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
'''
# Config List
config_list = {
    # IMG, PSF data
    'path_ref': 0,  # 1: start from '~' 0:current or abs path
    'psf_file': "./20220105_polar/stack_220114_psf_40cm_demosaic_12bit_199987us_1.0ag_1.0dg_1.6094r_1.7266b_10channel.mat",  # if path_ref=1, start with /
    'img_file': "./20220105_polar/lpcp0112_40cm_demosaic_12bit_699999us_4.0ag_1.0dg_1.6094r_1.7266b.tiff",
    # 'psf_file': "./waller_data/example_psfs.mat",  # if path_ref=1, start with /
    # 'img_file': "./waller_data/example_raw.png",
    ###Save setup
    # 'save_dir': "./waller_data/torch0506_USAF_with_addtional_L2norm_onPSF_PSF_Load_Bias_L2normalize_Resize_RAW_Load_Bias_Resize_Maxnormalize_tau6e-4_tauz_6e-10",  # if path_ref=1, start with /
    'save_dir': "./20220105_polar/torch0511_scene0118_2_45cm_PSF_Load_Bias_L2normalize_Resize_RAW_Load_Bias_Resize_Maxnormalize_tau6e-4_tauz_6e-10",  # if path_ref=1, start with /
    # 'save_dir': "./20220105_polar/torch0506_lpcp_with_addtional_L2norm_onPSF_PSF_Load_Bias_L2normalize_Resize_RAW_Load_Bias_Resize_Maxnormalize_change_tauz9e-3",  # if path_ref=1, start with /
    # 'save_every': 200,  # Save image stack as .mat every N iterations. Use 0 to never save (except for at the end);
    'save_fig': True,
    # Data Setup
    'color_to_process': "mono",  # 'red','green','blue', or 'mono'. If raw file is mono, this is ignored
    # 'psf_bias': 102,  # if PSF needs sensor bias removed, put that here.
    # 'raw_bias': 100,  # If camera has bias, subtract from measurement file.
    'psf_bias': 0,  # if PSF needs sensor bias removed, put that here.
    'raw_bias': 600,  # If camera has bias, subtract from measurement file.
    'lateral_downsample': 8,  # down sample image
    'axial_downsample': 1,  # Axial averageing of impulse stack. Must be multiple of 2 and >= 1.
    'start_z': 0,  # First plane to reconstruct. 1 indexed, as is tradition.
    'end_z': 0,  # Last plane to reconstruct. If set to 0, use last plane in file.
    # GPU setup
    'useGPU': True,
    'numGPU': 0,
    # Recon Parameters
    'max_iter': 3000,  # Maximum iteration count  Default: 200
    'print_interval': 200, # Print cost every N iterations. Default 1. If set to 0, don't print.
    'disp_figs': 200,  # If set to 0, never display. If set to N>=1, show every N.
    'regularizer': '3dtvz',
    # Optimization Parameters
    'mu1': 1,  # 0.26
    'mu2': 1,  # 0.68,
    'mu3': 1,  # 3.5
    'tau': 6.0e-4,  # 0.008 sparsity parameter for TV
    'tau_z' : 6.0e-10,
    'tau_n': 0.06,  # sparsity param for native sparsity
    # Tuning Parameter
    'autotune': 1,  # 1:auto-find mu every step. 0:defined values. If set to N>1, tune for N steps then stop.
    'mu_inc': 1.1,  # Inrement and decrement values for mu during autotune.
    'mu_dec': 1.1,  #
    'resid_tol': 1.5,  # Primal/dual gap tolerance.
    # Display setup
    'roih': 200,
    'roiw': 200,
}
'''
class ADMM3D():
    ## ADMM Solver for Python ##
    def __init__(self, config_list):
        # Config list to self.variables
        for con, val in config_list.items():
            try:
                exec("self.{}=val".format(con))
                print("{} = {}".format(con, val))
            except:
                print("Parameter: {} setup error".format(con))
        
        # DataStamp
        self.dtstamp = time.strftime('%m%d_%H%M%S', time.localtime(time.time())) + "_" + self.color_to_process + "_tau" + str(self.tau) + "_ztau" + str(self.tau_z)

        # GPU setting
        # global cp, fft
        if torch.cuda.is_available():
            print("Device:{} selected".format(self.numGPU))
            self.device = torch.device('cuda:{}'.format(self.numGPU))
        else:
            print("GPU not available")
            self.device = torch.device('cpu')
        print(self.device)

        # Path setting
        self.cwd = os.getcwd()
        self.homed = os.path.expanduser('~')
        if self.path_ref:
            self.psf_file = self.homed + self.psf_file
            self.img_file = self.homed + self.img_file
        # Save path
        if not os.path.isdir("{}/{}".format(self.save_dir, self.dtstamp)): os.makedirs(
            "{}/{}".format(self.save_dir, self.dtstamp))
        # Initialize data
        self.init_data()
        # Setup folder
        self.setup_folder()
        # self.save_state()
        self.save_path = os.path.join(self.save_dir, self.dtstamp)
        
        # 250203 추가
        self.objective_history = []
        self.data_fidelity_history = []
        self.regularizer_history = []
        self.primal_resid_s_history = []
        self.dual_resid_s_history = []
        self.mu1_history = []
        self.mu2_history = []
        self.mu3_history = []

    # PreProcessing
    def init_data(self):
        ### 1. Load PSF
        psf = io.loadmat(self.psf_file, mat_dtype=True)['psf_stack']
        psf = psf.astype('float32')
        psf = psf - self.psf_bias
        psf[psf < 0] = 0
        psf = psf / LA.norm(psf)  # Norm 2 normalization
        psf = self.img_resize(psf, "lateral", self.lateral_downsample)
        psf = self.img_resize(psf, "axial", self.axial_downsample)

        ### 2. Load raw image
        raw = cv2.imread(self.img_file, flags=cv2.IMREAD_UNCHANGED)
        raw = np.array(raw, dtype='float32')
        if len(raw.shape) == 3:
            # If using mono or a single color channel, convert accordingly.
            if self.color_to_process.lower() in ['mono', 'red', 'green', 'blue']:
                if self.color_to_process.lower() == 'mono':
                    raw = cv2.cvtColor(raw, cv2.COLOR_RGB2GRAY)
                else:
                    # Convert BGR (default from cv2) to RGB and select the desired channel.
                    raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
                    color_list = {'red': 0, 'green': 1, 'blue': 2}
                    raw = raw[:, :, color_list[self.color_to_process.lower()]]
            # For merged channels (e.g. 'rg' or 'rgb'), keep the full RGB image.
            elif self.color_to_process.lower() in ['rg', 'rgb']:
                raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        # Else if raw is already single channel, nothing to do.

        # Apply bias correction
        raw = raw - self.raw_bias
        raw[raw < 0] = 0
        # Lateral downsampling (applied to raw only laterally)
        raw = self.img_resize(raw, "lateral", self.lateral_downsample)
        # Normalize raw image to have maximum 1 (for algorithm stability)
        raw = raw / np.max(raw)

        # Get dimensions from the PSF
        [self.Nx, self.Ny, self.Nz] = psf.shape

        self.psf = torch.from_numpy(psf).to(self.device)
        # For raw: if multi-channel, keep as is; if single channel, add channel axis if needed.
        if len(raw.shape) == 2:
            self.raw = torch.from_numpy(raw).to(self.device)
        else:
            self.raw = torch.from_numpy(raw).to(self.device)

        return psf, raw

    def img_bias(self, x, bias=0):
        return self.non_neg(x - bias)

    def non_neg(self, x):
        x = torch.maximum(x, torch.tensor((0)))
        return x

    # Save, Folder Setting Functions
    def setup_folder(self):
    # 만약 save_dir이 '/'로 끝나면 마지막 문자만 제거
        if self.save_dir.endswith('/'):
            self.save_dir = self.save_dir[:-1]
        # dtstamp 폴더를 포함한 전체 저장 경로 생성
        save_path = os.path.join(self.save_dir, self.dtstamp)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        return

    # ADMM Main Function
    def admm(self):
        self.st = time.time()
        # Case 1: Single-channel processing.
        if self.color_to_process.lower() in ['red', 'green', 'blue', 'mono']:
            final_im = self.admm_solver(self.psf, self.raw)
        # Case 2: Multi-channel processing (e.g., 'rg' or 'rgb').
        elif self.color_to_process.lower() in ['rg', 'rgb']:
            channels_to_process = list(self.color_to_process.lower())
            results = []
            # Define a map from letter to channel index (assuming raw image is in RGB order)
            color_map = {'r': 0, 'g': 1, 'b': 2}
            for c in channels_to_process:
                print("Reconstructing channel {}...".format(c))
                # For raw: select the c-th channel.
                if len(self.raw.shape) == 3:
                    raw_channel = self.raw[:, :, color_map[c]]
                else:
                    raise ValueError("Raw image does not have multiple channels for merging.")
                # For PSF: if PSF is 4D (with channel dimension), select the corresponding channel;
                # otherwise, use the same PSF for all channels.
                if self.psf.ndim == 4:
                    psf_channel = self.psf[:, :, :, color_map[c]]
                else:
                    psf_channel = self.psf
                # Perform reconstruction for the specific channel.
                xhat_channel = self.admm_solver(psf_channel, raw_channel)
                results.append(xhat_channel)
            # Stack the results along the third axis (channels)
            final_im = np.stack(results, axis=-1)
            # Global normalization: compute the global min and max over all channels,
            # then scale the merged result to [0, 255].
            global_min = final_im.min()
            global_max = final_im.max()
            final_im = (final_im - global_min) / (global_max - global_min) * 255
            final_im = final_im.astype(np.uint8)
        else:
            print("Unknown color_to_process configuration. Using mono reconstruction.")
            final_im = self.admm_solver(self.psf, self.raw)

        print('Total elapsed Time: {}'.format(np.round(time.time() - self.st, 2)))
        print('Reconstruction Finished!')
        io.savemat("{}/{}/final_data_uncropped.mat".format(self.save_dir, self.dtstamp),
                   {"final": final_im}, format='5', do_compression=True)
        return final_im

    def pad2d(self, x):
        v_pad = int(np.floor(self.Nx/2))
        h_pad = int(np.floor(self.Ny/2))

        if x.ndim == 3:
            return f.pad(x, (0, 0, h_pad, h_pad, v_pad, v_pad))
        else:
            # x = np.expand_dims(x, axis=2)
            tmp = torch.unsqueeze(x, dim=2)
            return f.pad(tmp, (0, self.Nz-1, h_pad, h_pad, v_pad, v_pad), "constant", 0)

    def crop2d(self, x):
        v_crop = int(np.floor(self.Nx/2))
        h_crop = int(np.floor(self.Ny/2))
        v, h = x.shape[:2]
        return x[v_crop:v - v_crop, h_crop:h - h_crop]

    def crop3d(self, x):
        return self.crop2d(x[:, :, 0])

    def admm_solver(self, psf, b):
        # 0) PSF 전처리 & FFT
        psf = psf / torch.linalg.norm(psf.ravel())   # Additional Norm 2 normalization in accordance with MATLAB code
        # 문제의 부분
        # psf = torch.roll(torch.flip(psf, dims=[2]), int(np.ceil(self.Nz / 2) + 1), dims=2)  # Rolling for 3d fft
        shift = int(self.Nz // 2)
        psf = torch.roll(torch.flip(psf, dims=[2]), shift, dims=2)
        self.Hs = fft.fftn(fft.ifftshift(self.pad2d(psf)))  # 3D FFT on padded PSF
        self.Hs_conj = torch.conj(self.Hs)
        self.HtH = torch.abs(self.Hs * self.Hs_conj)
        
        # 1) 초기화
        vk = torch.zeros_like(self.Hs, dtype=torch.float32)  # initialize vk as zero-padded size
        xi = vk.clone().detach()  # Dual variable for Hs*v = nu
        rho = vk.clone().detach()  # Dual variable for v = w (nonnegativity)
        Dtb = self.pad2d(b)      # Pad raw image to 3D

        #### Regularizer setup (3D TV)
        PsiTPsi = self.generate_laplacian(vk) # 여기 1 (Gramian Matrix)
        eta_1 = vk[:-1, :, :].clone().detach()
        eta_2 = vk[:, :-1, :].clone().detach()
        eta_3 = vk[:, :, :-1].clone().detach()
        [uk1, uk2, uk3] = self.Psi(vk) # 여기 2
        Lvk1 = uk1.clone().detach()
        Lvk2 = uk2.clone().detach()
        Lvk3 = uk3.clone().detach()

        # Precomputed denominators in Fourier domain
        v_mult = 1 / (self.mu1 * self.HtH + self.mu2 * PsiTPsi + self.mu3)
        DtD = self.pad2d(torch.ones_like(b))
        nu_mult = 1 / (DtD + self.mu1)  # Denom. for nu update

        # Initialize intermediate variables for iterations
        Hvkp = torch.zeros_like(vk)
        vkp = torch.zeros_like(vk)

        # Initialize metric arrays (using numpy) for current iteration count
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
            Hvk = Hvkp.clone().detach()  # copy previous v in Fourier domain
            # Nu update (x-update in 2D code)(data fidelity)
            nukp = nu_mult * (self.mu1 * (xi / self.mu1 + Hvk) + Dtb)
            # w update (nonnegativity constraint)
            wkp = torch.maximum(rho / self.mu3 + vk, torch.zeros_like(vk))
            #### 3D TV soft-thresholding update ### 여기 3 다른 정규화함수로 교체
            [uk1, uk2, uk3] = self.soft_threshold_3d_z(Lvk1 + eta_1 / self.mu2,
                                                        Lvk2 + eta_2 / self.mu2,
                                                        Lvk3 + eta_3 / self.mu2,
                                                        self.tau / self.mu2, self.tau_z / self.mu2)
            # Calculate numerator for v update
            # 여기 self.PsiT 4
            vkp_numerator = (self.mu3 * (wkp - rho / self.mu3) +
                             self.mu2 * self.PsiT(uk1 - eta_1 / self.mu2,
                                                   uk2 - eta_2 / self.mu2,
                                                   uk3 - eta_3 / self.mu2) +
                             self.mu1 * self.Hadj(nukp - xi / self.mu1))
            # v update via FFT-based division
            vkp = torch.real(fft.ifftn(v_mult * fft.fftn(vkp_numerator)))

            # Update dual variable for Hs*v = nu
            Hvkp = torch.real(fft.ifftn(self.Hs * fft.fftn(vkp)))
            r_sv = Hvkp - nukp
            xi = xi + self.mu1 * r_sv
            dual_resid_s[n] = self.mu1 * self.l2norm(Hvk - Hvkp)
            primal_resid_s[n] = self.l2norm(r_sv)

            # Data fidelity term (difference between cropped Hvkp and raw image)
            # data_fidelity[n] = 0.5 * self.l2norm(self.crop3d(Hvkp) - self.raw) ** 2
            # Data fidelity term (difference between cropped Hvkp and raw image channel)
            data_fidelity[n] = 0.5 * self.l2norm(self.crop3d(Hvkp) - b) ** 2

            #### Regularizer update
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

            dual_resid_u[n] = self.mu2 * torch.sqrt(self.l2norm(Lvk1_ - Lvk1) ** 2 +
                                                      self.l2norm(Lvk2_ - Lvk2) ** 2 +
                                                      self.l2norm(Lvk3_ - Lvk3) ** 2)
            primal_resid_u[n] = torch.sqrt(self.l2norm(Lvk1 - uk1) ** 2 +
                                            self.l2norm(Lvk2 - uk2) ** 2 +
                                            self.l2norm(Lvk3 - uk3) ** 2)
            regularizer_penalty[n] = self.tau * (torch.sum(torch.abs(Lvk1)) +
                                                 torch.sum(torch.abs(Lvk2)) +
                                                 torch.sum(torch.abs(Lvk3)))
            # Total objective (loss)
            objective[n] = data_fidelity[n] + regularizer_penalty[n]

            # Nonnegativity dual update
            r_sw = vkp - wkp
            rho = rho + self.mu3 * r_sw
            dual_resid_w[n] = self.mu3 * self.l2norm(vk - vkp)
            primal_resid_w[n] = self.l2norm(vkp - wkp)

            # Parameter autotune (if enabled)
            if self.autotune:
                [self.mu1, mu1_update] = self.update_param(self.mu1, primal_resid_s[n], dual_resid_s[n])
                [self.mu2, mu2_update] = self.update_param(self.mu2, primal_resid_u[n], dual_resid_u[n])
                [self.mu3, mu3_update] = self.update_param(self.mu3, primal_resid_w[n], dual_resid_w[n])
                if mu1_update or mu2_update or mu3_update:
                    v_mult = 1 / (self.mu1 * self.HtH + self.mu2 * PsiTPsi + self.mu3)
                    nu_mult = 1 / (DtD + self.mu1)

            vk = vkp  # update vk for next iteration

            # Record metrics at this iteration for tracking
            self.objective_history.append(objective[n])
            self.data_fidelity_history.append(data_fidelity[n])
            self.regularizer_history.append(regularizer_penalty[n])
            self.primal_resid_s_history.append(primal_resid_s[n])
            self.dual_resid_s_history.append(dual_resid_s[n])
            self.mu1_history.append(self.mu1)
            self.mu2_history.append(self.mu2)
            self.mu3_history.append(self.mu3)

            # Print status if required
            if self.print_interval and ((n+1) % self.print_interval == 0 or n == 0):
                print('iter:{} time:{:.3f} cost:{:.3f} data_fidelity:{:.3f} regularizer_penalty:{:.3f}'
                      .format(n, np.round(time.time() - self.st, 2), objective[n],
                              data_fidelity[n], regularizer_penalty[n]))
                print('Primal/Dual v:{:.5f} Primal/Dual u:{:.5f} Primal/Dual w:{:.5f} mu1:{:.3f} mu2:{:.3f} mu3:{:.3f}'
                      .format((primal_resid_s[n] / dual_resid_s[n]),
                              (primal_resid_u[n] / dual_resid_u[n]),
                              (primal_resid_w[n] / dual_resid_w[n]),
                              self.mu1, self.mu2, self.mu3))

            # Display figures if required
            if self.disp_figs and ((n+1) % self.disp_figs == 0 or n == 0):
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

        print("vk shape in draw_fig:", vk.shape)
        return vkp

    ## Member Functions ##
    # PSF, ADMM processing

    def Hadj(self, x):
        return torch.real(fft.ifftn(self.Hs_conj * fft.fftn(x)))

    def Psi(self, x):
        return -torch.diff(x, n=1, dim=0), -torch.diff(x, n=1, dim=1), -torch.diff(x, n=1, dim=2)

    def PsiT(self, P1, P2, P3):
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
        '''
        Perform isotropic soft thresholding on volume differences, v, h, and d
        using parameter tau. If a 4th input it added, assume it's the original
        volume and soft threshold that as well (for TV + sparsity).
        for TV+native: pass in 1 more inputs being original volume
        '''

        def size(x, i):
            return x.shape[i - 1]

        vararg_out = []

        if size(v, 1) != 0:
            mag_xy = torch.sqrt(torch.cat([v, torch.zeros(1, size(v, 2), size(v, 3)).to(self.device)], dim=0) ** 2
                          + torch.cat([h, torch.zeros(size(h, 1), 1, size(h, 3)).to(self.device)], dim=1) ** 2)
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
        import os
        import matplotlib.pyplot as plt
        import numpy as np

        # 1. Global normalization
        vk = vk.astype(np.float32)
        global_min = vk.min()
        global_max = vk.max()
        if global_max - global_min != 0:
            norm_v = (vk - global_min) / (global_max - global_min)
        else:
            norm_v = vk

        # 2. Crop each slice after reconstruction.
        #    Crop parameters: center at (x,y)=(581,297), side length = 120 pixels.
        #    Phantom(581,297) Simulation(571,320)
        crop_center_x = 581  # x 좌표 (열) # 581()
        crop_center_y = 297  # y 좌표 (행) # 297(phantom)
        half_crop = 60       # 120/2

        def crop_img(img):
            # img는 2D 배열 (height, width)
            return img[crop_center_y - half_crop : crop_center_y + half_crop,
                       crop_center_x - half_crop : crop_center_x + half_crop]

        # norm_v shape: (H, W, D) for mono channel
        H, W, D = norm_v.shape
        cropped_stack = np.empty((half_crop*2, half_crop*2, D), dtype=norm_v.dtype)
        for i in range(D):
            cropped_stack[:, :, i] = crop_img(norm_v[:, :, i])

        # 3. Plot all slices in a grid.
        n_slices = cropped_stack.shape[2]
        # grid: 열 수를 sqrt(n_slices) 정도로 정함.
        n_cols = int(np.ceil(np.sqrt(n_slices)))
        n_rows = int(np.ceil(n_slices / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        # axes가 2차원 배열가 아닐 경우 변환
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1 or n_cols == 1:
            axes = np.atleast_2d(axes)
        for i in range(n_slices):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].imshow(cropped_stack[:, :, i], cmap='gray', vmin=0, vmax=1)
            # psf_labels가 있으면 해당 라벨로 제목 설정.
            if hasattr(self, 'psf_labels') and self.psf_labels is not None and i < len(self.psf_labels):
                axes[row, col].set_title("{}".format(self.psf_labels[i]))
            else:
                axes[row, col].set_title("Slice {}".format(i))
            axes[row, col].axis('off')
        # 빈 subplot이 있다면 숨김 처리
        total_subplots = n_rows * n_cols
        for j in range(n_slices, total_subplots):
            row = j // n_cols
            col = j % n_cols
            axes[row, col].axis('off')
        plt.suptitle("Iteration {}".format(n))
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        # 저장: 모든 slice grid 플롯
        if self.save_fig:
            # iter_folder = os.path.join(self.save_dir, self.dtstamp, "iter_{}".format(n))
            iter_folder = os.path.join(self.save_dir, self.dtstamp)
            if not os.path.exists(iter_folder):
                os.makedirs(iter_folder)
            overall_filename = os.path.join(iter_folder, "slices_iter_{}.png".format(n))
            plt.savefig(overall_filename)
        plt.show()

        # 4. Compute image subtraction: first slice minus last slice.
        # diff = np.abs(cropped_stack[:, :, 0] - cropped_stack[:, :, -1])
        # plt.figure(figsize=(4, 4))
        # plt.imshow(diff, cmap='gray', vmin=0, vmax=1)
        # plt.title("Subtraction: First - Last")
        # plt.axis('off')
        # if self.save_fig:
        #     diff_filename = os.path.join(iter_folder, "iter_{}_subtraction.png".format(n))
        #     plt.savefig(diff_filename)
        # plt.show()

        # 5. Save individual slice images without colorbar or title.
        if self.save_fig:
            slice_folder = os.path.join(self.save_dir, self.dtstamp, "iter_{}_slices".format(n))
            if not os.path.exists(slice_folder):
                os.makedirs(slice_folder)
            for i in range(n_slices):
                if hasattr(self, 'psf_labels') and self.psf_labels is not None and i < len(self.psf_labels):
                    filename = "{}.png".format(self.psf_labels[i])
                else:
                    filename = "slice_{}.png".format(i)
                plt.imsave(os.path.join(slice_folder, filename),
                           cropped_stack[:, :, i], cmap='gray', vmin=0, vmax=1)

        
    @staticmethod
    def crop_center(img, crop_size=150):
        if img.ndim == 2:
            h, w = img.shape
            startx = (w - crop_size) // 2
            starty = (h - crop_size) // 2
            return img[starty:starty+crop_size, startx:startx+crop_size]
        elif img.ndim == 3:
            h, w, c = img.shape
            startx = (w - crop_size) // 2
            starty = (h - crop_size) // 2
            return img[starty:starty+crop_size, startx:startx+crop_size, :]

    def plot_iteration_metrics(self):
        iterations = np.arange(len(self.objective_history))
        plt.figure(figsize=(12, 10))

        plt.subplot(3, 2, 1)
        plt.plot(iterations, self.objective_history, label="Objective")
        plt.xlabel("Iteration")
        plt.ylabel("Objective")
        plt.title("Objective vs Iterations")
        plt.legend()

        plt.subplot(3, 2, 2)
        plt.plot(iterations, self.data_fidelity_history, label="Data Fidelity")
        plt.plot(iterations, self.regularizer_history, label="Regularizer")
        plt.xlabel("Iteration")
        plt.ylabel("Cost")
        plt.title("Fidelity & Regularizer vs Iterations")
        plt.legend()

        plt.subplot(3, 2, 3)
        plt.plot(iterations, self.primal_resid_s_history, label="Primal Residual")
        plt.plot(iterations, self.dual_resid_s_history, label="Dual Residual")
        plt.xlabel("Iteration")
        plt.ylabel("Residual")
        plt.title("Residuals vs Iterations")
        plt.legend()

        plt.subplot(3, 2, 4)
        plt.plot(iterations, self.mu1_history, label="mu1")
        # plt.plot(iterations, self.mu2_history, label="mu2")
        # plt.plot(iterations, self.mu3_history, label="mu3")
        plt.xlabel("Iteration")
        plt.ylabel("Parameter value")
        plt.title("mu1 vs Iterations")
        plt.legend()
        
        plt.subplot(3, 2, 5)
        # plt.plot(iterations, self.mu1_history, label="mu1")
        plt.plot(iterations, self.mu2_history, label="mu2")
        # plt.plot(iterations, self.mu3_history, label="mu3")
        plt.xlabel("Iteration")
        plt.ylabel("Parameter value")
        plt.title("mu2 vs Iterations")
        plt.legend()
        
        plt.subplot(3, 2, 6)
        # plt.plot(iterations, self.mu1_history, label="mu1")
        # plt.plot(iterations, self.mu2_history, label="mu2")
        plt.plot(iterations, self.mu3_history, label="mu3")
        plt.xlabel("Iteration")
        plt.ylabel("Parameter value")
        plt.title("mu3 vs Iterations")
        plt.legend()

        plt.tight_layout()
        plt.show()


    # Image Processing Functions
    def img_resize(self, X, flag, f):
        if flag == "lateral":
            num = int(np.log2(f))
            if num ==0:
                pass
            else:
                for i in range(num):
                    # X = 0.25 * (X[::2, ::2, :, ...] + X[1::2, ::2, :, ...] + X[::2, 1::2, :, ...] + X[1::2, 1::2, :, ...])
                    X = 0.25 * (X[::2, ::2, ...] + X[1::2, ::2, ...] + X[::2, 1::2, ...] + X[1::2, 1::2, ...])
        elif flag == "axial":
            num = int(np.log2(f))
            if num ==0:
                pass
            else:
                for i in range(num):
                    X = 0.5 * (X[:, :, ::2, ...] + X[:, :, 1::2, ...])
        return X

    def l2norm(self, X):
        return torch.linalg.norm(X.ravel(), ord=2)

    # 불필요
    # def vec(self, X):
    #     return torch.reshape(X, (-1, 1))

    def norm_8bit(self, X):
        max_X = X.max()
        if max_X == 0:
            max_X = 1
        A = X / max_X * 255
        return A.astype('uint8')

    def norm_8bit_tensor(self, X):
        max_X = torch.max(X)
        if max_X == 0:
            max_X = torch.ones_like(max_X).to(self.device)
        X = (X / max_X * 255)
        return X.int()

    def projection_plot(self, data, label=0, show=True):
        # data = data[450:750, 2550:3150, :]
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

        if self.save_fig:
            plt.savefig('{}/{}/out_{}.png'.format(self.save_dir, self.dtstamp, label))

        if show:
            plt.show()
        return fig


if __name__ == "__main__":
    # Running the algorithm
    config = config_list
    A = ADMM3D(config)
    final_im = A.admm()
    A.draw_fig(final_im, A.max_iter)
