# -*- coding:utf-8 -*-
"""
Refactored ADMM 3D Solver for Lensless Imaging

This module provides a modular ADMM solver that separates the core optimization
algorithm from the regularization terms. This design allows easy swapping of
different regularizers (3D TV, L1, etc.) without modifying the main solver.

Key improvements:
- Modular regularizer system
- Cleaner separation of concerns
- Easier to extend with new regularizers
- Better maintainability
"""

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

# Import regularizers
from regularizers import BaseRegularizer, TV3DRegularizer


class ADMM3D:
    """
    Modular ADMM Solver for 3D reconstruction in lensless imaging.

    This solver separates the core ADMM algorithm from regularization terms,
    allowing easy experimentation with different regularizers.
    """

    def __init__(self, config_list, regularizer=None):
        """
        Initialize ADMM solver with configuration and regularizer.

        Args:
            config_list: Dictionary containing solver configuration
            regularizer: Instance of BaseRegularizer (if None, uses default 3D TV)
        """
        # --- 기존 config 복사 로직 유지 ---
        self.target_resize_h = None  # 리사이징 목표 H 초기화
        self.target_resize_w = None  # 리사이징 목표 W 초기화

        # Copy config to instance variables
        for con, val in config_list.items():
            try:
                setattr(self, con, val)
                print("{} = {}".format(con, val))
            except:
                print("Parameter: {} setup error".format(con))

        # Create timestamp
        self.dtstamp = (
            time.strftime("%m%d_%H%M%S", time.localtime(time.time()))
            + "_"
            + self.color_to_process
            + "_tau"
            + str(self.tau)
            + "_ztau"
            + str(self.tau_z)
        )

        # GPU setup
        if torch.cuda.is_available():
            print("Device:{} selected".format(self.numGPU))
            self.device = torch.device("cuda:{}".format(self.numGPU))
        else:
            print("GPU not available")
            self.device = torch.device("cpu")
        print(self.device)

        # Setup regularizer
        if regularizer is None:
            self.regularizer = TV3DRegularizer(
                device=self.device, tau=self.tau, tau_z=self.tau_z
            )
        else:
            if not isinstance(regularizer, BaseRegularizer):
                raise ValueError("Regularizer must inherit from BaseRegularizer")
            self.regularizer = regularizer

        print(f"Using regularizer: {type(self.regularizer).__name__}")

        # Path setup
        self.cwd = os.getcwd()
        self.homed = os.path.expanduser("~")
        if self.path_ref:
            self.psf_file = self.homed + self.psf_file
            self.img_file = self.homed + self.img_file

        # Initialize data and folders
        self.init_data()
        self.setup_folder()
        self.save_path = os.path.join(self.save_dir, self.dtstamp)

        # Tracking variables
        self.objective_history = []
        self.data_fidelity_history = []
        self.regularizer_history = []
        self.primal_resid_s_history = []
        self.dual_resid_s_history = []
        self.mu1_history = []
        self.mu2_history = []
        self.mu3_history = []

    def init_data(self):
        """Load and preprocess PSF and raw image data."""
        # Load PSF
        psf = io.loadmat(self.psf_file, mat_dtype=True)["psf_stack"]
        psf = psf.astype("float32")
        # --- PSF Permute (shape 확인 후 D, H, W 순서로) ---
        if (
            psf.ndim == 3 and psf.shape[0] != psf.shape[1]
        ):  # Assume (D, H, W) or (H, W, D)
            if (
                psf.shape[0] < psf.shape[1] and psf.shape[0] < psf.shape[2]
            ):  # If D is likely first dim
                print("Assuming PSF is already (D, H, W).")
            elif (
                psf.shape[2] < psf.shape[0] and psf.shape[2] < psf.shape[1]
            ):  # If D is likely last dim
                print("Permuting PSF stack from (H, W, D) to (D, H, W)...")
                psf = np.transpose(psf, (2, 0, 1))
            else:
                print(
                    f"Warning: Could not reliably determine PSF dimension order from shape {psf.shape}. Assuming (D, H, W)."
                )
        elif psf.ndim != 3:
            raise ValueError(f"Unexpected PSF dimension: {psf.ndim}")
        print(
            f"Permuted PSF stack shape: {psf.shape}"
        )  # Should be (num_slices, H_orig, W_orig)
        psf = psf - self.psf_bias
        psf[psf < 0] = 0
        psf = psf / LA.norm(psf)
        psf = self.img_resize(psf, "lateral", self.lateral_downsample)
        psf = self.img_resize(psf, "axial", self.axial_downsample)
        print("psf shape: ", psf.shape)
        # Load raw image
        raw = cv2.imread(self.img_file, flags=cv2.IMREAD_UNCHANGED)
        raw = np.array(raw, dtype="float32")

        if len(raw.shape) == 3:
            if self.color_to_process.lower() in ["mono", "red", "green", "blue"]:
                if self.color_to_process.lower() == "mono":
                    raw = cv2.cvtColor(raw, cv2.COLOR_RGB2GRAY)
                else:
                    raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
                    color_list = {"red": 0, "green": 1, "blue": 2}
                    raw = raw[:, :, color_list[self.color_to_process.lower()]]
            elif self.color_to_process.lower() in ["rg", "rgb"]:
                raw = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)

        raw = raw - self.raw_bias
        raw[raw < 0] = 0
        # --- 내부 리사이징 수행 ---
        if self.target_resize_h is not None and self.target_resize_w is not None:
            target_h, target_w = self.target_resize_h, self.target_resize_w
            print(f"Resizing PSF stack internally to {target_h}x{target_w}...")
            num_slices_psf = psf.shape[0]
            psf_resized = np.zeros(
                (num_slices_psf, target_h, target_w), dtype=np.float32
            )
            for i in range(num_slices_psf):
                psf_resized[i] = cv2.resize(
                    psf[i], (target_w, target_h), interpolation=cv2.INTER_LINEAR
                )
            psf = psf_resized  # Update psf to resized version

            print(f"Resizing raw image internally to {target_h}x{target_w}...")
            raw_resized = cv2.resize(
                raw, (target_w, target_h), interpolation=cv2.INTER_LINEAR
            )
            raw = raw_resized  # Update raw to resized version
        else:
            print(
                "No target resize dimensions specified, using original/downsampled size."
            )
            # --- 리사이징 안 할 경우, 기존 Downsampling 로직 적용 ---
            psf = self.img_resize(psf, "lateral", self.lateral_downsample)
            psf = self.img_resize(psf, "axial", self.axial_downsample)
            raw = self.img_resize(raw, "lateral", self.lateral_downsample)
        raw = self.img_resize(raw, "lateral", self.lateral_downsample)
        raw = raw / np.max(raw)
        print("raw shape: ", raw.shape)

        # Get dimensions
        [self.Nx, self.Ny, self.Nz] = psf.shape

        # Convert to tensors
        self.psf = torch.from_numpy(psf).to(self.device)
        if len(raw.shape) == 2:
            self.raw = torch.from_numpy(raw).to(self.device)
        else:
            self.raw = torch.from_numpy(raw).to(self.device)

        return psf, raw

    def setup_folder(self):
        """Setup save directory."""
        if self.save_dir.endswith("/"):
            self.save_dir = self.save_dir[:-1]
        save_path = os.path.join(self.save_dir, self.dtstamp)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    def admm(self):
        """Main ADMM reconstruction method."""
        self.st = time.time()

        if self.color_to_process.lower() in ["red", "green", "blue", "mono"]:
            final_im = self.admm_solver(self.psf, self.raw)
        elif self.color_to_process.lower() in ["rg", "rgb"]:
            channels_to_process = list(self.color_to_process.lower())
            results = []
            color_map = {"r": 0, "g": 1, "b": 2}

            for c in channels_to_process:
                print("Reconstructing channel {}...".format(c))

                if len(self.raw.shape) == 3:
                    raw_channel = self.raw[:, :, color_map[c]]
                else:
                    raise ValueError("Raw image does not have multiple channels.")

                if self.psf.ndim == 4:
                    psf_channel = self.psf[:, :, :, color_map[c]]
                else:
                    psf_channel = self.psf

                xhat_channel = self.admm_solver(psf_channel, raw_channel)
                results.append(xhat_channel)

            final_im = np.stack(results, axis=-1)
            global_min = final_im.min()
            global_max = final_im.max()
            final_im = (final_im - global_min) / (global_max - global_min) * 255
            final_im = final_im.astype(np.uint8)
        else:
            print("Unknown color_to_process. Using mono reconstruction.")
            final_im = self.admm_solver(self.psf, self.raw)

        print("Total elapsed Time: {}".format(np.round(time.time() - self.st, 2)))
        print("Reconstruction Finished!")

        io.savemat(
            "{}/{}/final_data_uncropped.mat".format(self.save_dir, self.dtstamp),
            {"final": final_im},
            format="5",
            do_compression=True,
        )

        # Generate final visualizations
        print("\nGenerating final visualizations...")
        should_display_final = getattr(self, "show_figs", True)
        should_save_final = self.save_fig

        if should_save_final or should_display_final:
            # Final slice visualization
            self.draw_fig(
                final_im, "FINAL", display=should_display_final, save=should_save_final
            )
            # Final projection visualization
            self.projection_plot(final_im, label="FINAL", show=should_display_final)

        return final_im

    def admm_solver(self, psf, b):
        """Core ADMM solver with modular regularizer support."""
        # PSF preprocessing
        psf = psf / torch.linalg.norm(psf.ravel())
        shift = int(self.Nz // 2)
        psf = torch.roll(torch.flip(psf, dims=[2]), shift, dims=2)

        # Forward model setup
        self.Hs = fft.fftn(fft.ifftshift(self.pad2d(psf)))
        self.Hs_conj = torch.conj(self.Hs)
        self.HtH = torch.abs(self.Hs * self.Hs_conj)

        # Initialize variables
        vk = torch.zeros_like(self.Hs, dtype=torch.float32)
        xi = vk.clone().detach()
        rho = vk.clone().detach()
        Dtb = self.pad2d(b)

        # Setup regularizer
        reg_vars = self.regularizer.setup_variables(vk)
        PsiTPsi = self.regularizer.compute_PsiTPsi(vk)

        # Update denominators
        v_mult = 1 / (self.mu1 * self.HtH + self.mu2 * PsiTPsi + self.mu3)
        DtD = self.pad2d(torch.ones_like(b))
        nu_mult = 1 / (DtD + self.mu1)

        # Initialize iteration variables
        Hvkp = torch.zeros_like(vk)
        vkp = torch.zeros_like(vk)

        # Metrics
        dual_resid_s = np.zeros(self.max_iter)
        primal_resid_s = np.zeros(self.max_iter)
        dual_resid_u = np.zeros(self.max_iter)
        primal_resid_u = np.zeros(self.max_iter)
        dual_resid_w = np.zeros(self.max_iter)
        primal_resid_w = np.zeros(self.max_iter)
        objective = np.zeros(self.max_iter)
        data_fidelity = np.zeros(self.max_iter)
        regularizer_penalty = np.zeros(self.max_iter)

        # Main iteration loop
        n = 0
        while n < self.max_iter:
            Hvk = Hvkp.clone().detach()

            # Nu update
            nukp = nu_mult * (self.mu1 * (xi / self.mu1 + Hvk) + Dtb)

            # w update
            wkp = torch.maximum(rho / self.mu3 + vk, torch.zeros_like(vk))

            # Regularizer soft-thresholding - handle different regularizer types
            if hasattr(self.regularizer, "tau_z"):  # 3D TV-style regularizer
                inputs_for_threshold = [
                    reg_vars["Lvk1"] + reg_vars["eta_1"] / self.mu2,
                    reg_vars["Lvk2"] + reg_vars["eta_2"] / self.mu2,
                    reg_vars["Lvk3"] + reg_vars["eta_3"] / self.mu2,
                ]
                tau_scaled = self.regularizer.tau / self.mu2
                tau_z_scaled = self.regularizer.tau_z / self.mu2
                uk_outputs = self.regularizer.soft_threshold(
                    *inputs_for_threshold, tau=(tau_scaled, tau_z_scaled)
                )
            else:  # L1-style regularizer
                inputs_for_threshold = reg_vars["Lvk"] + reg_vars["eta"] / self.mu2
                tau_scaled = self.regularizer.tau / self.mu2
                uk_outputs = self.regularizer.soft_threshold(
                    inputs_for_threshold, tau=tau_scaled
                )

            # v update - handle different regularizer types
            if hasattr(self.regularizer, "tau_z"):  # 3D TV-style regularizer
                vkp_numerator = (
                    self.mu3 * (wkp - rho / self.mu3)
                    + self.mu2
                    * self.regularizer.apply_PsiT(
                        uk_outputs[0] - reg_vars["eta_1"] / self.mu2,
                        uk_outputs[1] - reg_vars["eta_2"] / self.mu2,
                        uk_outputs[2] - reg_vars["eta_3"] / self.mu2,
                    )
                    + self.mu1 * self.Hadj(nukp - xi / self.mu1)
                )
            else:  # L1-style regularizer
                vkp_numerator = (
                    self.mu3 * (wkp - rho / self.mu3)
                    + self.mu2
                    * self.regularizer.apply_PsiT(
                        uk_outputs[0] - reg_vars["eta"] / self.mu2
                    )
                    + self.mu1 * self.Hadj(nukp - xi / self.mu1)
                )
            vkp = torch.real(fft.ifftn(v_mult * fft.fftn(vkp_numerator)))

            # Update Hvkp
            Hvkp = torch.real(fft.ifftn(self.Hs * fft.fftn(vkp)))

            # Residuals for data fidelity
            r_sv = Hvkp - nukp
            xi = xi + self.mu1 * r_sv
            dual_resid_s[n] = self.mu1 * self.l2norm(Hvk - Hvkp)
            primal_resid_s[n] = self.l2norm(r_sv)
            data_fidelity[n] = 0.5 * self.l2norm(self.crop3d(Hvkp) - b) ** 2

            # Regularizer update - handle different regularizer types
            if hasattr(self.regularizer, "tau_z"):  # 3D TV-style regularizer
                Lvk_old = (
                    reg_vars["Lvk1"].clone(),
                    reg_vars["Lvk2"].clone(),
                    reg_vars["Lvk3"].clone(),
                )
                Lvk_new = self.regularizer.apply_Psi(vkp)
                reg_vars["Lvk1"], reg_vars["Lvk2"], reg_vars["Lvk3"] = Lvk_new
            else:  # L1-style regularizer
                Lvk_old = (reg_vars["Lvk"].clone(),)
                Lvk_new = self.regularizer.apply_Psi(vkp)
                reg_vars["Lvk"] = Lvk_new[0]

            # Update regularizer dual variables
            reg_vars = self.regularizer.update_dual_variables(
                reg_vars, Lvk_new, uk_outputs, self.mu2
            )

            # Compute regularizer residuals
            dual_resid_u[n], primal_resid_u[n] = self.regularizer.compute_residuals(
                Lvk_old, Lvk_new, uk_outputs, self.mu2
            )

            regularizer_penalty[n] = self.regularizer.compute_penalty(Lvk_new)
            objective[n] = data_fidelity[n] + regularizer_penalty[n]

            # Nonnegativity constraint
            r_sw = vkp - wkp
            rho = rho + self.mu3 * r_sw
            dual_resid_w[n] = self.mu3 * self.l2norm(vk - vkp)
            primal_resid_w[n] = self.l2norm(vkp - wkp)

            # Parameter auto-tuning
            if self.autotune:
                [self.mu1, mu1_update] = self.update_param(
                    self.mu1, primal_resid_s[n], dual_resid_s[n]
                )
                [self.mu2, mu2_update] = self.update_param(
                    self.mu2, primal_resid_u[n], dual_resid_u[n]
                )
                [self.mu3, mu3_update] = self.update_param(
                    self.mu3, primal_resid_w[n], dual_resid_w[n]
                )

                if mu1_update or mu2_update or mu3_update:
                    v_mult = 1 / (self.mu1 * self.HtH + self.mu2 * PsiTPsi + self.mu3)
                    nu_mult = 1 / (DtD + self.mu1)

            vk = vkp

            # Record metrics
            self.objective_history.append(objective[n])
            self.data_fidelity_history.append(data_fidelity[n])
            self.regularizer_history.append(regularizer_penalty[n])
            self.primal_resid_s_history.append(primal_resid_s[n])
            self.dual_resid_s_history.append(dual_resid_s[n])
            self.mu1_history.append(self.mu1)
            self.mu2_history.append(self.mu2)
            self.mu3_history.append(self.mu3)

            # Print status
            if self.print_interval and ((n + 1) % self.print_interval == 0 or n == 0):
                print(
                    "iter:{} time:{:.3f} cost:{:.3f} data_fidelity:{:.3f} regularizer_penalty:{:.3f}".format(
                        n,
                        np.round(time.time() - self.st, 2),
                        objective[n],
                        data_fidelity[n],
                        regularizer_penalty[n],
                    )
                )

            # Display and/or save figures if required
            should_process_figs = (
                ((n + 1) % self.disp_figs == 0 or n == 0) if self.disp_figs else False
            )
            should_display_figs = (
                self.disp_figs
                and should_process_figs
                and getattr(self, "show_figs", True)
            )
            should_save_figs = self.save_fig and should_process_figs

            if should_save_figs or should_display_figs:
                if self.useGPU:
                    out = vkp.to("cpu").numpy()
                    proj = vkp.to("cpu").numpy()
                    # Use new display/save logic
                    self.draw_fig(
                        out, n, display=should_display_figs, save=should_save_figs
                    )
                    if should_display_figs:
                        self.projection_plot(proj, label=n, show=True)
                    elif should_save_figs:
                        self.projection_plot(proj, label=n, show=False)

            if self.useGPU:
                vkp = vkp.to("cpu").numpy()
            n = n + 1

        return vkp

    # Utility methods
    def pad2d(self, x):
        v_pad = int(np.floor(self.Nx / 2))
        h_pad = int(np.floor(self.Ny / 2))
        if x.ndim == 3:
            return f.pad(x, (0, 0, h_pad, h_pad, v_pad, v_pad))
        else:
            tmp = torch.unsqueeze(x, dim=2)
            return f.pad(
                tmp, (0, self.Nz - 1, h_pad, h_pad, v_pad, v_pad), "constant", 0
            )

    def crop2d(self, x):
        v_crop = int(np.floor(self.Nx / 2))
        h_crop = int(np.floor(self.Ny / 2))
        v, h = x.shape[:2]
        return x[v_crop : v - v_crop, h_crop : h - h_crop]

    def crop3d(self, x):
        return self.crop2d(x[:, :, 0])

    def Hadj(self, x):
        return torch.real(fft.ifftn(self.Hs_conj * fft.fftn(x)))

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

    def l2norm(self, X):
        return torch.linalg.norm(X.ravel(), ord=2)

    def img_resize(self, X, flag, f):
        if flag == "lateral":
            num = int(np.log2(f))
            if num == 0:
                pass
            else:
                for i in range(num):
                    X = 0.25 * (
                        X[::2, ::2, ...]
                        + X[1::2, ::2, ...]
                        + X[::2, 1::2, ...]
                        + X[1::2, 1::2, ...]
                    )
        elif flag == "axial":
            num = int(np.log2(f))
            if num == 0:
                pass
            else:
                for i in range(num):
                    X = 0.5 * (X[:, :, ::2, ...] + X[:, :, 1::2, ...])
        return X

    def draw_fig(self, vk, n, display=True, save=True):
        """
        Draw reconstruction figures with separated display and save logic.

        Args:
            vk: reconstruction volume
            n: iteration number
            display: whether to display the figures (default: True)
            save: whether to save the figures (default: True)
        """
        import os
        import matplotlib
        import matplotlib.pyplot as plt
        import numpy as np

        # Set matplotlib to non-interactive mode if not displaying
        if not display:
            # Force matplotlib to use non-interactive backend
            plt.ioff()  # Turn off interactive mode
            original_backend = matplotlib.get_backend()
            matplotlib.use("Agg")  # Non-interactive backend

        # 1. Global normalization
        vk = vk.astype(np.float32)
        global_min = vk.min()
        global_max = vk.max()
        if global_max - global_min != 0:
            norm_v = (vk - global_min) / (global_max - global_min)
        else:
            norm_v = vk

        # 2. Crop each slice after reconstruction.
        #    Crop parameters: center at (x,y)=(571,320), side length = 120 pixels.
        crop_center_x = 571  # x coordinate (column)
        crop_center_y = 320  # y coordinate (row)
        half_crop = 60  # 120/2

        def crop_img(img):
            # img is 2D array (height, width)
            return img[
                crop_center_y - half_crop : crop_center_y + half_crop,
                crop_center_x - half_crop : crop_center_x + half_crop,
            ]

        # norm_v shape: (H, W, D) for mono channel
        H, W, D = norm_v.shape
        cropped_stack = np.empty((half_crop * 2, half_crop * 2, D), dtype=norm_v.dtype)
        for i in range(D):
            cropped_stack[:, :, i] = crop_img(norm_v[:, :, i])

        # 3. Plot all slices in a grid.
        n_slices = cropped_stack.shape[2]
        n_cols = int(np.ceil(np.sqrt(n_slices)))
        n_rows = int(np.ceil(n_slices / n_cols))

        # Create figure - force non-interactive mode if not displaying
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))

        # Handle different axes configurations
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1 or n_cols == 1:
            axes = np.atleast_2d(axes)

        for i in range(n_slices):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].imshow(cropped_stack[:, :, i], cmap="gray", vmin=0, vmax=1)

            # Use psf_labels if available, otherwise use slice number
            if (
                hasattr(self, "psf_labels")
                and self.psf_labels is not None
                and i < len(self.psf_labels)
            ):
                axes[row, col].set_title("{}".format(self.psf_labels[i]))
            else:
                axes[row, col].set_title("Slice {}".format(i))
            axes[row, col].axis("off")

        # Hide empty subplots
        total_subplots = n_rows * n_cols
        for j in range(n_slices, total_subplots):
            row = j // n_cols
            col = j % n_cols
            axes[row, col].axis("off")

        plt.suptitle("Iteration {}".format(n))
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Save the figure if required
        if save:
            iter_folder = os.path.join(self.save_dir, self.dtstamp)
            if not os.path.exists(iter_folder):
                os.makedirs(iter_folder)
            overall_filename = os.path.join(iter_folder, "slices_iter_{}.png".format(n))
            plt.savefig(overall_filename, dpi=150, bbox_inches="tight")
            print(f"✓ Saved reconstruction figure: {overall_filename}")

        # Display the figure if required
        if display:
            plt.ion()  # Turn on interactive mode for display
            plt.show(block=False)
            plt.pause(0.1)
        else:
            # Immediately close the figure to prevent any display
            plt.close(fig)

        # 5. Save individual slice images without colorbar or title if required
        if save:
            slice_folder = os.path.join(
                self.save_dir, self.dtstamp, "iter_{}_slices".format(n)
            )
            if not os.path.exists(slice_folder):
                os.makedirs(slice_folder)
            for i in range(n_slices):
                if (
                    hasattr(self, "psf_labels")
                    and self.psf_labels is not None
                    and i < len(self.psf_labels)
                ):
                    filename = "{}.png".format(self.psf_labels[i])
                else:
                    filename = "slice_{}.png".format(i)

                # Save individual slice
                plt.imsave(
                    os.path.join(slice_folder, filename),
                    cropped_stack[:, :, i],
                    cmap="gray",
                    vmin=0,
                    vmax=1,
                )
            print(f"✓ Saved {n_slices} individual slices to: {slice_folder}")

        # Restore interactive mode if we turned it off
        if not display:
            plt.ion()  # Restore interactive mode
            matplotlib.use(original_backend)  # Restore original backend

    def projection_plot(self, data, label=0, show=True):
        """
        Create 3D projection plots (xy, xz, yz).

        Args:
            data: 3D reconstruction volume
            label: iteration number for labeling
            show: whether to display the plot (default: True)
        """
        import matplotlib
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        # Set non-interactive mode if not showing
        if not show:
            plt.ioff()
            original_backend = matplotlib.get_backend()
            matplotlib.use("Agg")

        # Create projections
        import numpy as np

        dataxy = np.sum(data, axis=2)  # Sum along z-axis
        dataxz = np.sum(data, axis=1)  # Sum along y-axis
        datayz = np.sum(data, axis=0)  # Sum along x-axis

        fig, axes = plt.subplots(figsize=(17, 4), nrows=1, ncols=3)

        # XY projection
        ax0 = axes[0].imshow(dataxy, cmap="gray")
        fig.colorbar(
            ax0,
            cax=make_axes_locatable(axes[0]).append_axes("right", size="5%", pad=0.05),
            orientation="vertical",
        )
        axes[0].set_title("XY projection")
        axes[0].set_ylabel("Y")
        axes[0].set_xlabel("X")

        # XZ projection
        ax1 = axes[1].imshow(dataxz, cmap="gray")
        fig.colorbar(
            ax1,
            cax=make_axes_locatable(axes[1]).append_axes("right", size="5%", pad=0.05),
            orientation="vertical",
        )
        axes[1].set_title("XZ projection")
        axes[1].set_ylabel("X")
        axes[1].set_xlabel("Z")

        # YZ projection
        ax2 = axes[2].imshow(datayz, cmap="gray")
        fig.colorbar(
            ax2,
            cax=make_axes_locatable(axes[2]).append_axes("right", size="5%", pad=0.05),
            orientation="vertical",
        )
        axes[2].set_title("YZ projection")
        axes[2].set_ylabel("Y")
        axes[2].set_xlabel("Z")

        for ax in axes:
            ax.set_aspect(aspect="auto")

        plt.suptitle("3D Projections - Iteration {}".format(label))

        # Save first if required
        if self.save_fig:
            save_path = "{}/{}/projections_iter_{}.png".format(
                self.save_dir, self.dtstamp, label
            )
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"✓ Saved projection plot: {save_path}")

        # Then show if required
        if show:
            plt.ion()  # Turn on interactive mode for display
            plt.show(block=False)
            plt.pause(0.1)
        else:
            # Close immediately to prevent display
            plt.close(fig)

        # Restore interactive mode if we turned it off
        if not show:
            plt.ion()
            matplotlib.use(original_backend)

        return fig

    def plot_iteration_metrics(self):
        """Plot convergence metrics."""
        iterations = np.arange(len(self.objective_history))
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        plt.plot(iterations, self.objective_history)
        plt.title("Objective")
        plt.xlabel("Iteration")

        plt.subplot(2, 2, 2)
        plt.plot(iterations, self.data_fidelity_history, label="Data Fidelity")
        plt.plot(iterations, self.regularizer_history, label="Regularizer")
        plt.title("Cost Components")
        plt.xlabel("Iteration")
        plt.legend()

        plt.subplot(2, 2, 3)
        plt.plot(iterations, self.primal_resid_s_history, label="Primal")
        plt.plot(iterations, self.dual_resid_s_history, label="Dual")
        plt.title("Residuals")
        plt.xlabel("Iteration")
        plt.legend()

        plt.subplot(2, 2, 4)
        plt.plot(iterations, self.mu1_history, label="mu1")
        plt.plot(iterations, self.mu2_history, label="mu2")
        plt.plot(iterations, self.mu3_history, label="mu3")
        plt.title("Parameters")
        plt.xlabel("Iteration")
        plt.legend()

        plt.tight_layout()
        plt.show()


# # Example usage
# if __name__ == "__main__":
#     config_list = {
#         'path_ref': 0,
#         'psf_file': "./data/psf_stack.mat",
#         'img_file': "./data/raw_image.tiff",
#         'save_dir': "./results/",
#         'save_fig': True,
#         'color_to_process': "mono",
#         'psf_bias': 0,
#         'raw_bias': 600,
#         'lateral_downsample': 8,
#         'axial_downsample': 1,
#         'start_z': 0,
#         'end_z': 0,
#         'useGPU': True,
#         'numGPU': 0,
#         'max_iter': 1000,
#         'print_interval': 200,
#         'disp_figs': 200,
#         'mu1': 1,
#         'mu2': 1,
#         'mu3': 1,
#         'tau': 6.0e-4,
#         'tau_z': 6.0e-10,
#         'autotune': 1,
#         'mu_inc': 1.1,
#         'mu_dec': 1.1,
#         'resid_tol': 1.5,
#     }

#     # Use default regularizer
#     solver = ADMM3D(config_list)
#     final_im = solver.admm()
#     solver.plot_iteration_metrics()
