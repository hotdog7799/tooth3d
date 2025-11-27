import torch
import torch.fft as fft
import numpy as np
from .base_regularizer import BaseRegularizer

class AnisotropicDiffusionRegularizer(BaseRegularizer):
    """
    Anisotropic Diffusion Regularizer for tooth reconstruction.
    
    This regularizer preserves edges while smoothing surfaces by applying
    diffusion that adapts to local gradients. It's particularly good for
    tooth structures with smooth surfaces and sharp edges.
    """
    
    def __init__(self, device, tau=6e-4, tau_z=6e-10, edge_threshold=0.1, **kwargs):
        """
        Initialize anisotropic diffusion regularizer.
        
        Args:
            device: torch device
            tau: Base regularization weight for lateral directions
            tau_z: Base regularization weight for axial direction
            edge_threshold: Threshold for edge detection (higher = more edge preservation)
        """
        super().__init__(device, tau=tau, tau_z=tau_z, **kwargs)
        self.tau = tau
        self.tau_z = tau_z
        self.edge_threshold = edge_threshold
        self.diffusion_weights = None
    
    def _compute_diffusion_weights(self, x):
        """Compute anisotropic diffusion weights based on local gradients."""
        # Compute gradients
        grad_x = torch.diff(x, n=1, dim=0)
        grad_y = torch.diff(x, n=1, dim=1)
        grad_z = torch.diff(x, n=1, dim=2)
        
        # Compute gradient magnitudes
        grad_mag_x = torch.abs(grad_x)
        grad_mag_y = torch.abs(grad_y)
        grad_mag_z = torch.abs(grad_z)
        
        # Compute diffusion coefficients (lower where gradients are high)
        # g(∇I) = 1 / (1 + (|∇I|/K)²) where K is edge_threshold
        diff_coeff_x = 1.0 / (1.0 + (grad_mag_x / self.edge_threshold) ** 2)
        diff_coeff_y = 1.0 / (1.0 + (grad_mag_y / self.edge_threshold) ** 2)
        diff_coeff_z = 1.0 / (1.0 + (grad_mag_z / self.edge_threshold) ** 2)
        
        return diff_coeff_x, diff_coeff_y, diff_coeff_z
    
    def setup_variables(self, vk):
        """Setup anisotropic diffusion regularizer variables."""
        # Initialize dual variables for the three gradient directions
        eta_1 = vk[:-1, :, :].clone().detach()
        eta_2 = vk[:, :-1, :].clone().detach()
        eta_3 = vk[:, :, :-1].clone().detach()
        
        # Initialize gradient outputs
        uk1, uk2, uk3 = self.apply_Psi(vk)
        Lvk1 = uk1.clone().detach()
        Lvk2 = uk2.clone().detach()
        Lvk3 = uk3.clone().detach()
        
        # Initialize diffusion weights
        self.diffusion_weights = self._compute_diffusion_weights(vk)
        
        return {
            'eta_1': eta_1,
            'eta_2': eta_2,
            'eta_3': eta_3,
            'Lvk1': Lvk1,
            'Lvk2': Lvk2,
            'Lvk3': Lvk3
        }
    
    def compute_PsiTPsi(self, vk):
        """Compute adaptive Laplacian operator."""
        # Use average diffusion coefficient for the operator
        if self.diffusion_weights is not None:
            avg_diff_x, avg_diff_y, avg_diff_z = self.diffusion_weights
            avg_weight = (torch.mean(avg_diff_x) + torch.mean(avg_diff_y) + torch.mean(avg_diff_z)) / 3
        else:
            avg_weight = 1.0
        
        laplacian = vk.clone().detach()
        laplacian.zero_()
        laplacian[0, 0, 0] = 6 * avg_weight
        laplacian[0, 1, 0] = -1 * avg_weight
        laplacian[1, 0, 0] = -1 * avg_weight
        laplacian[0, 0, 1] = -1 * avg_weight
        laplacian[0, -1, 0] = -1 * avg_weight
        laplacian[-1, 0, 0] = -1 * avg_weight
        laplacian[0, 0, -1] = -1 * avg_weight
        
        return torch.abs(fft.fftn(laplacian))
    
    def apply_Psi(self, x):
        """Apply adaptive 3D gradient operator with diffusion weighting."""
        # Update diffusion weights
        self.diffusion_weights = self._compute_diffusion_weights(x)
        diff_x, diff_y, diff_z = self.diffusion_weights
        
        # Apply weighted gradients
        grad_x = -torch.diff(x, n=1, dim=0)
        grad_y = -torch.diff(x, n=1, dim=1)
        grad_z = -torch.diff(x, n=1, dim=2)
        
        # Weight by diffusion coefficients
        weighted_grad_x = grad_x * diff_x
        weighted_grad_y = grad_y * diff_y
        weighted_grad_z = grad_z * diff_z
        
        return (weighted_grad_x, weighted_grad_y, weighted_grad_z)
    
    def apply_PsiT(self, P1, P2, P3):
        """Apply transpose of adaptive gradient operator."""
        A = torch.cat([
            torch.unsqueeze(P1[0, :, :], dim=0), 
            torch.diff(P1, 1, 0), 
            -torch.unsqueeze(P1[-1, :, :], dim=0)
        ], dim=0)
        
        B = torch.cat([
            torch.unsqueeze(P2[:, 0, :], dim=1), 
            torch.diff(P2, 1, 1), 
            -torch.unsqueeze(P2[:, -1, :], dim=1)
        ], dim=1)
        
        C = torch.cat([
            torch.unsqueeze(P3[:, :, 0], dim=2), 
            torch.diff(P3, 1, 2), 
            -torch.unsqueeze(P3[:, :, -1], dim=2)
        ], dim=2)
        
        return A + B + C
    
    def soft_threshold(self, v, h, d, tau):
        """Apply adaptive soft thresholding based on local structure."""
        if isinstance(tau, (tuple, list)):
            tau_xy, tau_z = tau
        else:
            tau_xy = tau
            tau_z = self.tau_z / self.tau * tau if hasattr(self, 'tau') else tau
        
        # Get current diffusion weights
        if self.diffusion_weights is not None:
            diff_x, diff_y, diff_z = self.diffusion_weights
        else:
            diff_x = torch.ones_like(v)
            diff_y = torch.ones_like(h)
            diff_z = torch.ones_like(d)
        
        def size(x, i):
            return x.shape[i - 1]
        
        vararg_out = []
        
        if size(v, 1) != 0:
            # Compute magnitude for xy directions
            mag_xy = torch.sqrt(
                torch.cat([v, torch.zeros(1, size(v, 2), size(v, 3)).to(self.device)], dim=0) ** 2 +
                torch.cat([h, torch.zeros(size(h, 1), 1, size(h, 3)).to(self.device)], dim=1) ** 2
            )
            
            # Adaptive thresholding - use inverse of diffusion coefficient
            # Where diffusion is low (edges), use higher threshold
            adaptive_tau_xy = tau_xy / (torch.cat([diff_x, torch.zeros(1, size(v, 2), size(v, 3)).to(self.device)], dim=0) + 1e-6)
            
            magtmag_xy = self._soft_threshold_scalar(mag_xy, adaptive_tau_xy)
            mmult_xy = magtmag_xy / (mag_xy + torch.finfo(torch.float32).eps)
            mmult_xy = mmult_xy * (mag_xy > 0)
            mmult_xy = torch.nan_to_num(mmult_xy)
            
            # Compute magnitude for z direction
            mag_z = torch.sqrt(
                torch.cat([d, torch.zeros((size(d, 1), size(d, 2), 1)).to(self.device)], dim=2) ** 2
            )
            
            # Adaptive thresholding for z direction
            adaptive_tau_z = tau_z / (torch.cat([diff_z, torch.zeros((size(d, 1), size(d, 2), 1)).to(self.device)], dim=2) + 1e-6)
            
            magtmag_z = self._soft_threshold_scalar(mag_z, adaptive_tau_z)
            mmult_z = magtmag_z / (mag_z + torch.finfo(torch.float32).eps)
            mmult_z = mmult_z * (mag_z > 0)
            mmult_z = torch.nan_to_num(mmult_z)
            
            # Apply multipliers
            vararg_out.append(v * mmult_xy[:-1, :, :])
            vararg_out.append(h * mmult_xy[:, :-1, :])
            vararg_out.append(d * mmult_z[:, :, :-1])
        
        return vararg_out
    
    def _soft_threshold_scalar(self, X, tau_c):
        """Apply scalar soft thresholding."""
        return torch.sign(X) * torch.maximum(torch.zeros_like(X), torch.abs(X) - tau_c)
    
    def update_dual_variables(self, reg_vars, Lvk_outputs, uk_outputs, mu2):
        """Update dual variables."""
        Lvk1, Lvk2, Lvk3 = Lvk_outputs
        uk1, uk2, uk3 = uk_outputs
        
        # Compute residuals
        r_su_1 = Lvk1 - uk1
        r_su_2 = Lvk2 - uk2
        r_su_3 = Lvk3 - uk3
        
        # Update dual variables
        reg_vars['eta_1'] = reg_vars['eta_1'] + mu2 * r_su_1
        reg_vars['eta_2'] = reg_vars['eta_2'] + mu2 * r_su_2
        reg_vars['eta_3'] = reg_vars['eta_3'] + mu2 * r_su_3
        
        return reg_vars
    
    def compute_residuals(self, Lvk_old, Lvk_new, uk_outputs, mu2):
        """Compute primal and dual residuals."""
        Lvk1_old, Lvk2_old, Lvk3_old = Lvk_old
        Lvk1, Lvk2, Lvk3 = Lvk_new
        uk1, uk2, uk3 = uk_outputs
        
        # Dual residual
        dual_resid = mu2 * torch.sqrt(
            self.l2norm(Lvk1_old - Lvk1) ** 2 +
            self.l2norm(Lvk2_old - Lvk2) ** 2 +
            self.l2norm(Lvk3_old - Lvk3) ** 2
        )
        
        # Primal residual
        primal_resid = torch.sqrt(
            self.l2norm(Lvk1 - uk1) ** 2 +
            self.l2norm(Lvk2 - uk2) ** 2 +
            self.l2norm(Lvk3 - uk3) ** 2
        )
        
        return dual_resid, primal_resid
    
    def compute_penalty(self, Lvk_outputs):
        """Compute adaptive penalty based on local structure."""
        Lvk1, Lvk2, Lvk3 = Lvk_outputs
        
        # Use diffusion weights to compute adaptive penalty
        if self.diffusion_weights is not None:
            diff_x, diff_y, diff_z = self.diffusion_weights
            
            # Penalty is weighted by diffusion coefficients
            penalty = (
                torch.sum(diff_x * torch.abs(Lvk1)) * self.tau +
                torch.sum(diff_y * torch.abs(Lvk2)) * self.tau +
                torch.sum(diff_z * torch.abs(Lvk3)) * self.tau_z
            )
        else:
            penalty = self.tau * (
                torch.sum(torch.abs(Lvk1)) +
                torch.sum(torch.abs(Lvk2)) +
                torch.sum(torch.abs(Lvk3))
            )
        
        return penalty 