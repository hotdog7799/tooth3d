import torch
import torch.fft as fft
import numpy as np
from .base_regularizer import BaseRegularizer

class CenterWeightedRegularizer(BaseRegularizer):
    """
    Center-Weighted Regularizer for tooth reconstruction.
    
    This regularizer applies stronger regularization to edge regions
    while preserving the center where the tooth is located. It combines
    3D TV with spatially-varying weights.
    """
    
    def __init__(self, device, tau=6e-4, tau_z=6e-10, center_weight=0.1, edge_weight=1.0, **kwargs):
        """
        Initialize center-weighted regularizer.
        
        Args:
            device: torch device
            tau: Base regularization weight for lateral directions
            tau_z: Base regularization weight for axial direction
            center_weight: Weight for center region (lower = less regularization)
            edge_weight: Weight for edge region (higher = more regularization)
        """
        super().__init__(device, tau=tau, tau_z=tau_z, **kwargs)
        self.tau = tau
        self.tau_z = tau_z
        self.center_weight = center_weight
        self.edge_weight = edge_weight
        self.weight_mask = None
    
    def _create_weight_mask(self, shape):
        """Create spatial weight mask with center bias."""
        if self.weight_mask is not None and self.weight_mask.shape == shape:
            return self.weight_mask
        
        h, w, d = shape
        
        # Create 2D radial mask for lateral directions
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        
        # Distance from center
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_distance = np.sqrt(center_x**2 + center_y**2)
        
        # Normalized distance (0 at center, 1 at edges)
        normalized_distance = distance / max_distance
        
        # Create weight map (lower at center, higher at edges)
        weight_2d = self.center_weight + (self.edge_weight - self.center_weight) * normalized_distance
        
        # Extend to 3D
        weight_3d = np.tile(weight_2d[:, :, np.newaxis], (1, 1, d))
        
        self.weight_mask = torch.from_numpy(weight_3d).float().to(self.device)
        return self.weight_mask
    
    def setup_variables(self, vk):
        """Setup center-weighted regularizer variables."""
        # Create weight mask
        self._create_weight_mask(vk.shape)
        
        # Initialize dual variables for the three gradient directions
        eta_1 = vk[:-1, :, :].clone().detach()
        eta_2 = vk[:, :-1, :].clone().detach()
        eta_3 = vk[:, :, :-1].clone().detach()
        
        # Initialize gradient outputs
        uk1, uk2, uk3 = self.apply_Psi(vk)
        Lvk1 = uk1.clone().detach()
        Lvk2 = uk2.clone().detach()
        Lvk3 = uk3.clone().detach()
        
        return {
            'eta_1': eta_1,
            'eta_2': eta_2,
            'eta_3': eta_3,
            'Lvk1': Lvk1,
            'Lvk2': Lvk2,
            'Lvk3': Lvk3
        }
    
    def compute_PsiTPsi(self, vk):
        """Compute weighted 3D Laplacian operator."""
        # Use average weight for the operator
        avg_weight = torch.mean(self.weight_mask)
        
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
        """Apply 3D gradient operator."""
        return (-torch.diff(x, n=1, dim=0), 
                -torch.diff(x, n=1, dim=1), 
                -torch.diff(x, n=1, dim=2))
    
    def apply_PsiT(self, P1, P2, P3):
        """Apply transpose of 3D gradient operator."""
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
        """Apply spatially-weighted soft thresholding."""
        if isinstance(tau, (tuple, list)):
            tau_xy, tau_z = tau
        else:
            tau_xy = tau
            tau_z = self.tau_z / self.tau * tau if hasattr(self, 'tau') else tau
        
        # Get weight masks for each gradient direction
        weight_mask = self._create_weight_mask((v.shape[0]+1, v.shape[1], v.shape[2]))
        weight_v = weight_mask[:-1, :, :]  # x-direction
        weight_h = weight_mask[:, :-1, :]  # y-direction
        weight_d = weight_mask[:, :, :-1]  # z-direction
        
        def size(x, i):
            return x.shape[i - 1]
        
        vararg_out = []
        
        if size(v, 1) != 0:
            # Compute magnitude for xy directions
            mag_xy = torch.sqrt(
                torch.cat([v, torch.zeros(1, size(v, 2), size(v, 3)).to(self.device)], dim=0) ** 2 +
                torch.cat([h, torch.zeros(size(h, 1), 1, size(h, 3)).to(self.device)], dim=1) ** 2
            )
            
            # Apply spatial weighting to threshold
            weighted_tau_xy = tau_xy * weight_mask
            magtmag_xy = self._soft_threshold_scalar(mag_xy, weighted_tau_xy)
            
            mmult_xy = magtmag_xy / (mag_xy + torch.finfo(torch.float32).eps)
            mmult_xy = mmult_xy * (mag_xy > 0)
            mmult_xy = torch.nan_to_num(mmult_xy)
            
            # Compute magnitude for z direction
            mag_z = torch.sqrt(
                torch.cat([d, torch.zeros((size(d, 1), size(d, 2), 1)).to(self.device)], dim=2) ** 2
            )
            
            # Apply spatial weighting for z direction
            weighted_tau_z = tau_z * weight_d
            magtmag_z = self._soft_threshold_scalar(mag_z[:, :, :-1], weighted_tau_z)
            
            # Pad back to original size
            magtmag_z_full = torch.cat([magtmag_z, torch.zeros((size(d, 1), size(d, 2), 1)).to(self.device)], dim=2)
            mmult_z = magtmag_z_full / (mag_z + torch.finfo(torch.float32).eps)
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
        """Compute spatially-weighted penalty."""
        Lvk1, Lvk2, Lvk3 = Lvk_outputs
        
        # Get weight masks
        weight_mask = self._create_weight_mask((Lvk1.shape[0]+1, Lvk1.shape[1], Lvk1.shape[2]))
        weight_v = weight_mask[:-1, :, :]
        weight_h = weight_mask[:, :-1, :]
        weight_d = weight_mask[:, :, :-1]
        
        return (
            torch.sum(weight_v * torch.abs(Lvk1)) * self.tau +
            torch.sum(weight_h * torch.abs(Lvk2)) * self.tau +
            torch.sum(weight_d * torch.abs(Lvk3)) * self.tau_z
        ) 