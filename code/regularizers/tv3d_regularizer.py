import torch
import torch.fft as fft
import copy
from .base_regularizer import BaseRegularizer

class TV3DRegularizer(BaseRegularizer):
    """
    3D Total Variation Regularizer with separate weighting for z-direction.
    
    This regularizer implements anisotropic 3D TV with different weights
    for lateral (xy) and axial (z) directions, commonly used in 3D imaging
    applications like lensless reconstruction.
    """
    
    def __init__(self, device, tau=6e-4, tau_z=6e-10, **kwargs):
        """
        Initialize 3D TV regularizer.
        
        Args:
            device: torch device
            tau: Regularization weight for lateral (xy) directions
            tau_z: Regularization weight for axial (z) direction
        """
        super().__init__(device, tau=tau, tau_z=tau_z, **kwargs)
        self.tau = tau
        self.tau_z = tau_z
    
    def setup_variables(self, vk):
        """Setup 3D TV regularizer variables."""
        # Initialize dual variables for the three gradient directions
        eta_1 = vk[:-1, :, :].clone().detach()  # x-direction dual
        eta_2 = vk[:, :-1, :].clone().detach()  # y-direction dual
        eta_3 = vk[:, :, :-1].clone().detach()  # z-direction dual
        
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
        """Compute the 3D Laplacian operator in Fourier domain."""
        laplacian = vk.clone().detach()
        laplacian.zero_()
        laplacian[0, 0, 0] = 6
        laplacian[0, 1, 0] = -1
        laplacian[1, 0, 0] = -1
        laplacian[0, 0, 1] = -1
        laplacian[0, -1, 0] = -1
        laplacian[-1, 0, 0] = -1
        laplacian[0, 0, -1] = -1
        return torch.abs(fft.fftn(laplacian))
    
    def apply_Psi(self, x):
        """Apply 3D gradient operator (negative differences)."""
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
        """
        Perform anisotropic 3D TV soft thresholding with separate z-weighting.
        
        Args:
            v, h, d: Gradient components in x, y, z directions
            tau: Tuple (tau_xy, tau_z) or single value
            
        Returns:
            tuple: Thresholded gradient components
        """
        if isinstance(tau, (tuple, list)):
            tau_xy, tau_z = tau
        else:
            tau_xy = tau
            tau_z = self.tau_z / self.tau * tau if hasattr(self, 'tau') else tau
        
        def size(x, i):
            return x.shape[i - 1]
        
        vararg_out = []
        
        if size(v, 1) != 0:
            # Compute magnitude for xy directions
            mag_xy = torch.sqrt(
                torch.cat([v, torch.zeros(1, size(v, 2), size(v, 3)).to(self.device)], dim=0) ** 2 +
                torch.cat([h, torch.zeros(size(h, 1), 1, size(h, 3)).to(self.device)], dim=1) ** 2
            )
            
            # Soft threshold xy magnitude
            magtmag_xy = self._soft_threshold_scalar(mag_xy, tau_xy)
            mmult_xy = magtmag_xy / (mag_xy + torch.finfo(torch.float32).eps)
            mmult_xy = mmult_xy * (mag_xy > 0)
            mmult_xy = torch.nan_to_num(mmult_xy)
            
            # Compute magnitude for z direction
            mag_z = torch.sqrt(
                torch.cat([d, torch.zeros((size(d, 1), size(d, 2), 1)).to(self.device)], dim=2) ** 2
            )
            
            # Soft threshold z magnitude
            magtmag_z = self._soft_threshold_scalar(mag_z, tau_z)
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
        """Update dual variables for 3D TV."""
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
        """Compute 3D TV penalty."""
        Lvk1, Lvk2, Lvk3 = Lvk_outputs
        return self.tau * (
            torch.sum(torch.abs(Lvk1)) +
            torch.sum(torch.abs(Lvk2)) +
            torch.sum(torch.abs(Lvk3))
        ) 