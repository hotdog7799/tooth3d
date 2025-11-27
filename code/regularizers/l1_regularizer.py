import torch
import torch.fft as fft
from .base_regularizer import BaseRegularizer

class L1Regularizer(BaseRegularizer):
    """
    L1 Sparsity Regularizer for promoting sparse reconstructions.
    
    This regularizer promotes sparsity in the reconstruction, which is
    beneficial for tooth structures with clear boundaries and sparse features.
    """
    
    def __init__(self, device, tau=1e-3, **kwargs):
        """
        Initialize L1 regularizer.
        
        Args:
            device: torch device
            tau: Regularization weight for L1 sparsity
        """
        super().__init__(device, tau=tau, **kwargs)
        self.tau = tau
    
    def setup_variables(self, vk):
        """Setup L1 regularizer variables."""
        # For L1, we work directly with the image
        eta = vk.clone().detach()
        Lvk = vk.clone().detach()
        
        return {
            'eta': eta,
            'Lvk': Lvk
        }
    
    def compute_PsiTPsi(self, vk):
        """For L1, Psi is identity, so PsiTPsi is 1."""
        return torch.ones_like(vk)
    
    def apply_Psi(self, x):
        """Apply identity operator (Psi = I for L1)."""
        return (x,)
    
    def apply_PsiT(self, x):
        """Apply transpose of identity operator."""
        return x
    
    def soft_threshold(self, x, tau):
        """Apply L1 soft thresholding."""
        if isinstance(tau, (tuple, list)):
            tau = tau[0]  # Use first element if tuple
        
        return [torch.sign(x) * torch.maximum(torch.zeros_like(x), torch.abs(x) - tau)]
    
    def update_dual_variables(self, reg_vars, Lvk_outputs, uk_outputs, mu2):
        """Update dual variables for L1."""
        Lvk = Lvk_outputs[0]
        uk = uk_outputs[0]
        
        # Compute residual
        r_su = Lvk - uk
        
        # Update dual variable
        reg_vars['eta'] = reg_vars['eta'] + mu2 * r_su
        
        return reg_vars
    
    def compute_residuals(self, Lvk_old, Lvk_new, uk_outputs, mu2):
        """Compute primal and dual residuals."""
        Lvk_old_val = Lvk_old[0]
        Lvk_new_val = Lvk_new[0]
        uk = uk_outputs[0]
        
        # Dual residual
        dual_resid = mu2 * self.l2norm(Lvk_old_val - Lvk_new_val)
        
        # Primal residual
        primal_resid = self.l2norm(Lvk_new_val - uk)
        
        return dual_resid, primal_resid
    
    def compute_penalty(self, Lvk_outputs):
        """Compute L1 penalty."""
        Lvk = Lvk_outputs[0]
        return self.tau * torch.sum(torch.abs(Lvk)) 