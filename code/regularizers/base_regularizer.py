from abc import ABC, abstractmethod
import torch
import torch.fft as fft
import numpy as np

class BaseRegularizer(ABC):
    """
    Abstract base class for regularizers used in ADMM optimization.
    
    This class defines the interface that all regularizers must implement
    to work with the ADMM solver. Each regularizer handles the specific
    operations needed for its type of regularization (TV, L1, etc.).
    """
    
    def __init__(self, device, **kwargs):
        """
        Initialize the regularizer.
        
        Args:
            device: torch device (cpu/cuda)
            **kwargs: regularizer-specific parameters
        """
        self.device = device
        self.params = kwargs
    
    @abstractmethod
    def setup_variables(self, vk):
        """
        Setup regularizer-specific variables before ADMM iterations.
        
        Args:
            vk: Initial variable tensor with shape matching the problem
            
        Returns:
            dict: Dictionary containing all regularizer variables needed for iterations
        """
        pass
    
    @abstractmethod
    def compute_PsiTPsi(self, vk):
        """
        Compute the regularizer's operator transpose times operator (Ψᵀ Ψ).
        This is used in the v-update denominator.
        
        Args:
            vk: Variable tensor
            
        Returns:
            torch.Tensor: PsiTPsi operator in Fourier domain
        """
        pass
    
    @abstractmethod
    def apply_Psi(self, x):
        """
        Apply the regularizer's sparsifying operator Ψ.
        
        Args:
            x: Input tensor
            
        Returns:
            tuple: Outputs of the sparsifying operator
        """
        pass
    
    @abstractmethod
    def apply_PsiT(self, *args):
        """
        Apply the transpose of the regularizer's sparsifying operator Ψᵀ.
        
        Args:
            *args: Inputs to the transpose operator
            
        Returns:
            torch.Tensor: Output of the transpose operator
        """
        pass
    
    @abstractmethod
    def soft_threshold(self, *args, tau):
        """
        Apply soft thresholding for the regularizer.
        
        Args:
            *args: Gradients or other variables to threshold
            tau: Thresholding parameter
            
        Returns:
            tuple: Thresholded variables
        """
        pass
    
    @abstractmethod
    def update_dual_variables(self, reg_vars, Lvk_outputs, uk_outputs, mu2):
        """
        Update dual variables associated with the regularizer.
        
        Args:
            reg_vars: Dictionary of regularizer variables
            Lvk_outputs: Current Psi outputs
            uk_outputs: Thresholded outputs
            mu2: ADMM parameter
            
        Returns:
            dict: Updated regularizer variables
        """
        pass
    
    @abstractmethod
    def compute_residuals(self, Lvk_old, Lvk_new, uk_outputs, mu2):
        """
        Compute primal and dual residuals for the regularizer.
        
        Args:
            Lvk_old: Previous Psi outputs
            Lvk_new: Current Psi outputs  
            uk_outputs: Thresholded outputs
            mu2: ADMM parameter
            
        Returns:
            tuple: (dual_residual, primal_residual)
        """
        pass
    
    @abstractmethod
    def compute_penalty(self, Lvk_outputs):
        """
        Compute the regularization penalty term.
        
        Args:
            Lvk_outputs: Current Psi outputs
            
        Returns:
            torch.Tensor: Regularization penalty value
        """
        pass
    
    def l2norm(self, X):
        """Utility function for L2 norm computation."""
        return torch.linalg.norm(X.ravel(), ord=2) 