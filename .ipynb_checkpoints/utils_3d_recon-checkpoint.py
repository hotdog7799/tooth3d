import numpy as np
import matplotlib.pyplot as plt
import torch.fft as fft
import torch
import cv2
import glob
import time
import os
import PIL
from PIL import Image
from torch.utils.data import Dataset
# from torchvision import transforms
#
from datetime import datetime
from IPython.display import clear_output
from mpl_toolkits.axes_grid1 import make_axes_locatable

from matplotlib.colors import LinearSegmentedColormap

class ImageDataset(Dataset):
    """Custom image file dataset loader."""
    
    def __init__(self, data_path, label_path, transforms_image, transforms_label):
        # Initialize dataset paths and length
        self.img_paths = []
        self.label_paths = []
        self.transforms_image = transforms_image
        self.transforms_label = transforms_label
        for (img_path, label_path) in zip(sorted(glob.glob(data_path + '/*.tiff')), sorted(glob.glob(label_path + '/*'))):
            self.img_paths.append(img_path)
            self.label_paths.append(label_path)
        self.len = len(self.img_paths)

    def __getitem__(self, index):
        # Return image and label for a given index
        img = Image.open(self.img_paths[index])
        label = Image.open(self.label_paths[index])
        if img.format == 'TIFF':
            img = np.array(img)/255
        
        if self.transforms_image is not None:
            img = self.transforms_image(img)
        if self.transforms_label is not None:
            label = self.transforms_label(label)
                            
        return (img, label)

    def __len__(self):
        # Return dataset length
        return self.len

class CustomCenterCrop(object):
    def __init__(self, crop_size, center):
        self.crop_size = crop_size
        self.center = center

    def __call__(self, img):
        h, w = img.shape[-2:]  # Use img.shape for a PyTorch tensor
        new_h, new_w = self.crop_size
        center_x, center_y = self.center

        # Calculate the top-left corner of the crop
        top = int(center_x - new_w / 2)
        left = int(center_y - new_h / 2)

        # Perform the center crop
        img = transforms.functional.crop(img, top, left, new_h, new_w)

        return img

def hyperparameter_table(camera):
    """
    Define hyperparameters for a given camera.
    
    Parameters:
    - camera: Camera information
    
    Returns:
    - table: Dictionary of hyperparameters
    """
    if camera == 'v':
        table = {'mu1':[3.8009656577742135e-07],
                 'mu2':[7.55148676034878e-07],
                 'mu3':[6.681235049654788e-07],
                 'tau':[5.023222797717608e-07]}
        
    elif camera =='t':
        table = {'mu1':[1.6140469938363822e-07],
                 'mu2':[2.7243615363659046e-07],
                 'mu3':[1.8733291540229402e-07],
                 'tau':[5.492345280799782e-07]}
        
    else:
        table = {'mu1': [4.925799856891899e-08],
                 'mu2': [1.1417107259603654e-07],
                 'mu3': [7.892058562219972e-08],
                 'tau': [5.542606800190697e-07]}
    return table

def to_tensor_or_numpy(data):
    """
    Convert input data to either a PyTorch Tensor or a Numpy array.

    Parameters:
    - data: Input data, either a PyTorch Tensor or a Numpy array

    Returns:
    - Converted data as a PyTorch Tensor or a Numpy array
    """
    # Check if input is a Tensor
    if torch.is_tensor(data):
        if data.is_cuda:
            data = data.cpu().detach()

        # Permute color channels if needed
        if len(data.shape) == 3:
            data = data.permute(1, 2, 0)  # Change color channel order (C, H, W) -> (H, W, C)
        elif len(data.shape) == 4:
            data = data.permute(0, 2, 3, 1)  # Change color channel order (B, C, H, W) -> (B, H, W, C)

        return data.numpy()

    # Check if input is a Numpy array
    elif isinstance(data, np.ndarray):
        # Permute color channels if needed
        data = torch.from_numpy(data)
        if len(data.shape) == 3:
            data = data.permute(2, 0, 1)  # Change color channel order (H, W, C) -> (C, H, W)
        elif len(data.shape) == 4:
            data = data.permute(0, 3, 1, 2)  # Change color channel order (B, H, W, C) -> (B, C, H, W)
        return data

    else:
        raise ValueError("Input must be a PyTorch Tensor or a Numpy array")


def psnr(img1, img2):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) between two images.
    
    Parameters:
    - img1: First input image
    - img2: Second input image
    
    Returns:
    - PSNR value
    """
    if torch.is_tensor(img1):
        mse = torch.mean((img1 - img2) ** 2).detach().cpu().numpy()
    else:
        mse = np.mean((img1 - img2) ** 2)
    if mse < 1e-9:
        return "Same Image"
    return 10 * np.log10(1. / mse)

def center_crop(img, center=None, size=(0, 0), mode="crop", **kwargs):
    """
    Crop the input image based on the center and size.
    
    Parameters:
    - img: Input image (numpy array or torch tensor)
    - center: Center coordinates for cropping
    - size: Size of the cropped region
    - mode: Cropping mode ("crop", "same", "center")
    
    Returns:
    - output: Cropped image
    """
    if torch.is_tensor(img):
        return center_crop_t(img, center=center, size=size, mode=mode, **kwargs)
    else:
        return center_crop_n(img, center=center, size=size, mode=mode, **kwargs)

def center_crop_n(img, center=None, size=(0, 0), mode="crop"):
    """
    Crop numpy array based on the center and size.
    
    Parameters:
    - img: Input numpy array
    - center: Center coordinates for cropping
    - size: Size of the cropped region
    - mode: Cropping mode ("crop", "same", "center")
    
    Returns:
    - output: Cropped numpy array
    """
    img_h, img_w = np.shape(img)[:2]
    crop_h, crop_w = size
    crop_h_half, crop_h_mod = divmod(crop_h, 2)
    crop_w_half, crop_w_mod = divmod(crop_w, 2)

    if center is None:
        crop_center_h = img_h // 2
        crop_center_w = img_w // 2
    else:
        crop_center_h, crop_center_w = center

    if mode == "crop":
        output = img[crop_center_h - crop_h_half: crop_center_h + crop_h_half + crop_h_mod,
                     crop_center_w - crop_w_half: crop_center_w + crop_w_half + crop_w_mod, ...]
    elif mode == "same":
        output = np.zeros_like(img)
        output[crop_center_h - crop_h_half: crop_center_h + crop_h_half + crop_h_mod,
               crop_center_w - crop_w_half: crop_center_w + crop_w_half + crop_w_mod, ...] = img[
            crop_center_h - crop_h_half: crop_center_h + crop_h_half + crop_h_mod,
            crop_center_w - crop_w_half: crop_center_w + crop_w_half + crop_w_mod, ...
        ]
    elif mode == "center":
        output = np.zeros_like(img)
        output[img_h - crop_h_half: img_h + crop_h_half + crop_h_mod,
               img_w - crop_w_half: img_w + crop_w_half + crop_w_mod, ...] = img[
            crop_center_h - crop_h_half: crop_center_h + crop_h_half + crop_h_mod,
            crop_center_w - crop_w_half: crop_center_w + crop_w_half + crop_w_mod, ...
        ]
    return output

def center_crop_t(img, center=None, size=(0, 0), mode="crop"):
    """
    Crop torch tensor based on the center and size.
    
    Parameters:
    - img: Input torch tensor
    - center: Center coordinates for cropping
    - size: Size of the cropped region
    - mode: Cropping mode ("crop", "same", "center")
    
    Returns:
    - output: Cropped torch tensor
    """
    img_h, img_w = img.size()[-2:]
    crop_h, crop_w = size
    crop_h_half, crop_h_mod = divmod(crop_h, 2)
    crop_w_half, crop_w_mod = divmod(crop_w, 2)

    if center is None:
        crop_center_h = img_h // 2
        crop_center_w = img_w // 2
    else:
        crop_center_h, crop_center_w = center

    if mode == "crop":
        output = img[..., crop_center_h - crop_h_half: crop_center_h + crop_h_half + crop_h_mod,
                    crop_center_w - crop_w_half: crop_center_w + crop_w_half + crop_w_mod]
    elif mode == "same":
        output = torch.zeros_like(img)
        output[..., crop_center_h - crop_h_half: crop_center_h + crop_h_half + crop_h_mod,
               crop_center_w - crop_w_half: crop_center_w + crop_w_half + crop_w_mod] = img[
            ..., crop_center_h - crop_h_half: crop_center_h + crop_h_half + crop_h_mod,
            crop_center_w - crop_w_half: crop_center_w + crop_w_half + crop_w_mod
        ]
    elif mode == "center":
        output = torch.zeros_like(img)
        output[..., img_h - crop_h_half: img_h + crop_h_half + crop_h_mod,
               img_w - crop_w_half: img_w + crop_w_half + crop_w_mod] = img[
            ..., crop_center_h - crop_h_half: crop_center_h + crop_h_half + crop_h_mod,
            crop_center_w - crop_w_half: crop_center_w + crop_w_half + crop_w_mod
        ]
    return output

def clamp(img, percentile_lower=0, percentile_upper=99.9):
    """
    Clamp image values within the specified percentile range.
    
    Parameters:
    - img: Input torch tensor
    - percentile_lower: Lower percentile for clamping
    - percentile_upper: Upper percentile for clamping
    
    Returns:
    - clipped_img: Clamped torch tensor
    """
    lower_percentile = percentile_lower / 100.0
    upper_percentile = percentile_upper / 100.0

    img_shape = img.size()
    if len(img_shape)==2:
        channel = 1
        height = img_shape[0]
        width = img_shape[1]
    else:
        channel = img_shape[-3]
        height = img_shape[-2]
        width = img_shape[-1]
    
    reshaped_img = img.reshape(channel * height * width)
    sorted_values, _ = torch.sort(reshaped_img, dim=0)

    lower_index = int(lower_percentile * (channel * height * width))
    upper_index = int(upper_percentile * (channel * height * width))

    lower_values = sorted_values[lower_index]
    upper_values = sorted_values[upper_index-1]

    lower_values = lower_values.reshape(1, 1, 1)
    upper_values = upper_values.reshape(1, 1, 1)

    clipped_img = torch.clamp(img, lower_values, upper_values)

    return clipped_img

def call(img_path, crop=False, center=(1824-120, 2736-140), size=(2600, 2600), figure=False, is_color=False):
    """
    Read image, optionally crop, and return the image.
    
    Parameters:
    - img_path: Path to the input image
    - crop: Whether to perform cropping
    - crop_center: Center coordinates for cropping
    - crop_size: Size of the cropped region
    - figure: Whether to display the image
    - is_color: Whether to use as 3 channel color image
    
    Returns:
    - img: Processed image
    """
    if is_color:
        img = cv2.imread(img_path, -1)
    else:
        img = cv2.imread(img_path, 0)
        img = np.expand_dims(img, axis=-1)
    img = np.array(img, dtype="float32")
    img_size = np.shape(img)
    if figure:
        plt.imshow(img, cmap='gray')
        plt.colorbar()
        plt.axis('off')
        plt.show()
    if crop:
        img = center_crop(img, center=(center[0], center[1]), size=(size[0], size[1]), mode='crop')
    return img

def get_dict(img_path, camera=None):
    """
    Create a dictionary mapping image names to file paths.
    
    Parameters:
    - camera: Camera information
    - img_path: Path to the image directory
    
    Returns:
    - img_dict: Dictionary mapping image names to file paths
    """
    img_dict = {}
    desired_extensions = ['tiff', 'png', 'jpg']
    img_path = os.path.join(img_path, '')

    for ext in desired_extensions:
        for path in glob.glob(os.path.join(img_path, f'**/*.{ext}'), recursive=True):
            relative_path = os.path.relpath(path, img_path)
            _, img_name = os.path.split(relative_path)
            key = os.path.splitext(img_name)[0]
            if path.find('archive') != -1 or path.find('result') != -1:
                continue
            if key.find('psf') != -1:
                img_dict['psf'] = path
            else:
                img_dict[key] = path

    return img_dict

def linalg_norm(img):
    if torch.is_tensor(img):
        return minmax_norm_t(img)
    else:
        return minmax_norm_n(img)
    
def minmax_norm(img):
    if torch.is_tensor(img):
        return minmax_norm_t(img)
    else:
        return minmax_norm_n(img)
    
def max_norm(img):
    if torch.is_tensor(img):
        return max_norm_t(img)
    else:
        return max_norm_n(img)

def linalg_norm_t(img):
    """
    Normalize a PyTorch tensor using the L2 norm.

    Parameters:
    - img: Input PyTorch tensor

    Returns:
    - Normalized PyTorch tensor
    """
    return img / torch.linalg.norm(img.contiguous().view(-1))

def minmax_norm_t(img):
    """
    Min-max normalize the tensor to the float range.
    
    Parameters:
    - img: Input image
    
    Returns:
    - Normalized image
    """
    return (img - img.min()) / (img.max() - img.min())

def max_norm_t(img):
    """
    Normalize the tensor to the float range based on the maximum value.
    
    Parameters:
    - img: Input image
    
    Returns:
    - Normalized image
    """
    return img / img.max()

def linalg_norm_n(img):
    """
    Normalize a Numpy array using the L2 norm.

    Parameters:
    - img: Input Numpy array

    Returns:
    - Normalized Numpy array
    """
    return (img / np.linalg.norm(img.reshape(-1)) * 255).astype(np.uint8)

def minmax_norm_n(img):
    """
    Min-max normalize the image to the 8-bit range.
    
    Parameters:
    - img: Input image
    
    Returns:
    - Normalized image
    """
    return ((img - np.min(img)) / (np.max(img) - np.min(img)) * 255).astype(np.uint8)

def max_norm_n(img):
    """
    Normalize the image to the 8-bit range based on the maximum value.
    
    Parameters:
    - img: Input image
    
    Returns:
    - Normalized image
    """
    return ((img / np.max(img)) * 255).astype(np.uint8)

def plot_ADMM(psf, raw, result, times, hyperparameters, clamp_=0.1, total_time=0, size=1200, crop=None, norm=None, iteration=None, is_color=False):
    """
    Display PSF, raw image, and result side by side with relevant information.
    
    Parameters:
    - psf: Point Spread Function
    - raw: Raw input image
    - result: Output result image
    - times: Time taken for the operation
    - hyperparameters: Dictionary of hyperparameters
    - clamp_: Clamping parameter
    - total_time: Total time taken
    - size: Crop size
    - crop: Crop factor
    - norm: Normalization method
    - iteration: Iteration number (optional)
    - is_color: Wheter image is RGB
    """
    display_time = time.time()
    
    # Convert torch tensors to NumPy arrays for plotting
    if torch.is_tensor(psf):
        psf = to_tensor_or_numpy(psf)
    if torch.is_tensor(raw):
        raw = to_tensor_or_numpy(raw)
    if torch.is_tensor(result):
        # Clamp result tensor within the specified percentile range
        if clamp_:
            result = clamp(result, percentile_upper=99. + (1. - clamp_))
        result = to_tensor_or_numpy(result)
    
    # Apply cropping if specified
    if crop:
        result = center_crop(result, center=None, size=(size, size), mode='crop')

    # Normalize images based on the specified normalization method
    if norm == 'max':
        psf = max_norm(psf)
        raw = max_norm(raw)
        result = max_norm(result)
        psf[psf<0] = 0
        raw[raw<0] = 0
        result[result<0] = 0
    elif norm == 'linalg':
        psf = linalg_norm(psf)
        raw = linalg_norm(raw)
        result = linalg_norm(result)        
    else:
        psf = minmax_norm(psf)
        raw = minmax_norm(raw)
        result = minmax_norm(result)
    
    # Plot PSF, raw, and result images side by side
    fig, ax = plt.subplots(1, 3)
    fig.set_figheight(8)
    fig.set_figwidth(30)
    
    if is_color:
        ax[0].imshow(cv2.cvtColor(psf, cv2.COLOR_BGR2RGB))
        ax[1].imshow(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB))
        ax[2].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    else:
        ax[0].imshow(psf, cmap='gray')
        ax[1].imshow(raw, cmap='gray')
        ax[2].imshow(result, cmap='gray')
        

    ax[0].set_title("PSF", fontsize=20)
    ax[1].set_title("RAW", fontsize=20)
    ax[2].set_title("OUTPUT", fontsize=20)

    ax[0].axis('off')
    ax[1].axis('off')
    ax[2].axis('off')
    
    # Display relevant information as title
    if total_time and iteration is not None:
        fig.suptitle('{} Iteration Elapsed: {:.2} ms (total {:3.3} ms), \n $\mu_1$:{:.2e}, $\mu_2$:{:.2e}, $\mu_3$:{:.2e}, $\ttau$:{:.2e}'.format(
            iteration,
            times * 1e3,
            total_time * 1e3,
            hyperparameters['mu1'][0],
            hyperparameters['mu2'][0],
            hyperparameters['mu3'][0],
            hyperparameters['tau'][0]), fontsize=25)
    else:
        fig.suptitle(
            'Total Elapsed (+ displaying): {:3.3} s, \n $\mu_1$:{:.2e}, $\mu_2$:{:.2e}, $\mu_3$:{:.2e}, $\ttau$:{:.2e}'.format(
                times + time.time() - display_time,
                hyperparameters['mu1'][0],
                hyperparameters['mu2'][0],
                hyperparameters['mu3'][0],
                hyperparameters['tau'][0]), fontsize=25)
    
    fig.tight_layout()

    plt.show()

    
def plot_ADMM_ch(mu1,mu2,mu3,tau,ch1,ch2,ch3, path1 = None, clamp_=0.1, size=1200, crop=None, norm=None, iteration=None, is_color=False,filename = 'RGB_channels'):
    """
    Display color channel result side by side with relevant information.
    
    Parameters:
    - result: Output result image
    - clamp_: Clamping parameter
    - size: Crop size
    - crop: Crop factor
    - norm: Normalization method
    - iteration: Iteration number (optional)
    - is_color: Wheter image is RGB
    - filename: Base filename to save the figure (optional)
    """
    # Generate current timestamp
    current_time = datetime.now().strftime("%m%d_%H%M%S")  # MonthDay_HourMinute format

    # Append the timestamp to the filename
    filename_with_time = os.path.join(path1, f"{filename}_([{mu1}][{mu2}][{mu3}][{tau}])_{current_time}.png")  # Complete filename with timestamp
    
    # Convert torch tensors to NumPy arrays for plotting
    if torch.is_tensor(ch1):
        # Clamp result tensor within the specified percentile range
        if clamp_:
            ch1 = clamp(ch1, percentile_upper=99. + (1. - clamp_))
        ch1 = to_tensor_or_numpy(ch1)
    if torch.is_tensor(ch2):
        # Clamp result tensor within the specified percentile range
        if clamp_:
            ch2 = clamp(ch2, percentile_upper=99. + (1. - clamp_))
        ch2 = to_tensor_or_numpy(ch2)
    if torch.is_tensor(ch3):
        # Clamp result tensor within the specified percentile range
        if clamp_:
            ch3 = clamp(ch3, percentile_upper=99. + (1. - clamp_))
        ch3 = to_tensor_or_numpy(ch3)
    
    # Apply cropping if specified
    if crop:
        ch1 = center_crop(ch1, center=None, size=(size, size), mode='crop')
        ch2 = center_crop(ch2, center=None, size=(size, size), mode='crop')
        ch3 = center_crop(ch3, center=None, size=(size, size), mode='crop')

    # Plot PSF, raw, and result images side by side
    fig, ax = plt.subplots(1, 3)
    fig.set_figheight(8)
    fig.set_figwidth(30)
    
    #ch1= b, ch3=r
    #  BGR -> RGB 
    ch1 = ch1[..., [2, 1, 0]]  #  r
    ch3 = ch3[..., [2, 1, 0]]  #  b
    
    
    if is_color:
        # im1 = ax[0].imshow(cv2.cvtColor(ch1, cv2.COLOR_BGR2RGB))
        # im2 = ax[1].imshow(cv2.cvtColor(ch2, cv2.COLOR_BGR2RGB))
        # im3 = ax[2].imshow(cv2.cvtColor(ch3, cv2.COLOR_BGR2RGB))
        # im1 = ax[0].imshow(ch1)
        im1 = ax[0].imshow(ch1, cmap='Blues_r',vmin=np.min(ch1), vmax=np.max(ch1))
        im2 = ax[1].imshow(ch2, cmap='Greens_r',vmin=np.min(ch2), vmax=np.max(ch2))
        im3 = ax[2].imshow(ch3, cmap='Reds_r',vmin=np.min(ch3), vmax=np.max(ch3)) # _r -> 컬러바 리버스
        # im1 = ax[0].imshow(ch1, cmap='Blues_r',vmin=0, vmax=1)
        # im2 = ax[1].imshow(ch2, cmap='Greens_r',vmin=0, vmax=1)
        # im3 = ax[2].imshow(ch3, cmap='Reds_r',vmin=0, vmax=1) # _r -> 컬러바 리버스
        
    ax[0].set_title("Blue channel",fontsize=30)
    ax[1].set_title("Green channel", fontsize=30)
    ax[2].set_title("Red channel", fontsize=30)
    
    cbar1 = plt.colorbar(im1, ax = ax[0])
    cbar2 = plt.colorbar(im2, ax = ax[1])
    cbar3 = plt.colorbar(im3, ax = ax[2])
    
    # Adjust colorbar font size
    # Increase font size for colorbar ticks
    cbar1.ax.tick_params(labelsize=30)  
    cbar2.ax.tick_params(labelsize=30)
    cbar3.ax.tick_params(labelsize=30)
    
    for a in ax:
        a.axis('off')  # Turn off axis for all
    plt.savefig(filename_with_time, bbox_inches='tight')
    
    fig.tight_layout()

    plt.close(fig)
    clear_output(wait=True)


def plot_training_results(outputs_l, labels_l, mu1s, mu2s, mu3s, taus):
    """
    Plot training results including loss, PSNR, and hyperparameters.

    Parameters:
    - outputs_l: Model outputs after cropping.
    - labels_l: Ground truth labels after cropping.
    - mu1s, mu2s, mu3s, taus: Lists containing hyperparameter values over epochs.
    """
    
    # If the shape of the outputs is 4D, squeeze it to 2D
    if len(outputs_l.shape) == 4:
        outputs_l = outputs_l[0, 0, :, :]
        labels_l = labels_l[0, 0, :, :]
        
    # Plot ground truth and model outputs side by side
    plt.figure(figsize=(20, 20))
    plt.subplot(2, 2, 1)
    plt.imshow(labels_l.cpu().numpy(), cmap='gray')
    plt.subplot(2, 2, 2)
    plt.imshow(outputs_l.to('cpu').detach().numpy(), cmap='gray')
    plt.show()

    # Plot hyperparameter values over epochs
    plt.figure(figsize=(20, 20))
    plt.subplot(4, 4, 1)
    plt.plot(mu1s, color='red', label='mu1')
    plt.legend()
    plt.xlabel("Epochs", fontsize=15)
    plt.ylabel("Parameters", fontsize=15)
    plt.subplot(4, 4, 2)
    plt.plot(mu2s, color='blue', label='mu2')
    plt.legend()
    plt.xlabel("Epochs", fontsize=15)
    plt.ylabel("Parameters", fontsize=15)
    plt.subplot(4, 4, 3)
    plt.plot(mu3s, color='green', label='mu3')
    plt.legend()
    plt.xlabel("Epochs", fontsize=15)
    plt.ylabel("Parameters", fontsize=15)
    plt.subplot(4, 4, 4)
    plt.plot(taus, color='black', label='tau')
    plt.legend()
    plt.xlabel("Epochs", fontsize=15)
    plt.ylabel("Parameters", fontsize=15)
    plt.show()

def red_div_green(ch):
    green_channel = ch[1, :, :]
    red_channel = ch[2, :, :]
    
    epsilon = 1e-10
    green_channel_safe = torch.where(green_channel == 0, epsilon, green_channel)
    
    red_over_green = red_channel / green_channel_safe
    red_over_green = minmax_norm(red_over_green)
    print(red_over_green)
    
    plt.imshow(red_over_green.cpu().numpy(), cmap='hot')
    plt.colorbar()
    plt.title('Red channel / Green channel')
    plt.show
    
def save_rgb_channels(ch1, ch2, ch3, p1, p2, p3, p4, save_path):
    from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
    # ch1 = red
    # ch2 = green
    # ch3 = blue    
    
    plt.figure(figsize=(18,8))
    
    plt.subplot(1,3,1)
    ax = plt.gca()
    im = plt.imshow(ch1, cmap='gray')
    plt.title('Red Channel')
    plt.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    for t in cbar.ax.get_yticklabels():
        t.set_fontsize(16)

    plt.subplot(1,3,2)
    ax = plt.gca()
    im = plt.imshow(ch2, cmap='gray')
    plt.title('Green Channel')
    plt.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    for t in cbar.ax.get_yticklabels():
        t.set_fontsize(16)
        
    plt.subplot(1,3,3)
    ax = plt.gca()
    im = plt.imshow(ch3, cmap='gray')
    plt.title('Blue Channel')
    plt.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    for t in cbar.ax.get_yticklabels():
        t.set_fontsize(16)

    plt.savefig(os.path.join(save_path, f'rgb_plots_{np.log10(1/p1):0.0f}_{np.log10(1/p2):0.0f}_{np.log10(1/p3):0.0f}_{np.log10(1/p4):0.0f}.png'), bbox_inches='tight')
    plt.close()

def save_rgb_channels2(img, p1, p2, p3, p4, save_path):
    from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
    ch1 = img[...,2]
    ch2 = img[...,1]
    ch3 = img[...,0]
    
    plt.figure(figsize=(18,8))
    
    plt.subplot(1,3,1)
    ax = plt.gca()
    im = plt.imshow(ch1, cmap = wave_to_cm(650))
    plt.title('Red Channel')
    plt.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    for t in cbar.ax.get_yticklabels():
        t.set_fontsize(16)

    plt.subplot(1,3,2)
    ax = plt.gca()
    im = plt.imshow(ch2, cmap = wave_to_cm(550))
    plt.title('Green Channel')
    plt.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    for t in cbar.ax.get_yticklabels():
        t.set_fontsize(16)
        
    plt.subplot(1,3,3)
    ax = plt.gca()
    im = plt.imshow(ch3, cmap = wave_to_cm(450))
    plt.title('Blue Channel')
    plt.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    for t in cbar.ax.get_yticklabels():
        t.set_fontsize(16)

    plt.savefig(os.path.join(save_path, f'rgb_plots_{np.log10(1/p1):0.0f}_{np.log10(1/p2):0.0f}_{np.log10(1/p3):0.0f}_{np.log10(1/p4):0.0f}.png'), bbox_inches='tight')
    plt.close()
    
    
def save_rgb_channels_fixed(ch1, ch2, ch3, p1, p2, p3, p4, save_path):
    from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
    
    plt.figure(figsize=(18,8))
    
    plt.subplot(1,3,1)
    ax = plt.gca()
    im = plt.imshow(ch1, cmap='gray',vmax=1,vmin=0)
    plt.title('Red Channel')
    plt.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    for t in cbar.ax.get_yticklabels():
        t.set_fontsize(16)

    plt.subplot(1,3,2)
    ax = plt.gca()
    im = plt.imshow(ch2, cmap='gray',vmax=1,vmin=0)
    plt.title('Green Channel')
    plt.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    for t in cbar.ax.get_yticklabels():
        t.set_fontsize(16)
        
    plt.subplot(1,3,3)
    ax = plt.gca()
    im = plt.imshow(ch3, cmap='gray',vmax=1,vmin=0)
    plt.title('Blue Channel')
    plt.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    for t in cbar.ax.get_yticklabels():
        t.set_fontsize(16)
        
def save_averaged_rgb(ch1,ch2,ch3,save_path):
    from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
    
    # Generate current timestamp
    current_time = datetime.now().strftime("%m%d_%H%M%S")  #
    
    plt.figure(figsize=(18,8))
    
    plt.subplot(1,3,1)
    ax = plt.gca()
    im = plt.imshow(ch1, cmap='gray')
    plt.title('Red Channel')
    plt.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    for t in cbar.ax.get_yticklabels():
        t.set_fontsize(16)

    plt.subplot(1,3,2)
    ax = plt.gca()
    im = plt.imshow(ch2, cmap='gray')
    plt.title('Green Channel')
    plt.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    for t in cbar.ax.get_yticklabels():
        t.set_fontsize(16)
        
    plt.subplot(1,3,3)
    ax = plt.gca()
    im = plt.imshow(ch3, cmap='gray')
    plt.title('Blue Channel')
    plt.axis('off')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    for t in cbar.ax.get_yticklabels():
        t.set_fontsize(16)    

    plt.savefig(os.path.join(save_path, f'averaged_rgb_plot_{current_time}.png'), bbox_inches='tight')
    plt.close() 
    

def load_images(mu1, mu2, mu3, tau, raw_numpy_dir):
    # Helper function to format the numbers
#    def format_number(num):
#        return f'{num:.4f}' if num >= 1e-4 else f'{num:.0e}'
    
    # Format the input values
#    mu1_str = format_number(mu1)
#    mu2_str = format_number(mu2)
#    mu3_str = format_number(mu3)
#    tau_str = format_number(tau)
    
    # Construct the file names
    filename_R = f'raw_np_{mu1}_{mu2}_{mu3}_{mu4}_iter{iteration}_R.npy'
    filename_G = f'raw_np_{mu1}_{mu2}_{mu3}_{mu4}_iter{iteration}_G.npy'
    filename_B = f'raw_np_{mu1}_{mu2}_{mu3}_{mu4}_iter{iteration}_B.npy'
    
    # Load the numpy arrays
    image_R = np.load(os.path.join(raw_numpy_dir, filename_R))
    image_G = np.load(os.path.join(raw_numpy_dir, filename_G))
    image_B = np.load(os.path.join(raw_numpy_dir, filename_B))
    
    return image_R, image_G, image_B
    # Example usage:
    # raw_numpy_dir = 'path_to_your_numpy_directory'
    # image_R, image_G, image_B = load_images(0.0001, 0.0002, 0.0003, 0.0004, raw_numpy_dir)
    
def wave_to_cm(wavelength, gamma=0.8, A=1):
    ''' taken from http://www.noah.org/wiki/Wavelength_to_RGB_in_Python
    This converts a given wavelength of light to an 
    approximate RGB color value. The wavelength must be given
    in nanometers in the range from 380 nm through 750 nm
    (789 THz through 400 THz).

    Based on code by Dan Bruton
    http://www.physics.sfasu.edu/astro/color/spectra.html
    Additionally alpha value set to 0.5 outside range
    '''
    wavelength = float(wavelength)
    # if wavelength >= 380 and wavelength <= 750:
    #     A = 1.
    # else:
    #     A=0.5
    if wavelength < 380:
        wavelength = 380.
    if wavelength >750:
        wavelength = 750.
    if wavelength >= 380 and wavelength <= 440:
        attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
        R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
        G = 0.0
        B = (1.0 * attenuation) ** gamma
    elif wavelength >= 440 and wavelength <= 490:
        R = 0.0
        G = ((wavelength - 440) / (490 - 440)) ** gamma
        B = 1.0
    elif wavelength >= 490 and wavelength <= 510:
        R = 0.0
        G = 1.0
        B = (-(wavelength - 510) / (510 - 490)) ** gamma
    elif wavelength >= 510 and wavelength <= 580:
        R = ((wavelength - 510) / (580 - 510)) ** gamma
        G = 1.0
        B = 0.0
    elif wavelength >= 580 and wavelength <= 645:
        R = 1.0
        G = (-(wavelength - 645) / (645 - 580)) ** gamma
        B = 0.0
    elif wavelength >= 645 and wavelength <= 750:
        attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
        R = (1.0 * attenuation) ** gamma
        G = 0.0
        B = 0.0
    else:
        R = 0.0
        G = 0.0
        B = 0.0
    
    colorwave = (R,G,B,A)
    
    return LinearSegmentedColormap.from_list('my_list', [(0,0,0), colorwave] , N=200)