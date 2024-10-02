import matplotlib.pyplot as plt
import scipy
from scipy.signal import convolve2d
import numpy as np
from scipy.ndimage.interpolation import map_coordinates
import math
from scipy import signal
from imageio import imread
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import os


def reduce(im, blur_filter):
    """
    Reduces an image by a factor of 2 using the blur filter
    :param im: Original image
    :param blur_filter: Blur filter
    :return: the downsampled image
    """

    # blur the image to help us get rid of the high freq
    # sub sample - select only every 2nd pixel in very 2nd row
    blured_image = scipy.ndimage.convolve(im, blur_filter)
    col_blured_img = scipy.ndimage.convolve\
        (blured_image, np.transpose(blur_filter))
    return col_blured_img[::2, ::2]


def expand(im, blur_filter):
    """
    Expand an image by a factor of 2 using the blur filter
    :param im: Original image
    :param blur_filter: Blur filter
    :return: the expanded image
    """
    new_row_size = im.shape[0]
    new_col_size = im.shape[1]
    np_array = np.array(blur_filter)
    fixed_filter = np.dot(np_array, 2)
    expanded_img = np.zeros(new_row_size * new_col_size * 4).reshape(
        new_row_size * 2, new_col_size * 2)
    expanded_img[0::2, 0::2] = im
    blured_and_expanded = scipy.ndimage.convolve(expanded_img, fixed_filter)
    return scipy.ndimage.convolve \
        (blured_and_expanded, np.transpose(fixed_filter))


def build_filter(filter_size):
    filter = [[1, 1]]
    conv = [[1, 1]]
    for i in range(filter_size - 2):
        filter = convolve2d(filter, conv)
    filter = filter / sum(filter[0])
    return filter


def build_gaussian_pyramid(im, max_levels, filter_size):
    """
    Builds a gaussian pyramid for a given image
    :param im: a grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter
            (an odd scalar that represents a squared filter)
            to be used in constructing the pyramid filter
    :return: pyr, filter_vec. Where pyr is the resulting pyramid as a
            standard python array with maximum length of max_levels,
            where each element of the array is a grayscale image.
            and filter_vec is a row vector of shape (1, filter_size)
            used for the pyramid construction.
    """
    filter = build_filter(filter_size)
    pyr = [im]
    for i in range(max_levels-1):
        temp_img = reduce(im, filter)
        if len(temp_img[0]) < 16 or len(temp_img) < 16:
            break
        else:
            pyr.append(temp_img)
            im = temp_img
    return pyr, filter


def build_laplacian_pyramid(im, max_levels, filter_size):
    """
    Builds a laplacian pyramid for a given image
    :param im: a grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid.
    :param filter_size: the size of the Gaussian filter
            (an odd scalar that represents a squared filter)
            to be used in constructing the pyramid filter
    :return: pyr, filter_vec. Where pyr is the resulting pyramid as a
            standard python array with maximum length of max_levels,
            where each element of the array is a grayscale image.
            and filter_vec is a row vector of shape (1, filter_size)
            used for the pyramid construction.
    """
    laplacian_pyr = []
    pyr = build_gaussian_pyramid(im, max_levels, filter_size)[0]
    filter_vec = build_gaussian_pyramid(im, max_levels, filter_size)[1]
    for i in range(len(pyr) - 1):
        new_pad_pyr = pyr[i]
        if (len(pyr[i]) < len(expand(pyr[i + 1], filter_vec))):
            new_pad_pyr = np.pad(pyr[i],
                                 (0, 1), 'constant', constant_values=0)[:, :-1]

        if (len(pyr[i][0]) < len(expand(pyr[i + 1], filter_vec)[0])):
            new_pad_pyr = np.pad(new_pad_pyr,
                                 (0, 1), 'constant', constant_values=0)[:-1]
        laplacian_pyr.append(new_pad_pyr - expand(pyr[i + 1], filter_vec))
    laplacian_pyr.append(pyr[len(pyr) - 1])
    return laplacian_pyr, filter_vec


def laplacian_to_image(lpyr, filter_vec, coeff):
    """

    :param lpyr: Laplacian pyramid
    :param filter_vec: Filter vector
    :param coeff: A python list in the same length as the number of levels in
            the pyramid lpyr.
    :return: Reconstructed image
    """

    for i in range(len(lpyr)):
        lpyr[i] = lpyr[i] * coeff[i]
    new_pad_pyr = expand(lpyr[-1],filter_vec)
    if (len(lpyr[-2]) < len(expand(lpyr[-1], filter_vec))):
        new_pad_pyr = new_pad_pyr[:-2]

    if (len(lpyr[-2][0]) < len(expand(lpyr[-1], filter_vec)[0])):
        new_pad_pyr = new_pad_pyr[:, :-2]

    new_pad_pyr = lpyr[-2] + new_pad_pyr
    im =  new_pad_pyr
    for i in range(len(lpyr) - 2, 0, -1):
        new_pad_pyr = expand(im, filter_vec)
        if (len(lpyr[i - 1]) < len(expand(im, filter_vec))):
            new_pad_pyr = new_pad_pyr[:-2]

        if (len(lpyr[i-1][0]) < len(expand(im, filter_vec)[0])):
            new_pad_pyr = new_pad_pyr[:, :-2]

        new_pad_pyr = lpyr[i-1] + new_pad_pyr
        im = new_pad_pyr

    return im


def render_pyramid(pyr, levels):
    """
    Render the pyramids as one large image with 'levels' smaller images
        from the pyramid
    :param pyr: The pyramid, either Gaussian or Laplacian
    :param levels: the number of levels to present
    :return: res a single black image in which the pyramid levels of the
            given pyramid pyr are stacked horizontally.
    """
    for i in range(levels):
        pyr[i] = clipping(pyr[i])
    row_size = pyr[0].shape[0]
    col_size = 0
    for i in range(levels):
        col_size += pyr[i].shape[1]
    res = np.zeros(row_size*col_size).reshape(row_size,col_size)
    start_col = 0
    for i in range(levels):
        end_row = pyr[i].shape[0]
        res[0:end_row, start_col:start_col+pyr[i].shape[1]] = pyr[i]
        start_col += pyr[i].shape[1]

    return res


def display_pyramid(pyr, levels):
    """
	display the rendered pyramid
	"""

    res = render_pyramid(pyr,levels)
    plt.imshow(res,cmap="gray")
    plt.show()


def clipping(img):
    """
    clipping an image value to a [0,1] values
    :param img: The image to clip
    :return: the normalized image
    """
    min_value = np.min(img)

    max_value = np.max(img)
    return (img - min_value)/ (max_value - min_value)


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im,
                     filter_size_mask):
    """
     Pyramid blending implementation
    :param im1: input grayscale image
    :param im2: input grayscale image
    :param mask: a boolean mask
    :param max_levels: max_levels for the pyramids
    :param filter_size_im: is the size of the Gaussian filter (an odd
            scalar that represents a squared filter)
    :param filter_size_mask: size of the Gaussian filter(an odd scalar
            that represents a squared filter) which defining the filter used
            in the construction of the Gaussian pyramid of mask
    :return: the blended image
    """
    first_laplacian_pyr = build_laplacian_pyramid(im1,max_levels,
                                                  filter_size_im)[0]
    second_laplacian_pyr = build_laplacian_pyramid(im2, max_levels,
                                                  filter_size_im)[0]
    mask = mask.astype(np.float64)
    mask_gausian = build_gaussian_pyramid(mask,max_levels,filter_size_mask)[0]
    laplacian_out = first_laplacian_pyr

    for k in range(len(first_laplacian_pyr)):
        laplacian_out[k] = mask_gausian[k]*first_laplacian_pyr[k] + \
                           (1-mask_gausian[k])*second_laplacian_pyr[k]

    coeff = np.ones(len(laplacian_out))
    filter_vec = build_filter(filter_size_im)
    img = laplacian_to_image(laplacian_out,filter_vec,coeff)

    clipped_img = np.clip(img,0,1)

    return clipped_img

def blending_example1():
    """
    Perform pyramid blending on two images RGB and a mask
    :return: image_1, image_2 the input images, mask the mask
        and out the blended image
    """
    im1 = read_image(relpath("external/thanos.png"),2)
    im2 = read_image(relpath("external/avital.png"),2)
    mask = read_image(relpath("external/mask.png"),1).astype(np.bool_)

    return_img = np.copy(im2)
    for i in range(3):
        return_img[:,:,i] = pyramid_blending(im2[:,:,i],im1[:,:,i],mask,10,
                                             5,5)

    img_lst = [[],im1,im2,mask,return_img]
    img_figures = plt.figure()
    for i in range(1,len(img_lst)):
        img_figures.add_subplot(2,2,i)
        plt.imshow(img_lst[i], cmap="gray")
        plt.axis('off')
    plt.show()
    return im1, im2, mask, return_img


def blending_example2():
    """
    Perform pyramid blending on two images RGB and a mask
    :return: image_1, image_2 the input images, mask the mask
        and out the blended image
    """
    im1 = read_image(relpath("external/ironman2.png"),2)
    im2 = read_image(relpath("external/goat2.png"),2)
    mask = read_image(relpath("external/maskgoat2.png"),1).astype(np.bool_)
    return_img = np.copy(im2)
    for i in range(3):
        return_img[:, :, i] = pyramid_blending(im2[:, :, i], im1[:, :, i],
                                               mask, 10,
                                               5, 5)

    img_lst = [[],im1,im2,mask,return_img]
    img_figures = plt.figure()
    for i in range(1,len(img_lst)):
        img_figures.add_subplot(2,2,i)
        plt.imshow(img_lst[i], cmap="gray")
        plt.axis('off')
    plt.show()
    return im1, im2, mask, return_img


def read_image(filename, representation):
    """
    Reads an image and converts it into a given representation
    :param filename: filename of image on disk
    :param representation: 1 for greyscale and 2 for RGB
    :return: Returns the image as an np.float64 matrix normalized to [0,1]
    """
    im = imread(filename)  # rgb return type
    if (representation == 1 and im.ndim == 3):
        gray_scale_img = rgb2gray(im)
        return np.float64(gray_scale_img)
    else:

        return np.float64(im / 255)

def relpath(filename):
    return os.path.join(os.path.dirname(__file__), filename)


if __name__ == '__main__':
    blending_example2()