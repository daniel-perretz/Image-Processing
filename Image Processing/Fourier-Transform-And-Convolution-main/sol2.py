import matplotlib.pyplot as plt
import numpy as np
from imageio import imread
from imageio import imwrite
from scipy.io import wavfile
from skimage.color import rgb2gray
from scipy.io.wavfile import read,write
from scipy.signal import spectrogram
from scipy.signal import convolve2d
import sys
import numpy as np
from scipy import signal
from scipy.ndimage.interpolation import map_coordinates
import numpy as np
import math
from scipy import signal
from scipy.ndimage.interpolation import map_coordinates



def DFT(signal):
    return np.dot(
        np.exp(-2j * np.pi * np.arange(len(signal)).reshape((len(signal), 1)) *
               np.arange(len(signal)) / len(signal)), signal)


def IDFT(fourier_signal):
    sum_idft = np.dot(np.exp(2j * np.pi * np.arange(len(fourier_signal))
                             .reshape((len(fourier_signal), 1)) *
                             np.arange(len(fourier_signal))
                             / len(fourier_signal)), fourier_signal)

    return sum_idft / len(fourier_signal)


def DFT2(image):
    complex_img = image.astype(np.complex128)
    img_row = len(image)
    num_col = image[0]
    for row in range(img_row):
        complex_img[row] = DFT(image[row])

    return np.dot(
        np.exp(-2j * np.pi * np.arange(img_row).reshape((img_row, 1)) *
               np.arange(img_row) / img_row), complex_img)


def IDFT2(fourier_image):
    im = np.apply_along_axis(IDFT, 0, fourier_image)
    return np.apply_along_axis(IDFT, 1, im)

def change_rate(filename,ratio):
    samplerate,data = wavfile.read(filename)
    write("./change_rate.wav",int(samplerate*ratio),data)


def change_samples(filename,ratio):
    samplerate, data = wavfile.read(filename)
    new_data = np.real(resize(data,ratio))
    write("./change_samples.wav",samplerate,new_data.astype(np.int16))
    return new_data.astype(np.float64)




def resize(data,ratio):
    if(ratio>1):
        fourier_data = DFT(data)
        shifted_dft = np.fft.fftshift(fourier_data)
        crop_samples = int((len(data) / (ratio)))
        new_range = int((len(data) - crop_samples) / 2)
        high_freq = shifted_dft[new_range:new_range+crop_samples]
        high_freq = np.fft.ifftshift(high_freq)
        inverse_fourier = IDFT(high_freq)
    elif(ratio<1):
        fourier_data = DFT(data)
        shifted_dft = np.fft.fftshift(fourier_data)
        crop_samples = int(((len(data) / (ratio))))
        new_range = int((crop_samples - len(data)))
        each_side = int(new_range/2)
        pad_array = np.zeros(crop_samples).astype(np.complex128)
        # pad_array[0:each_side] = 0
        pad_array[each_side:each_side+len(data)] = shifted_dft
        # pad_array[each_side+len(data):] = 0
        high_freq = np.fft.ifftshift(pad_array)
        inverse_fourier = IDFT(high_freq)

    else:
        return data

    return inverse_fourier


def resize_spectrogram(data,ratio):
    sample_spectogram = stft(data)
    resized = np.apply_along_axis(resize,1,sample_spectogram,ratio)

    return  istft(resized)



def resize_vocoder(data,ratio):
    sample_spectogram = stft(data)
    return istft(phase_vocoder(sample_spectogram, ratio)).astype(np.float64)

def conv_der(im):
    kernel = np.array([[0.5,0,-0.5]])
    x_derivative = convolve2d(im,kernel,mode='same')
    y_derivative = convolve2d(im,kernel.T,mode='same')
    magnitude = np.sqrt(np.abs(x_derivative)**2 + np.abs(y_derivative)**2)
    return magnitude


def fourier_der(im):
    trans =  DFT2(im)
    shifted = np.fft.fftshift(trans)
    N = len(im)
    if N%2 == 0:
        u = np.arange((-N/2),(N/2)).reshape((N,1))
    else:
        u = np.arange((-N / 2), (N / 2)+1).reshape((N, 1))
    trans_rows = shifted *(2j*np.pi/N*u)
    idft_shifted_row = np.fft.ifftshift(trans_rows)
    idft_row = IDFT2(idft_shifted_row)

    M = len(im[0])
    if M%2 == 0:
        v = np.arange((-M/2),(M/2))
    else:
        v = np.arange((-M/2),(M/2)+1)
    trans_cols = shifted *(2j*np.pi/M*v)
    idft_shifted_col = np.fft.ifftshift(trans_cols)
    idft_col = IDFT2(idft_shifted_col)

    magnitude = np.sqrt(np.abs(idft_row)**2 + np.abs(idft_col)**2)
    return magnitude

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




def stft(y, win_length=640, hop_length=160):
    fft_window = signal.windows.hann(win_length, False)

    # Window the time series.
    n_frames = 1 + (len(y) - win_length) // hop_length
    frames = [y[s:s + win_length] for s in np.arange(n_frames) * hop_length]

    stft_matrix = np.fft.fft(fft_window * frames, axis=1)
    return stft_matrix.T


def istft(stft_matrix, win_length=640, hop_length=160):
    n_frames = stft_matrix.shape[1]
    y_rec = np.zeros(win_length + hop_length * (n_frames - 1), dtype=np.float)
    ifft_window_sum = np.zeros_like(y_rec)

    ifft_window = signal.windows.hann(win_length, False)[:, np.newaxis]
    win_sq = ifft_window.squeeze() ** 2

    # invert the block and apply the window function
    ytmp = ifft_window * np.fft.ifft(stft_matrix, axis=0).real

    for frame in range(n_frames):
        frame_start = frame * hop_length
        frame_end = frame_start + win_length
        y_rec[frame_start: frame_end] += ytmp[:, frame]
        ifft_window_sum[frame_start: frame_end] += win_sq

    # Normalize by sum of squared window
    y_rec[ifft_window_sum > 0] /= ifft_window_sum[ifft_window_sum > 0]
    return y_rec


def phase_vocoder(spec, ratio):
    num_timesteps = int(spec.shape[1] / ratio)
    time_steps = np.arange(num_timesteps) * ratio

    # interpolate magnitude
    yy = np.meshgrid(np.arange(time_steps.size), np.arange(spec.shape[0]))[1]
    xx = np.zeros_like(yy)
    coordiantes = [yy, time_steps + xx]
    warped_spec = map_coordinates(np.abs(spec), coordiantes, mode='reflect', order=1).astype(np.complex)

    # phase vocoder
    # Phase accumulator; initialize to the first sample
    spec_angle = np.pad(np.angle(spec), [(0, 0), (0, 1)], mode='constant')
    phase_acc = spec_angle[:, 0]

    for (t, step) in enumerate(np.floor(time_steps).astype(np.int)):
        # Store to output array
        warped_spec[:, t] *= np.exp(1j * phase_acc)

        # Compute phase advance
        dphase = (spec_angle[:, step + 1] - spec_angle[:, step])

        # Wrap to -pi:pi range
        dphase = np.mod(dphase - np.pi, 2 * np.pi) - np.pi

        # Accumulate phase
        phase_acc += dphase

    return warped_spec


# if __name__ == '__main__':
#     image = read_image('external/0ad8a422-2794-4d52-9103-73e3807994f5.jfif',1)
#     img = fourier_der(image)
#     plt.imshow(img)
#     plt.show()