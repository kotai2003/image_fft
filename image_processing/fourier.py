import cv2
import numpy as np

class FF7():
    def __int__(self):
        print('FF7 is loaded.')

    def _min_max(self, x):
        return (x - x.min()) / (x.max() - x.min())

    def FFT_HPF_filter(self, img, freq=0.1):
        '''
        FFT + HPF image program
        :param img: numpy array, should be gray scale image
        :dft_shift : magnitude_spectrum of FFT image before masking
        :param r: frequency range (0,1)
        :return:
        '''
        # FFT
        dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = 20 * np.log((cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])) + 1)

        # HPF mask
        rows, cols = img.shape
        crow, ccol = int(rows / 2), int(cols / 2)

        mask = np.ones((rows, cols, 2), np.uint8)

        # mask radius r
        if rows >= cols:
            r = int(0.5 * freq * cols)
        else:
            r = int(0.5 * freq * rows)

        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r
        mask[mask_area] = 0
        fshift = dft_shift * mask
        fshift_mask_mag = 2000 * np.log((cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1])) + 1)

        # Inverse FFT
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
        # min_max_scaler between 0 and 255
        img_back = (127 * self._min_max(img_back) + 127).astype(np.uint8)

        return magnitude_spectrum, fshift_mask_mag, img_back
