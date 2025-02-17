import napari
from skimage import io
import os
import numpy as np
from scipy.io import savemat, loadmat
from scipy.ndimage import zoom, median_filter
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import tifffile
import h5py
import time
import matplotlib.pyplot as plt

class ImageProcessor:
    # Wycięcie jedynie obszaru komórki
    @staticmethod
    def masking_cell_area(image):
        macierz_cell = image[:, 2, :, :]
        min_value_macierz_cell = np.min(macierz_cell) - 1
        macierz_cell = macierz_cell - min_value_macierz_cell
        cell_area = image[:, 6, :, :] / 255
        macierz_cell = macierz_cell * cell_area
        return macierz_cell, cell_area

    # Znalezienie najlepszego kwadratu
    @staticmethod
    def znajdz_najlepszy_kwadrat(obraz):
        wysokosc, szerokosc = obraz.shape
        max_size = np.sum(obraz)
        bok_kwadratu = min(wysokosc, szerokosc, 220)
        max_jedynek = 0
        najlepsze_wspolrzedne = (0, 0)

        for i in range(wysokosc - bok_kwadratu + 1):
            for j in range(szerokosc - bok_kwadratu + 1):
                licznik_jedynek = np.sum(obraz[i:i + bok_kwadratu, j:j + bok_kwadratu])
                if licznik_jedynek == max_size:
                    return i, j, i + bok_kwadratu - 1, j + bok_kwadratu - 1
                if licznik_jedynek > max_jedynek:
                    max_jedynek = licznik_jedynek
                    najlepsze_wspolrzedne = (i, j)
        return najlepsze_wspolrzedne[0], najlepsze_wspolrzedne[1], najlepsze_wspolrzedne[0] + bok_kwadratu - 1, najlepsze_wspolrzedne[1] + bok_kwadratu - 1

    # Selekcja najlepszego kwadratu
    @staticmethod
    def selsct_best_squere(cell_area):
        cell_squeezed = np.sum(cell_area, axis=0)
        return ImageProcessor.znajdz_najlepszy_kwadrat(cell_squeezed)

    # Wczytanie maski EW
    @staticmethod
    def load_mask_EW(file_path):
        mask = loadmat(file_path)
        EW_int = mask['EW_int']
        mask_matrix = np.transpose(EW_int, (2, 0, 1))
        tifffile.imwrite('mask_matrix_small.tiff', mask_matrix)
        return mask_matrix

    # Interpolacja najbliższego sąsiada
    @staticmethod
    def rescale_to_target_shape(matrix, target_shape):
        scale_factors = tuple(t / o for t, o in zip(target_shape, matrix.shape))
        return zoom(matrix, scale_factors, order=0)

    # Wykonaj transformatę Fouriera na macierzy 3D
    @staticmethod
    def cut_Fourier_spectrum(rescaled_matrix_cell, mask_matrix):
        start_time = time.time()
        fourier_transform = np.fft.fftn(rescaled_matrix_cell)
        fourier_transform_shifted = np.fft.fftshift(fourier_transform)
        masked_fourier_transform = fourier_transform_shifted * mask_matrix
        ifft_image_shifted = ifftshift(masked_fourier_transform)
        inverse_fourier_transform = np.fft.ifftn(ifft_image_shifted)
        real_part_inverse_fourier_transform = np.real(inverse_fourier_transform)
        print(f"Czas wykonania FFT i IFFT: {time.time() - start_time} sekund")
        return real_part_inverse_fourier_transform

    # Zapis do pliku HDF5
    @staticmethod
    def save_to_H5_file(save_directory, file_name, rescaled_matrix_cell, real_part_inverse_fourier_transform):
        new_filename = file_name.replace(".ome.tif", "_converted.h5")
        image_path = os.path.join(save_directory, new_filename)
        with h5py.File(image_path, 'w') as f:
            f.create_dataset('Cell', data=rescaled_matrix_cell)
            f.create_dataset('OCT', data=real_part_inverse_fourier_transform)

class MainProcess:
    def __init__(self, data_directory, save_directory, mask_file_path):
        self.data_directory = data_directory
        self.save_directory = save_directory
        self.mask_file_path = mask_file_path

    def run(self):
        EW_mask = ImageProcessor.load_mask_EW(self.mask_file_path)
        whole_start_time = time.time()
        cnt = 1

        for file_name in sorted(os.listdir(self.data_directory)):
            if file_name.endswith('.tif'):
                start_time = time.time()
                print("Nazwa pliku:", file_name, "  nr.:", cnt)
                image_path = os.path.join(self.data_directory, file_name)
                image = io.imread(image_path)
                cell_matrix, cell_area = ImageProcessor.masking_cell_area(image)
                coordinates = ImageProcessor.selsct_best_squere(cell_area)
                cell_matrix = cell_matrix[:, coordinates[0]:coordinates[2], coordinates[1]:coordinates[3]]
                cell_matrix = cell_matrix.astype(np.int16)
                print("Rozmiar po wycieciu:", cell_matrix.shape)
                rescaled_matrix_cell = ImageProcessor.rescale_to_target_shape(cell_matrix, EW_mask.shape)
                rescaled_matrix_cell = median_filter(rescaled_matrix_cell, size=(3, 3, 3))
                OCT_cell_matrix = ImageProcessor.cut_Fourier_spectrum(rescaled_matrix_cell, EW_mask)
                OCT_cell_matrix = OCT_cell_matrix.astype(np.float32)
                ImageProcessor.save_to_H5_file(self.save_directory, file_name, rescaled_matrix_cell, OCT_cell_matrix)
                print(f"Czas wykonania obliczeń dla jednej komórki: {time.time() - start_time} sekund\n")
                cnt += 1

        print(f"Czas wykonania wszystkich obliczeń: {time.time() - whole_start_time} sekund")
        viewer = napari.Viewer()
        viewer.add_image(rescaled_matrix_cell, name="Cell", colormap='viridis')
        viewer.add_image(OCT_cell_matrix, name="OCT", colormap='viridis')
        print("Finished!")
        napari.run()

# Uruchomienie procesu
if __name__ == "__main__":
    # Ścieżki do katalogów
    data_directory = '.\Try'
    save_directory = '.\Try'
    mask_file_path = '.\EW_int_240_60.mat'
    process = MainProcess(data_directory, save_directory, mask_file_path)
    process.run()
