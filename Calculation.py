# Wersja 02.11.2024

import napari
from skimage import io
import os
import numpy as np
from scipy.io import savemat
from scipy.io import loadmat
from scipy.ndimage import zoom
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from scipy.ndimage import median_filter
import tifffile
import h5py
import time

import matplotlib.pyplot as plt

# Wycięcie jedynie obaszru komórki
def masking_cell_area(image):
    # Wyciągnięcie macierzy zawierającej komórkę
    macierz_cell = image[:, 2, :, :]
    # Normalizacja pikseli do 1
    min_value_macierz_cell = np.min(macierz_cell) - 1 
    macierz_cell = macierz_cell - min_value_macierz_cell

    # Wyciągnięcie macierzy zawierającej maskę
    cell_area = image[:, 6, :, :]
    cell_area = cell_area/255

    # Maskowanie obszaru komórki
    macierz_cell = macierz_cell * cell_area
    return(macierz_cell, cell_area)

def znajdz_najlepszy_kwadrat(obraz):
    # Wymiary oryginalnego obrazu
    wysokosc = len(obraz)
    szerokosc = len(obraz[0])

    max_size = np.sum(obraz)
    
    # Ustal długość boku kwadratu jako długość krótszego boku obrazu
    bok_kwadratu = min(wysokosc, szerokosc)

    if bok_kwadratu > 220:
        bok_kwadratu = 220
    
    # Zmienna pomocnicza do przechowywania najlepszej liczby jedynek oraz współrzędnych
    max_jedynek = 0
    najlepsze_wspolrzedne = (0, 0)  # domyślnie lewy górny róg
    
    # Przesuwamy kwadrat o boku `bok_kwadratu` przez cały obraz, aby znaleźć najlepszy obszar
    for i in range(wysokosc - bok_kwadratu + 1):
        for j in range(szerokosc - bok_kwadratu + 1):
            # Liczymy liczbę jedynek w bieżącym kwadracie o wymiarach bok_kwadratu x bok_kwadratu

            licznik_jedynek = sum(
                obraz[x][y] 
                for x in range(i, i + bok_kwadratu) 
                for y in range(j, j + bok_kwadratu)
            )
            
            if licznik_jedynek == max_size:
                najlepsze_wspolrzedne = (i, j)
                # Wycinamy najlepszy kwadrat z obrazu
                i_start, j_start = najlepsze_wspolrzedne
                
                # Zwracamy współrzędne najlepszego kwadratu oraz przycięty obraz
                wspolrzedne_przycietego = (i_start, j_start, i_start + bok_kwadratu - 1, j_start + bok_kwadratu - 1)
                return wspolrzedne_przycietego

            # Aktualizujemy, jeśli znaleźliśmy większą liczbę jedynek
            if licznik_jedynek > max_jedynek:
                max_jedynek = licznik_jedynek
                najlepsze_wspolrzedne = (i, j)
    
    # Wycinamy najlepszy kwadrat z obrazu
    i_start, j_start = najlepsze_wspolrzedne
    
    # Zwracamy współrzędne najlepszego kwadratu oraz przycięty obraz
    wspolrzedne_przycietego = (i_start, j_start, i_start + bok_kwadratu - 1, j_start + bok_kwadratu - 1)
    return wspolrzedne_przycietego

# Selekcja kwadratu z komórką
def selsct_best_squere(cell_area):
    # Tworzymy tablicę Size_area i wypełniamy ją sumami dla każdego plastra
    '''Size_area = np.sum(cell_area, axis=(1, 2))

    # Znajdujemy indeks plastra o największej sumie pikseli
    max_index = np.argmax(Size_area)

    # Zapisujemy plaster o największej sumie do nowej zmiennej
    max_slice = cell_area[max_index]'''

    # Sumowanie wzdłuż pierwszego wymiaru
    cell_squeezed = np.sum(cell_area, axis=0)

    # Wywołanie funkcji
    wspolrzedne = znajdz_najlepszy_kwadrat(cell_squeezed)

    # max_slice_squre = max_slice[wspolrzedne[0] : wspolrzedne[2], wspolrzedne[1] : wspolrzedne[3]]
    return wspolrzedne

# Tworzenie maski i procesing
def load_mask_EW():
    # Ścierzka do pliku z zakresem częstości OCT
    file_path = './EW_int_240_60.mat'

    # Wczytanie pliku
    mask = loadmat(file_path)

    # Dostęp do zmiennej EW_int (przykładowo, jeśli taka istnieje w pliku)
    EW_int = mask['EW_int']

    '''# Ścierzka do pliku z zakresem częstości OCT
    file_path = './New_V2_EW.mat'

    # Wczytanie pliku
    mask = loadmat(file_path)

    # Dostęp do zmiennej EW_int (przykładowo, jeśli taka istnieje w pliku)
    EW_bool = mask['EW']

    # Sprawdzenie kształtu macierzy
    print(EW_bool.shape)'''

    # Zamiana wymiarów na (180, 720, 720)
    mask_matrix = np.transpose(EW_int, (2, 0, 1))

    # Zapisz jako wielowarstwowy plik TIFF
    tifffile.imwrite('mask_matrix_small.tiff', mask_matrix)

    # Odczytanie obrazu 3D z pliku TIFF
    #mask_matrix = tifffile.imread('mask_matrix_small.tiff')

    # Sprawdzenie nowego kształtu macierzy
    #print("Rozmiar maski:", mask_matrix.shape)
    return mask_matrix

# Interpolacja Kubiczna
'''def rescale_to_target_shape(matrix, target_shape):
    # Obliczenie współczynników skalowania
    scale_factors = tuple(t / o for t, o in zip(target_shape, matrix.shape))
    
    # Przeskalowanie macierzy
    rescaled_matrix = zoom(matrix, scale_factors, order=3)  # 'order=3' dla interpolacji kubicznej
    
    return rescaled_matrix'''

# Interpolacja najbliższego sąsiada
def rescale_to_target_shape(matrix, target_shape):
    """
    Funkcja skaluje macierz do podanego rozmiaru, nie wprowadzając wartości ujemnych.
    Wykorzystuje interpolację najbliższego sąsiada (order=0) dla minimalnego zniekształcenia danych.
    
    Args:
    - matrix (np.ndarray): Macierz 3D do przeskalowania.
    - target_shape (tuple): Docelowy rozmiar macierzy.
    
    Returns:
    - np.ndarray: Przeskalowana macierz 3D.
    """
    # Obliczenie współczynników skalowania dla każdego wymiaru
    scale_factors = tuple(t / o for t, o in zip(target_shape, matrix.shape))
    
    # Przeskalowanie macierzy z użyciem interpolacji najbliższego sąsiada
    rescaled_matrix = zoom(matrix, scale_factors, order=0)
    
    return rescaled_matrix

# Wykonaj transformatę Fouriera na macierzy 3D
def cut_Fourier_spectrum(rescaled_matrix_cell, mask_matrix):
    start_time = time.time()

    fourier_transform = np.fft.fftn(rescaled_matrix_cell)

    # Przesunięcie zero częstotliwości do środka
    fourier_transform_shifted = np.fft.fftshift(fourier_transform)

    # Wycięcie widma charakterystyczneo dla OCT
    masked_fourier_transform = fourier_transform_shifted * mask_matrix

    '''viewer.add_image(np.log(1 + np.abs(fourier_transform_shifted)) , name="Widmo logarytmiczne", colormap='viridis')
    viewer.add_image(np.log(1 + np.abs(masked_fourier_transform)) , name="Wtcięte widmo", colormap='viridis')'''

    # Przeprowadzenie odwrotnej FFT
    ifft_image_shifted = ifftshift(masked_fourier_transform)  # Przesunięcie odwrotne
    inverse_fourier_transform = np.fft.ifftn(ifft_image_shifted)

    # Sprawdzenie kształtu macierzy
    #print("Po Fourierze:", inverse_fourier_transform.shape)

    # Wyciągnij część rzeczywistą
    real_part_inverse_fourier_transform = np.real(inverse_fourier_transform)

    # Zapis macierzy OCT
    #tifffile.imwrite('OCT_matrix.tiff', real_part_inverse_fourier_transform)

    # Zakończenie pomiaru czasu
    end_time = time.time()

    # Wyliczenie czasu trwania
    execution_time = end_time - start_time
    print(f"Czas wykonania FFT i IFFT: {execution_time} sekund")
    return real_part_inverse_fourier_transform

# Zapis do jednego pliku HDF5
def save_to_H5_file(save_directory, file_name, rescaled_matrix_cell, real_part_inverse_fourier_transform):
    # Usunięcie końcówki '.ome.tif' i dodanie '_converted.h5'
    new_filename = file_name.replace(".ome.tif", "_converted.h5")#converted

    image_path = os.path.join(save_directory, new_filename)
    with h5py.File(image_path, 'w') as f:
        f.create_dataset('Cell', data = rescaled_matrix_cell) 
        f.create_dataset('OCT', data = real_part_inverse_fourier_transform)

# Opisz parametry macierzy
def describe(matrix):
    # Sprawdzenie rozmiaru macierzy
    rozmiar = matrix.shape

    # Obliczenie podstawowych statystyk
    maksymalna_wartosc = np.max(matrix)
    minimalna_wartosc = np.min(matrix)
    srednia_wartosc = np.mean(matrix)
    mediana = np.median(matrix)

    # Wyświetlenie wyników
    print("Informacje o macierzy real_part_inverse_fourier_transform:")
    print("Rozmiar:", rozmiar)
    print("Maksymalna wartość woksela:", maksymalna_wartosc)
    print("Minimalna wartość woksela:", minimalna_wartosc)
    print("Średnia wartość woksela:", srednia_wartosc)
    print("Mediana wartości wokseli:", mediana)
    print("\n")

def analyze_3d_matrix(matrix):
    # Sprawdzenie, czy macierz jest trójwymiarowa
    if matrix.ndim != 3:
        raise ValueError("Macierz musi mieć dokładnie trzy wymiary.")
    
    # Obliczanie podstawowych parametrów
    min_val = matrix.min()
    max_val = matrix.max()
    mean_val = matrix.mean()
    median_val = np.median(matrix)
    size = matrix.size
    data_type = matrix.dtype
    
    # Wyświetlanie podstawowych parametrów
    print("Podstawowe parametry macierzy:")
    print(f"Minimalna wartość: {min_val}")
    print(f"Maksymalna wartość: {max_val}")
    print(f"Średnia wartość: {mean_val}")
    print(f"Mediana: {median_val}")
    print(f"Rozmiar (liczba elementów): {size}")
    print(f"Typ danych: {data_type}")
    
    # Tworzenie histogramu wartości w macierzy
    plt.figure(figsize=(12, 7))
    
    # Histogram
    #plt.subplot(1, 2, 1)
    plt.hist(matrix.ravel(), bins=30, color='skyblue', edgecolor='black', range=(1, max_val))
    plt.title("Histogram wartości w macierzy")
    plt.xlabel("Wartości")
    plt.ylabel("Częstotliwość")
    
    # Box plot
    '''plt.subplot(1, 2, 2)
    # Filtrowanie danych: wybieramy tylko wartości powyżej 1
    filtered_data = matrix.ravel()[matrix.ravel() > 1]
    plt.boxplot(matrix.ravel(), vert=False, patch_artist=True, boxprops=dict(facecolor='lightgreen', color='darkgreen'))
    plt.title("Box plot wartości w macierzy")
    plt.xlabel("Wartości")'''
    
    # Wyświetlenie wykresów
    plt.tight_layout()
    plt.show()

# Path to the directory
data_directory = 'D:/Praca_Magisterska_Dane/Mitochondrium_34'

save_directory = 'D:/Praca_Magisterska_Dane/Mitochondrium_34_policzone'

# Wczytanie obrazów
#files = sorted([f for f in os.listdir(data_directory) if f.endswith('.tif')])

#how_many_files = len(files)

# Wczytanie maski EW
EW_mask = load_mask_EW()

whole_start_time = time.time()
cnt = 1
for i, file_name in enumerate(sorted(os.listdir(data_directory))):
    if file_name.endswith('.tif'):
        start_time = time.time()
        print("Nazwa pliku:", file_name, "  nr.:", cnt)#, "/", how_many_files)
        image_path = os.path.join(data_directory, file_name)
        # Wczytanie macierzy
        image = io.imread(image_path)

        cell_matrix, cell_area = masking_cell_area(image)
        
        coordinates = selsct_best_squere(cell_area)

        cell_matrix = cell_matrix[:, coordinates[0] : coordinates[2], coordinates[1] : coordinates[3]]

        # Zmiana typu danych na int16
        cell_matrix = cell_matrix.astype(np.int16)
        print("Rozmiar po wycieciu:", cell_matrix.shape)

        # Przeskalowanie macierzy do wymiaru maski
        rescaled_matrix_cell = rescale_to_target_shape(cell_matrix, EW_mask.shape)

        # Filtracja medianowa
        rescaled_matrix_cell = median_filter(rescaled_matrix_cell, size=(3, 3, 3))

        # Symulacja OCT
        OCT_cell_matrix = cut_Fourier_spectrum(rescaled_matrix_cell, EW_mask)

        OCT_cell_matrix = OCT_cell_matrix.astype(np.float32)
        
        # Zapis do pliku HDF5
        save_to_H5_file(save_directory, file_name, rescaled_matrix_cell, OCT_cell_matrix)
        
        # Zakończenie pomiaru czasu
        end_time = time.time()

        # Wyliczenie czasu trwania
        execution_time = end_time - start_time
        print(f"Czas wykonania obliczeń dla jednej komórki: {execution_time} sekund")
        print("\n")
        cnt += 1

whole_end_time = time.time()

# Wyliczenie czasu trwania
whole_execution_time = whole_end_time - whole_start_time
print(f"Czas wykonania wszystkichobliczeń: {whole_execution_time} sekund")

# Initialize the viewer
viewer = napari.Viewer()
viewer.add_image(OCT_cell_matrix , name="OCT", colormap='viridis')
print("Finished!")
# Start the napari viewer
napari.run()