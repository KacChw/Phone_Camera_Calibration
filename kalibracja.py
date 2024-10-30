import numpy as np
import cv2
import glob
import os

CHECKERBOARD = (7, 7)
square_size = 1.6  # jedno pole szachownicy w cm

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.01)

# współrzędne punktów szachownicy w przestrzeni 3D, uwzględniając rozmiar pola
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * square_size

# punkty 3D (świat) i punkty 2D (obraz)
objpoints = []
imgpoints = []

image_folder = 'input_data'
images = glob.glob(os.path.join(image_folder, '*.jpg'))

if len(images) == 0:
    print("Brak zdjęć w folderze 'input_data'. Upewnij się, że folder zawiera pliki .jpg.")
else:
    print(f"Liczba wczytanych zdjęć: {len(images)}")

for i, fname in enumerate(images):
    print(f"Przetwarzam zdjęcie {i + 1}/{len(images)}: {fname}")
    img = cv2.imread(fname)

    # resize
    img = cv2.resize(img, (1024, 768))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # funkcja znajdujaca rogi
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    print(f"Rogi znalezione: {ret}")

    if ret:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # wyświetlenie obrazu z narysowanymi rogami (dla testów)
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow('Szachownica', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# kalibracja kamery
if len(objpoints) > 0 and len(imgpoints) > 0:
    print("Rozpoczynam kalibrację kamery...")
    ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None,
                                                                      None)

    # wyniki kalibracji
    print("Macierz kamery:\n", cameraMatrix)
    print("Współczynniki dystorsji:\n", distCoeffs)
    print("Wektory rotacji (rvecs):\n", rvecs)
    print("Wektory translacji (tvecs):\n", tvecs)


    np.save('camera_matrix.npy', cameraMatrix)
    np.save('dist_coeffs.npy', distCoeffs)
    print("Kalibracja zakończona pomyślnie. Dane zostały zapisane do plików camera_matrix.npy oraz dist_coeffs.npy.")
else:
    print("Nie udało się znaleźć wystarczającej liczby rogów szachownicy na zdjęciach do przeprowadzenia kalibracji.")
