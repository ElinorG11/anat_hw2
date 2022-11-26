# imports for the HW
import numpy as np
import matplotlib.pyplot as plt
import cv2


# ---------------------------------------------- Q1 ---------------------------------------------- #

# Q1a - Load puppy image grayscale

puppy = cv2.imread(filename='../given_data/puppy.jpg')

puppy_rgb = cv2.cvtColor(puppy, cv2.COLOR_BGR2RGB)
puppy_grayscale = cv2.cvtColor(puppy, cv2.COLOR_BGR2GRAY)

fig11, axes = plt.subplots(1, 2, figsize=(10, 10))
axes[0].imshow(puppy_rgb)
axes[0].set_title("Adorable Puppy")
axes[1].imshow(puppy_grayscale, cmap='gray')
axes[1].set_title("Adorable Puppy in Grayscale")
plt.tight_layout()
plt.show()

# Q1b - Display Grayscale Puppy Histogram

hist, bins = np.histogram(puppy_grayscale.ravel(), 256, [0, 255])

fig = plt.figure()
plt.xlim([0, 255])
plt.plot(hist)
plt.title('Histogram of Puppy in Grayscale')

plt.show()

# Q1c
def gamma_correction(img, gamma):
    """
    Perform gamma correction on a grayscale image.
    :param img: An input grayscale image - ndarray of uint8 type.
    :param gamma: the gamma parameter for the correction.
    :return:
    gamma_img: An output grayscale image after gamma correction -
    uint8 ndarray of size [H x W x 1].
    """
    # ====== YOUR CODE: ======
    gamma_img = np.zeros(np.shape(img))

    # for gamma correction formula we assumed pixel values are in the range of [0,1]
    gamma_img = [(x/255) ** gamma for x in img]
    # now we should perform opposite transform from [0,1] to [0,255]
    gamma_img = np.array([x * 255 for x in gamma_img]).reshape(np.shape(img))
    # ========================

    return gamma_img


corrected_img_1 = gamma_correction(puppy_grayscale, 0.5)
corrected_img_2 = gamma_correction(puppy_grayscale, 1.5)

fig13, axes = plt.subplots(1, 3, figsize=(10, 10))
axes[0].imshow(puppy_grayscale, cmap='gray')
axes[0].set_title("Adorable Puppy in Grayscale")
axes[1].imshow(corrected_img_1, cmap='gray')
axes[1].set_title("Correction with Gamma=0.5")
axes[2].imshow(corrected_img_2, cmap='gray')
axes[2].set_title("Correction with Gamma=1.5")
plt.tight_layout()
plt.show()


# ---------------------------------------------- Q2 ---------------------------------------------- #

def video_to_frames(vid_path: str, start_second, end_second):
    """
    Load a video and return its frames from the wanted time range.
    :param vid_path: video file path.
    :param start_second: time of first frame to be taken from the
                         video in seconds.
    :param end_second: time of last frame to be taken from the
                         video in seconds.
    :return:
        frame_set: a 4D uint8 np array of size [num_of_frames x H x W x C]
                   containing the wanted video frames.
    """
    # ====== YOUR CODE: ======
    video = cv2.VideoCapture(vid_path)
    fps = int(video.get(cv2.CAP_PROP_FPS))
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = start_second * fps
    end_frame = end_second * fps

    frame_set = []

    for i in range(end_frame + 1):
        _, frame = video.read()
        if i >= start_frame:
            frame_set.append(frame)
            plt.imshow(frame)
    # ========================
    return frame_set


# fs = video_to_frames("/Users/erez/Documents/anat2/given_data/Corsica.mp4", 60 * 4 + 10, 60 * 4 + 20)

def match_corr(corr_obj, img):
    """
    return the center coordinates of the location of 'corr_obj' in 'img'.
    :param corr_obj: 2D numpy array of size [H_obj x W_obj]
                     containing an image of a component.
    :param img: 2D numpy array of size [H_img x W_img]
                where H_img >= H_obj and W_img>=W_obj,
                containing an image with the 'corr_obj' component in it.
    :return:
        match_coord: the two center coordinates in 'img'
                     of the 'corr_obj' component.
    """

    # ====== YOUR CODE: ======
    corr_obj = corr_obj.astype(np.float32)
    img = img.astype(np.float32)
    plt.imshow(corr_obj)
    plt.show()
    plt.imshow(img)
    plt.show()

    filtered_image = cv2.filter2D(img, 200, corr_obj, borderType=cv2.BORDER_CONSTANT)
    plt.imshow(filtered_image)
    plt.show()
    # ========================
    match_coord = 5
    return match_coord


# ---------------------------------------------- Q3 ---------------------------------------------- #

# Q3a1 - Plot Keyboard
keyboard = cv2.imread(filename='../given_data/keyboard.jpg')

keyboard_rgb = cv2.cvtColor(keyboard, cv2.COLOR_BGR2RGB)

fig311 = plt.figure(2)
plt.title("Keyboard")
plt.imshow(keyboard_rgb)
plt.show()

# Q3a2 - Define Kernels
kernel_horizontal = np.ones((1, 8), np.uint8)
kernel_vertical = np.ones((1, 8), np.uint8).T

# Q3a3 - Apply Erosion Filtering Using Different Kernels
# We expect that the horizontal filter will smooth vertical changes
# the vertical filter will smooth the horizontal changes
horizontal_erosion = cv2.erode(keyboard_rgb, kernel_horizontal)
vertical_erosion = cv2.erode(keyboard_rgb, kernel_vertical)

fig312, axes = plt.subplots(1, 2, figsize=(10, 10))
axes[0].imshow(horizontal_erosion)
axes[0].set_title("Keyboad Erosion using Horizontal Kernel")
axes[1].imshow(vertical_erosion)
axes[1].set_title("Keyboad Erosion using Vertical Kernel")
plt.tight_layout()
plt.show()

# Q3a4
eroded_summation = cv2.add(horizontal_erosion, vertical_erosion)

fig314 = plt.figure(3)
plt.title("Summation of Eroded pictures")
plt.imshow(eroded_summation)
plt.show()

binary_keyboard = cv2.threshold(eroded_summation, 0.2*255, 255, type=cv2.THRESH_BINARY)[1]

# Q3b
inverse_keyboard = cv2.threshold(eroded_summation, 0.2*255, 255, type=cv2.THRESH_BINARY_INV)[1]

keyboard_median_filtering = cv2.medianBlur(src=inverse_keyboard, ksize=9)

fig3134, axes = plt.subplots(1, 3, figsize=(10, 10))
axes[0].imshow(binary_keyboard, cmap='gray')
axes[0].set_title("Binary Keyboard")
axes[1].imshow(inverse_keyboard, cmap='gray')
axes[1].set_title("Inverse Keyboard")
axes[2].imshow(keyboard_median_filtering)
axes[2].set_title("Filtered Keyboad")
plt.tight_layout()

plt.show()

# Q3c
kernel_window = np.ones((8, 8), np.uint8)
rectangular_erosion = cv2.erode(keyboard_median_filtering, kernel_window)

fig33 = plt.figure(4)
plt.title("Second Erosion with Rectangular Kernel")
plt.imshow(rectangular_erosion)
plt.show()

# Q3d1
keyboard_contour = rectangular_erosion/255

multiplication_result = keyboard_contour.astype(np.uint8) * keyboard.astype(np.uint8)

# Q3d2
sharpening_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
sharpened_keyboard = cv2.filter2D(multiplication_result, -1, sharpening_kernel, cv2.BORDER_CONSTANT)

fig34 = plt.figure(5)
plt.title("Keyboard with Sharpened Keys Contour")
plt.imshow(sharpened_keyboard)
plt.show()

threshold = 0.8*255

denoised_sharpened_keyboared = cv2.threshold(sharpened_keyboard, threshold, 255, cv2.THRESH_BINARY)[1]
fig343, axes = plt.subplots(1, 2, figsize=(10, 10))
axes[0].imshow(sharpened_keyboard)
axes[0].set_title("Sharpened Keyboard")
axes[1].imshow(denoised_sharpened_keyboared)
axes[1].set_title("Denoised Sharpened Keyboard")
plt.tight_layout()
plt.show()


# ---------------------------------------------- Q4 ---------------------------------------------- #

frame = video_to_frames("../given_data/Flash Gordon Trailer.mp4", 20, 20)[0]

fig411 = plt.figure()
plt.imshow(frame)
plt.show()

red_channel_frame = frame[:, :, 2]
fig412 = plt.figure()
plt.imshow(red_channel_frame, cmap='gray')
plt.show()

width = int(red_channel_frame.shape[1] / 2)
height = int(red_channel_frame.shape[0] / 2)
dim = (width, height)
decreased_frame = cv2.resize(red_channel_frame, dim)
fig413 = plt.figure()
plt.imshow(decreased_frame, cmap='gray')
plt.show()


def poisson_noisy_image(X, a):
    """
    Creates a Poisson noisy image.
    :param X: The Original image. np array of size [H x W] and of type uint8.
    :param a: number of photons scalar factor
    :return:
        Y: The noisy image. np array of size [H x W] and of type uint8.
    """
    # ====== YOUR CODE: ======
    X = X.astype(float)
    X = X * a

    N = np.array(np.random.poisson(a, X.shape))
    N = N.astype(float)
    X = np.add(X, N)

    X = X / a

    X = np.clip(X, 0, 255)
    Y = X.astype(np.uint8)

    # ========================
    return Y


Y = poisson_noisy_image(decreased_frame, 3)
fig414 = plt.figure()
plt.title("Resized Image with Poisson Noise")
plt.imshow(Y, cmap='gray')
plt.show()


def denoise_by_l2(Y, X, num_iter, lambda_reg):
    """
    L2 image denoising.
    :param Y: The noisy image. np array of size [H x W]
    :param X: The Original image. np array of size [H x W]
    :param num_iter: the number of iterations for the algorithm perform
    :param lambda_reg: the regularization parameter
    :return:
    Xout: The restored image. np array of size [H x W]
    Err1: The error between Xk at every iteration and Y.
    np array of size [num_iter]
    Err2: The error between Xk at every iteration and X.
    np array of size [num_iter]
    """
    # ====== YOUR CODE: ======
    Ycs = np.asmatrix(Y).flatten('F')
    Xcs = Ycs
    D = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

    # M = (I + l * D.TD)Xk
    Xcs_matrix = np.reshape(Xcs, Y.shape, order='F')  # Xcs transformed into a matrix
    DX = cv2.filter2D(Xcs_matrix, -1, D)  # applying filter D on Xcs
    DDX = cv2.filter2D(DX, -1, D)  # applying filter D on DXcs
    M = (1 + lambda_reg) * DDX
    Mcs = np.asmatrix(M).flatten('F')
    Gcs = np.subtract(Mcs, Ycs)


    # N = (I + l * D.TD)G
    G = np.reshape(Gcs, Xcs.shape, order='F')  # Xcs transformed into a matrix
    DG = cv2.filter2D(G, -1, D)  # applying filter D on Xcs
    DDG = cv2.filter2D(DG, -1, D)  # applying filter D on DXcs
    N = (1 + lambda_reg) * DDG
    Ncs = np.asmatrix(N).flatten('F')

    nomenator = np.matmul(Gcs, np.transpose(Gcs))
    denomenator = np.matmul(Gcs, np.transpose(Ncs))

    mu = (nomenator / denomenator)[0, 0]

    Err1, Err2 = [], []

    diff_noisy = np.subtract(Xcs, Ycs)
    DXcs = np.asmatrix(DX).flatten('F')
    err1 = np.matmul(diff_noisy, np.transpose(diff_noisy)) + lambda_reg * np.matmul(DXcs, np.transpose(DXcs))

    Xcs_orig = np.asmatrix(X).flatten('F')
    diff_orig = np.subtract(Xcs, Xcs_orig)
    err2 = np.matmul(diff_orig, np.transpose(diff_orig))

    Err1.append(err1[0, 0])
    Err2.append(err2[0, 0])

    for i in range(num_iter - 1):

        # ------------------ UPDATE Gk ------------------ #
        # M = (I + l * D.TD)Xk
        Xcs_matrix = np.reshape(Xcs, Y.shape, order='F')  # Xcs transformed into a matrix
        DX = cv2.filter2D(Xcs_matrix, -1, D)  # applying filter D on Xcs
        DDX = cv2.filter2D(DX, -1, D)  # applying filter D on DXcs
        M = (1 + lambda_reg) * DDX
        Mcs = np.asmatrix(M).flatten('F')
        Gcs = np.subtract(Mcs, Ycs)

        # ------------------ UPDATE MUk ------------------ #
        # N = (I + l * D.TD)G
        G = np.reshape(Gcs, Xcs.shape, order='F')  # Xcs transformed into a matrix
        DG = cv2.filter2D(G, -1, D)  # applying filter D on Xcs
        DDG = cv2.filter2D(DG, -1, D)  # applying filter D on DXcs
        N = (1 + lambda_reg) * DDG
        Ncs = np.asmatrix(N).flatten('F')

        nomenator = np.matmul(Gcs, np.transpose(Gcs))
        denomenator = np.matmul(Gcs, np.transpose(Ncs))

        mu = (nomenator / denomenator)[0, 0]

        # ------------------ UPDATE Xk ------------------ #
        Xcs = Xcs - mu * Gcs

        # ------------------ UPDATE Error ------------------ #
        diff_noisy = Xcs - Ycs
        DXcs = np.asmatrix(DX).flatten('F')
        err1 = np.matmul(diff_noisy, np.transpose(diff_noisy)) + lambda_reg * np.matmul(DXcs, np.transpose(DXcs))

        Xcs_orig = np.asmatrix(X).flatten('F')
        diff_orig = Xcs - Xcs_orig
        err2 = np.matmul(diff_orig, np.transpose(diff_orig))

        Err1.append(err1[0, 0])
        Err2.append(err2[0, 0])

    Xout = Xcs
    # ========================
    return Xout, Err1, Err2


denoised_img, Err1, Err2 = denoise_by_l2(Y, decreased_frame, 50, 0.5)

fig421, ax = plt.subplots(figsize=(10, 10))
ax.imshow(denoised_img, cmap='gray')
ax.set_title("Denoised Image by L2")
plt.tight_layout()
plt.show()


fig422, ax = plt.subplots(figsize=(10, 10))
plt.plot(np.linspace(0, 50, 50), np.log(Err1), "-b", label="Err1")
plt.plot(np.linspace(0, 50, 50), np.log(Err2), "-r", label="Err2")
plt.legend(loc="upper left")
ax.set_title("Err1, Err2 vs Number of Iterations")
plt.tight_layout()
plt.show()
