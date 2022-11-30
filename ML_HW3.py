import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from patch_predict import get_lasso_models, repair_predict, repatching, repair_score, error_rate, channel_recover


# Standardize
def channel_normalize(img_array):
    img_array_rgb = img_array
    if(img_array.dtype == 'float32' or img_array.dtype == 'float64'):
        img_array_rgb = matplotlib.colors.hsv_to_rgb(img_array).astype(np.int32)
    return np.where(img_array_rgb==[-100,-100,-100],img_array_rgb,img_array_rgb/256)


# Get a patch with i, j as the center and width and height as h
def get_patch(i, j, h, img_array):
    height = img_array.shape[0]
    width = img_array.shape[1]
    if (i < h / 2 - 1 or j < h / 2 - 1 or i > height - h / 2 or j > width - h / 2):
        print("i or j out of range !!!")
    else:
        row_start = int(i - np.floor(h / 2))
        row_end = int(i + np.ceil(h / 2))
        col_start = int(j - np.floor(h / 2))
        col_end = int(j + np.ceil(h / 2))
        return img_array[row_start:row_end, col_start:col_end]


def get_all_patch(h,img_array,d_h=1,d_w=1):
    height = img_array.shape[0]
    width = img_array.shape[1]
    if (height-h)%d_h != 0 or (width-h)%d_w != 0:
        print("distance between patchs not match !!!")
    else:
        start_i = int(np.ceil(h/2-1))
        end_i = int(np.ceil(height-h/2))
        start_j = int(np.ceil(h/2-1))
        end_j = int(np.ceil(width-h/2))
        all_patch = []
        all_patch_coord = []
        for i in range(start_i, end_i, d_h):
            for j in range(start_j, end_j, d_w):
                all_patch.append(get_patch(i,j,h,img_array))
                all_patch_coord.append([i,j])
        return np.array(all_patch),np.array(all_patch_coord)


# Add random noise to the image
def make_noise(img_array, prc):
    img_array = np.require(img_array, dtype='f4', requirements=['O', 'W']).astype(np.int32)
    nb_row = img_array.shape[0]
    nb_col = img_array.shape[1]
    nb_noise = int(nb_row*nb_col*prc)
    row_index = np.arange(img_array.shape[0])
    col_index = np.arange(img_array.shape[1])
    coord = np.transpose([np.tile(row_index, nb_col), np.repeat(col_index, nb_row)])
    coord_index = np.random.choice(coord.shape[0],nb_noise,replace=False)
    coord_noise = coord[coord_index]
    coord_noise = np.hsplit(coord_noise,2)
    img_array[coord_noise[0],coord_noise[1]] = [-100,-100,-100]
    return img_array


def get_noise_pixels_coords(img_array):
    coord_h,coord_w = [],[]
    for i in range(img_array.shape[0]):
        for j in range(img_array.shape[1]):
            if not (img_array[i, j] - [-100, -100, -100]).any():
                coord_h.append(i)
                coord_w.append(j)
    return np.array(coord_h), np.array(coord_w)


def get_true_values(img_array_noise, img_array):
    coord = get_noise_pixels_coords(img_array_noise)
    return img_array[coord]


def lasso_train(diction, patch_noise, alpha, h, y_test_rgb, errors, img_display_arrays, y_predicts):
    models = get_lasso_models(diction, channel_normalize(patch_noise), alpha, max_iter=1000)
    y_predict_rgb = repair_predict(models, diction, channel_normalize(patch_noise))
    patch_repaired = repatching(y_predict_rgb, channel_normalize(patch_noise), h)
    score = repair_score(channel_normalize(y_test_rgb), y_predict_rgb)
    error = error_rate(score, channel_normalize(y_test_rgb).mean())
    errors.append(error)
    img_display_arrays.append(patch_repaired)
    y_predicts.append(channel_recover(y_predict_rgb))
    return models


# Display some images once
def display_imgs(img_arrays,titles,h,w):
    for i in range(len(img_arrays)):
        plt.subplot(h,w,i+1)
        img = img_arrays[i].copy()
        img = np.where(img == [-100, -100, -100], [0, 0, 0], img)
        if titles[i] != None:
            plt.title(titles[i])
        plt.imshow(img)
    plt.show()


def main():
    img_array = plt.imread("test_img/img_beach.jpg")

    h = 25     # The size of the patch
    d_h = 25   # The vertical distance between different patches
    d_w = 25   # The horizontal distance between different patches
    dictionnaire = get_all_patch(h, channel_normalize(img_array), d_h, d_w)[0]

    # Get a patch and make noise
    patch = get_patch(150, 200, 25, img_array)
    patch_noise = make_noise(patch, 0.4)

    y_test_rgb = get_true_values(patch_noise, patch)

    errors = []
    img_display_arrays = []
    y_predicts = []
    alphas = [0.1, 0.005, 0.001, 0.0005, 0.0001, ]  # Find a suitable alpha for lasso
    # alphas = [50, 10, 5, 1, 0.1, ]  # Find a suitable alpha for ridge
    for alpha in tqdm(alphas):
        lasso_train(dictionnaire, patch_noise, alpha, h, y_test_rgb, errors, img_display_arrays, y_predicts)
    errors = np.array(errors)
    img_display_arrays = np.array(img_display_arrays)
    y_predicts = np.array(y_predicts)

    plt.figure(figsize=(25, 35))
    img_display_titles_rep = ["alpha = 0.1","alpha = 0.005","alpha = 0.001","alpha = 0.0005","alpha = 0.0001",
                              "alpha = 0.00001","alpha = 0.0000001"]
    display_imgs(img_display_arrays, img_display_titles_rep, 1, 5)
    print(errors)
    print(y_predicts)


if __name__ == "__main__":
    main()