import matplotlib.pyplot as plt
import numpy as np
import noise_repair


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
    img_array = plt.imread("test_img/img_sky.jpg")

    h = 5
    d_h = 5
    d_w = 5

    img_array_noise = make_noise(img_array, 0.15)   # Generate noise
    NP = noise_repair.Noise_repair(img_array_noise, alpha=1, max_iter=1000)  # Define NP
    NP.repair(h, d_h, d_w)  # Repair the image with noise

    display_arrays = [img_array_noise, NP.img_array_repaired, img_array]
    display_titles = ["img noisy", "img repaired", "img original"]
    plt.figure(figsize=(30, 30))
    display_imgs(display_arrays, display_titles, 1, 3)


if __name__ == "__main__":
    main()