import numpy as np
from patch_predict import get_lasso_models, repair_predict, repatching
from img_tools_backup import channel_normalize, get_all_patch
from img_tools_backup import get_patch_bruite, get_dictionnaire
import time


class Noise_repair:
    def __init__(self, img_array_noise, alpha=0.01, max_iter=1000):
        self.img_array_noise = img_array_noise
        self.alpha = alpha
        self.max_iter = max_iter
        self.dictionnaire = []
        self.patchs_noise = []
        self.patchs_noise_coords = []
        self.patchs_repaired = []
        self.img_array_repaired = img_array_noise.copy()

    def get_dict_and_noise(self, h, d_h, d_w):
        img_noise_normalized = channel_normalize(self.img_array_noise)
        patchs = get_all_patch(h, img_noise_normalized, d_w, d_h)
        self.dictionnaire = get_dictionnaire(patchs)
        self.patchs_noise, self.patchs_noise_coords = get_patch_bruite(patchs)

    def get_all_patchs_repaired(self, h):
        start = time.time()
        for patch_noise in self.patchs_noise:
            models = get_lasso_models(self.dictionnaire, patch_noise, self.alpha, self.max_iter)
            y_predict_rgb = repair_predict(models, self.dictionnaire, patch_noise)
            self.patchs_repaired.append(repatching(y_predict_rgb, patch_noise, h))
        end = time.time()
        times = end - start
        print("%.2f %s" % (times, "s"))

    def noise_remove(self, h):
        n = 0
        for coord in self.patchs_noise_coords:
            i = coord[0]
            j = coord[1]
            offset = int(np.floor(h / 2))
            start_i = i - offset
            start_j = j - offset
            self.img_array_repaired[start_i:start_i + h, start_j:start_j + h] = self.patchs_repaired[n]
            n += 1

    def repair(self, h, d_h, d_w):
        self.get_dict_and_noise(h, d_h, d_w)
        self.get_all_patchs_repaired(h)
        self.noise_remove(h)
