import numpy as np
# from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error as MSE


# Convert the patch into a one-dimensional vector
def patch_to_vector(patch_array):
    h = patch_array.shape[0]
    return patch_array.reshape(h * h, 3)


# Convert the vector back to patch
def vector_to_patch(vector, h):
    return vector.reshape(h, h, 3)


# Channel restoration of the standardized image for easy display
def channel_recover(img_array):
    return np.where(img_array == [-100, -100, -100], img_array, img_array * 256).astype(np.int32)


# Establish the lasso model corresponding to the three RGB channels
def get_lasso_models(dictionnaire, patch_noise, alpha=0.01, max_iter=1000):
    models = []
    dic = []
    for patch in dictionnaire:
        dic.append(patch_to_vector(patch))
    dic = np.array(dic)

    p_n = patch_to_vector(patch_noise)
    x_train_r, x_train_g, x_train_b, y_train_r, y_train_g, y_train_b = [], [], [], [], [], []

    noise = [-100., -100., -100.]
    for n in range(p_n.shape[0]):
        if (p_n[n] - noise).any():
            x_train_r.append(dic[:, n, 0])
            x_train_g.append(dic[:, n, 1])
            x_train_b.append(dic[:, n, 2])
            y_train_r.append(p_n[n, 0])
            y_train_g.append(p_n[n, 1])
            y_train_b.append(p_n[n, 2])

    model_r = Lasso(alpha=alpha, max_iter=max_iter).fit(x_train_r, y_train_r)
    model_g = Lasso(alpha=alpha, max_iter=max_iter).fit(x_train_g, y_train_g)
    model_b = Lasso(alpha=alpha, max_iter=max_iter).fit(x_train_b, y_train_b)

    # model_r = Ridge(alpha=alpha, max_iter=max_iter).fit(x_train_r, y_train_r)
    # model_g = Ridge(alpha=alpha, max_iter=max_iter).fit(x_train_g, y_train_g)
    # model_b = Ridge(alpha=alpha, max_iter=max_iter).fit(x_train_b, y_train_b)

    models.append(model_r)
    models.append(model_g)
    models.append(model_b)

    return np.array(models)


# Use the lasso model to make predictions
def repair_predict(models, dictionnaire, patch_noise):
    dic = []
    for patch in dictionnaire:
        dic.append(patch_to_vector(patch))
    dic = np.array(dic)

    p_n = patch_to_vector(patch_noise)
    x_test_r, x_test_g, x_test_b = [], [], []

    noise = [-100., -100., -100.]
    for n in range(p_n.shape[0]):
        if (p_n[n] - noise).any() == False:
            x_test_r.append(dic[:, n, 0])
            x_test_g.append(dic[:, n, 1])
            x_test_b.append(dic[:, n, 2])

    y_predict_r = models[0].predict(x_test_r)
    y_predict_g = models[1].predict(x_test_g)
    y_predict_b = models[2].predict(x_test_b)

    y_predict_rgb = list(zip(y_predict_r, y_predict_g, y_predict_b))
    y_predict_rgb = np.array([list(rgb) for rgb in y_predict_rgb])

    return y_predict_rgb


# Patch patches containing noise
def repatching(y_predict_rgb, patch_noise, h):
    noise = [-100., -100., -100.]
    p_n = patch_to_vector(patch_noise)
    i = 0
    for n in range(p_n.shape[0]):
        if not (p_n[n] - noise).any():
            p_n[n] = y_predict_rgb[i]
            i = i + 1
    p_n = vector_to_patch(p_n, h)
    return channel_recover(p_n)


# Evaluate the prediction results
def repair_score(y_test_rgb, y_predict_rgb):
    mse_r = MSE(y_predict_rgb[:, 0], y_test_rgb[:, 0])
    mse_g = MSE(y_predict_rgb[:, 1], y_test_rgb[:, 1])
    mse_b = MSE(y_predict_rgb[:, 2], y_test_rgb[:, 2])
    return (mse_r + mse_g + mse_b) / 3


# Calculation error rate
def error_rate(score, mean):
    return '%.2f%%' % ((score / mean) * 100)