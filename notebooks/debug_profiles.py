import numpy as np

np.set_printoptions(precision=2, linewidth=150)


def is_approx_equal(x, y, epsilon=1e-3):
    return np.abs(x - y) / (np.sqrt(np.abs(x) * np.abs(y)) + epsilon) < epsilon


def compute_kernel_profiles_complete(kernel_size):
    radius_max = kernel_size // 2
    n_profiles = radius_max**2 + radius_max + 1
    x_grid = np.arange(-radius_max, radius_max + 1, 1)
    x, y = np.meshgrid(x_grid, x_grid)
    theta = (np.arctan2(y, x) + 2 * np.pi) % (2 * np.pi)
    r = np.sqrt(x**2 + y**2)
    kernel_profiles = np.zeros((kernel_size, kernel_size, n_profiles))
    kernel_profiles[kernel_size // 2, kernel_size // 2, 0] = 1
    theta_shifts = [k * np.pi / 2 for k in range(4)]
    profile_counter = 1
    for i in range(1, radius_max + 1):
        n_pixels = 8 * i
        d_theta = theta[np.where(((np.abs(x) == i) | (np.abs(y) == i))
                                 & (r <= np.sqrt(2) * i))]
        d_theta.sort()
        d_theta = d_theta[:n_pixels // 4]
        for dt in d_theta:
            shifts = (dt + np.array(theta_shifts)) % (2 * np.pi)
            for t in shifts:
                kernel_profiles[is_approx_equal(theta, t) &
                                ((np.abs(x) == i) |
                                 (np.abs(y) == i)) & (r <= np.sqrt(2) * i),
                                profile_counter] = 1
            profile_counter += 1

    return kernel_profiles


profiles = compute_kernel_profiles_complete(11)

print(profiles[:, :, 9])
