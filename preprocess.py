import numpy as np

def prepare_input(features_flat, expected_timesteps=100, num_features=4):
    data = np.array(features_flat).reshape(-1, num_features)

    if data.shape[0] > expected_timesteps:
        data = data[:expected_timesteps]
    elif data.shape[0] < expected_timesteps:
        pad_size = expected_timesteps - data.shape[0]
        padding = np.zeros((pad_size, num_features))
        data = np.vstack((data, padding))

    return np.expand_dims(data, axis=0)  # shape: (1, timesteps, features)
