import pickle
import random
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch


def pkl_save(name, var):
    with open(name, 'wb') as f:
        pickle.dump(var, f)


def pkl_load(name):
    with open(name, 'rb') as f:
        return pickle.load(f)


def torch_pad_nan(arr, left=0, right=0, dim=0):
    if left > 0:
        padshape = list(arr.shape)
        padshape[dim] = left
        arr = torch.cat((torch.full(padshape, np.nan), arr), dim=dim)
    if right > 0:
        padshape = list(arr.shape)
        padshape[dim] = right
        arr = torch.cat((arr, torch.full(padshape, np.nan)), dim=dim)
    return arr


def pad_nan_to_target(array, target_length, axis=0, both_side=False):
    assert array.dtype in [np.float16, np.float32, np.float64]
    pad_size = target_length - array.shape[axis]
    if pad_size <= 0:
        return array
    npad = [(0, 0)] * array.ndim
    if both_side:
        npad[axis] = (pad_size // 2, pad_size - pad_size // 2)
    else:
        npad[axis] = (0, pad_size)
    return np.pad(array, pad_width=npad, mode='constant', constant_values=np.nan)


def split_with_nan(x, sections, axis=0):
    assert x.dtype in [np.float16, np.float32, np.float64]
    arrs = np.array_split(x, sections, axis=axis)
    target_length = arrs[0].shape[axis]
    for i in range(len(arrs)):
        arrs[i] = pad_nan_to_target(arrs[i], target_length, axis=axis)
    return arrs


def take_per_row(A, indx, num_elem):
    all_indx = indx[:, None] + np.arange(num_elem)
    return A[torch.arange(all_indx.shape[0])[:, None], all_indx]


def centerize_vary_length_series(x):
    prefix_zeros = np.argmax(~np.isnan(x).all(axis=-1), axis=1)
    suffix_zeros = np.argmax(~np.isnan(x[:, ::-1]).all(axis=-1), axis=1)
    offset = (prefix_zeros + suffix_zeros) // 2 - prefix_zeros
    rows, column_indices = np.ogrid[:x.shape[0], :x.shape[1]]
    offset[offset < 0] += x.shape[1]
    column_indices = column_indices - offset[:, np.newaxis]
    return x[rows, column_indices]


def data_dropout(arr, p):
    B, T = arr.shape[0], arr.shape[1]
    mask = np.full(B * T, False, dtype=np.bool)
    ele_sel = np.random.choice(
        B * T,
        size=int(B * T * p),
        replace=False
    )
    mask[ele_sel] = True
    res = arr.copy()
    res[mask.reshape(B, T)] = np.nan
    return res


def name_with_datetime(prefix='default'):
    now = datetime.now()
    return prefix + '_' + now.strftime("%Y%m%d_%H%M%S")


def init_dl_program(
        device_name,
        seed=None,
        use_cudnn=True,
        deterministic=False,
        benchmark=False,
        use_tf32=False,
        max_threads=None
):
    import torch
    if max_threads is not None:
        torch.set_num_threads(max_threads)  # intraop
        if torch.get_num_interop_threads() != max_threads:
            torch.set_num_interop_threads(max_threads)  # interop
        try:
            import mkl
        except:
            pass
        else:
            mkl.set_num_threads(max_threads)

    if seed is not None:
        random.seed(seed)
        seed += 1
        np.random.seed(seed)
        seed += 1
        torch.manual_seed(seed)

    if isinstance(device_name, (str, int)):
        device_name = [device_name]

    devices = []
    for t in reversed(device_name):
        t_device = torch.device(t)
        devices.append(t_device)
        if t_device.type == 'cuda':
            assert torch.cuda.is_available()
            torch.cuda.set_device(t_device)
            if seed is not None:
                seed += 1
                torch.cuda.manual_seed(seed)
    devices.reverse()
    torch.backends.cudnn.enabled = use_cudnn
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark

    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = use_tf32
        torch.backends.cuda.matmul.allow_tf32 = use_tf32

    return devices if len(devices) > 1 else devices[0]


def extract_srta_timeseries(df, frequency_hz, window_length_sec, window_overlap_coef):
    window_size = int(frequency_hz * window_length_sec)
    overlap_size = int(window_size * window_overlap_coef)
    step_size = window_size - overlap_size

    all_segments = []
    all_labels = []

    for mission_name in df['mission_name'].unique():
        mission_data = df[df['mission_name'] == mission_name].sort_index()

        # Drop 'mission_name' and 'timestamp' columns
        columns_to_drop = ['mission_name']
        mission_data = mission_data.drop(columns=columns_to_drop)

        num_samples = len(mission_data)
        num_windows = (num_samples - window_size) // step_size + 1

        for start in range(0, num_windows * step_size, step_size):
            end = start + window_size
            window_data = mission_data.iloc[start:end]

            if len(window_data) == window_size:
                all_segments.append(window_data.values)
                all_labels.append(mission_name)

    # Convert to numpy arrays
    X = np.array(all_segments)
    y = np.array(all_labels)
    feature_names = mission_data.columns.tolist()

    return X, y, feature_names


def plot_random_window(X, y, feature_names):
    random_index = np.random.randint(0, len(X))
    window_data = X[random_index]
    window_class = y[random_index]

    num_features = window_data.shape[1]

    fig, axes = plt.subplots(num_features, 1, figsize=(10, 2 * num_features), sharex=True)

    for i in range(num_features):
        axes[i].plot(window_data[:, i])
        axes[i].set_ylabel(feature_names[i])
        axes[i].grid(True)

    axes[-1].set_xlabel('Time Steps')
    fig.suptitle(f'Random Window {random_index} - Class: {window_class}')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
