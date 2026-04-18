# %%
import os

import h5py
import numpy as np
import scipy.io

# %%

data_dir = "/home/user/桌面/ftd_dataset/basedata/"
output_file = os.path.join(data_dir, "ftd_lstm.h5")

# %%
with h5py.File(output_file, "w") as hdf5_file:
    altitude_folders = os.listdir(data_dir)
    ftd_data_list = {}
    for altitude_folder in altitude_folders:
        altitude_dir = os.path.join(data_dir, altitude_folder)
        if not os.path.isdir(altitude_dir):
            continue
        altitude_group = hdf5_file.create_group(altitude_folder)
        action_folders = os.listdir(altitude_dir)
        altitude_data_list = {}
        for action_folder in action_folders:
            action_dir = os.path.join(altitude_dir, action_folder)
            if os.path.isdir(action_dir):
                action_group = altitude_group.create_group(action_folder)
                files = os.listdir(action_dir)
                ftd_data = [
                    scipy.io.loadmat(os.path.join(action_dir, file))["fdata"][
                        :, :106
                    ]
                    for file in files
                    if ".mat" in file
                ]
                ftd_data = np.vstack(ftd_data)
                print(
                    f"file: {os.path.join(action_dir)},shape: {ftd_data.shape}"
                )
                x = np.hstack(
                    (ftd_data[:, :7], ftd_data[:, 12:14], ftd_data[:, 15:19],
                     ftd_data[:, 23:25], ftd_data[:, 27:28], ftd_data[:, 29:30], ftd_data[:, 44:45],
                     ftd_data[:, 46:47], ftd_data[:, 55:57], ftd_data[:, 81:82], ftd_data[:, 83:88],
                     ftd_data[:, 89:93])
                )
                y = np.hstack((ftd_data[:, 61:62],ftd_data[:, 63:67],ftd_data[:, 68:69]))

                action_group.create_dataset("x", data=x)
                action_group.create_dataset("y", data=y)
            elif ".mat" in action_dir:
                ftd_data = scipy.io.loadmat(action_dir)["fdata"][:, :106]
                print(
                    f"file: {os.path.join(action_dir)},shape: {ftd_data.shape}"
                )
                parts = action_dir.split("_")
                action_group = altitude_group.create_group(parts[-2])

                x = np.hstack(
                    (ftd_data[:, :7], ftd_data[:, 12:14], ftd_data[:, 15:19],
                     ftd_data[:, 23:25], ftd_data[:, 27:28],ftd_data[:, 29:30], ftd_data[:, 44:45],
                     ftd_data[:, 46:47], ftd_data[:, 55:57], ftd_data[:, 81:82], ftd_data[:, 83:88],
                     ftd_data[:, 89:93])
                )
                y = np.hstack((ftd_data[:, 61:62], ftd_data[:, 63:67], ftd_data[:, 68:69]))

                action_group.create_dataset("x", data=x)
                action_group.create_dataset("y", data=y)
        ftd_data_list[altitude_folder] = altitude_data_list
hdf5_file.close()
# %%
with h5py.File(output_file, "r") as hdf5_file:
    for altitude_folder in hdf5_file:
        altitude_group = hdf5_file[altitude_folder]
        print(altitude_group)
        for action_folder in altitude_group:
            action_group = altitude_group[action_folder]
            print(f"Action Folder: {action_group['x']}")
