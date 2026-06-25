"""
Prepare the HYPSO-2 dataset by calibrating images, remapping labels,
dropping spectral bands, and optionally applying normalization or PCA.
"""

from hypso import Hypso2
from pathlib import Path
from typing import Tuple
from sklearn.decomposition import PCA
import numpy as np
import json
import sys



def main():
    dataset = Path("dataset") / ("name")
    out = Path("out_folder") / ("out_name")
    calibration = "1a" # or 1b, 1c, 1d
    has_labels = True
    do_z_score_normalization = True
    do_per_pixel_spectral_normalization = False
    do_pca = False
    pca_k = 0
    pca_whiten = False
    train_ids = []
    bands_to_drop = [0,1,2,3,4,5,6,7,118,119]

    out.mkdir(parents=True, exist_ok=True)

    def calibrate(nc_path: Path) -> np.ndarray:
        satobj = Hypso2(path=nc_path)

        if calibration == "1a":
            cube = satobj.l1a_cube.to_numpy().astype(np.float32, copy=False)
        elif calibration == "1b":
            satobj.generate_l1b_cube(smile=False, destripe=False)
            cube = satobj.l1b_cube.to_numpy().astype(np.float32, copy=False)

        elif calibration == "1c":
            satobj.generate_l1c_cube()
            cube = satobj.l1c_cube.to_numpy().astype(np.float32, copy=False)

        elif calibration == "1d":
            satobj.generate_l1b_cube()
            satobj.generate_l1d_cube()
            cube = satobj.l1d_cube.to_numpy().astype(np.float32, copy=False)
        else:
            sys.exit("An error occurred: calibration is not valid")
        
        return cube.astype(np.float32, copy=False)

    def remap_labels(labels: np.ndarray) -> np.ndarray:
        mask = labels.copy()
        labels[mask == 1] = 0
        labels[mask == 2] = 1
        labels[mask == 3] = 2
        labels[mask == 4] = 0
        labels[mask == 5] = 0
        labels[mask == 6] = 0
        labels[mask == 7] = 0
        labels[mask == 8] = 0
        return labels

        
    def drop_bands(cube: np.ndarray, bands_to_drop: list) -> np.ndarray:
        cube = cube.astype(np.float32, copy=False)

        B = cube.shape[2]

        for b in bands_to_drop:
            if b < 0 or b >= B:
                raise ValueError(f"Band index {b} out of range (0, {B-1})")

        mask = np.ones(B, dtype=bool)
        mask[bands_to_drop] = False

        return cube[:, :, mask]
        
    counter_nc = 0
    counter_npy = 0
    counter_json = 0
    cube_shape = np.empty(2)
    label_shape = np.empty(2)

    total = 0
    i = 0
    for scene in dataset.iterdir():
        for item in scene.iterdir():
            if item.suffix == ".nc":
                cube = calibrate(item)
                cube = drop_bands(cube, bands_to_drop)
                nc_path = out / f"data{i}"
                np.save(nc_path, cube)
                cube_shape = cube.shape[:2]
                counter_nc += 1

            if item.suffix == ".json":
                with open(item, "r") as f:
                    meta_data = json.load(f)
                    meta_arr = np.array([
                        meta_data["latitude"],
                        meta_data["longitude"],
                        meta_data["elevation"],
                        meta_data["solar_zenith_angle"],
                        meta_data["solar_azimuth_angle"],
                        meta_data["sat_zenith_angle"],
                        meta_data["sat_azimuth_angle"],
                    ])
                    json_path = out / f"meta{i}"
                    np.save(json_path, meta_arr)
                    counter_json += 1
            if item.suffix == ".npy" and has_labels:
                label_path = out / f"label{i}"
                label_data = np.load(item)
                label_data = remap_labels(label_data)
                np.save(label_path, label_data)
                label_shape = label_data.shape
                counter_npy += 1
        if counter_nc != 1:
            sys.exit("An error occurred: counter_nc != 1")
        if counter_npy != 1 and has_labels:
            sys.exit("An error occurred: counter_npy != 1")
        if counter_json != 1:
            sys.exit("An error occurred: counter_json != 1")
        if not np.array_equal(cube_shape, label_shape) and has_labels:
            sys.exit("An error occurred: cube_shape != label_shape")
        if has_labels:
            print(f"Cube and labels {i} saved with shape: {cube_shape}, unique labels: {np.unique(label_data)}")

        counter_nc = 0
        counter_npy = 0
        counter_json = 0
        i += 1
        total += 1


    def calculate_mu_sd() -> Tuple[np.ndarray, np.ndarray]:
        x = []
        for i in train_ids:
            data_path = out / (f"data{i}.npy")
            temp = np.load(data_path).astype(np.float32)
            H, W, B = temp.shape
            x.append(temp.reshape(-1, B))
        x = np.vstack(x)
        mu = x.mean(0).astype(np.float32)
        sd = (x.std(0) + 1e-6).astype(np.float32)
        return mu, sd
    
    
    if do_z_score_normalization:
        print("Doing normalization ..")
        mu, sd = calculate_mu_sd()
        print(f"mu shape: {mu.shape}, mu lenght: {mu.size}")
        print(f"sd shape: {sd.shape}, sd lenght: {sd.size}")


        with open(out / "mu_sd.txt", "w") as f:
            f.write("mu:\n")
            f.write(np.array2string(mu, separator=", "))
            f.write("\n\nsd:\n")
            f.write(np.array2string(sd, separator=", "))
            f.write("\n")

        for i in range(total):
            data_path = out / f"data{i}.npy"
            temp = np.load(data_path).astype(np.float32)
            temp = (temp - mu) / sd
            np.save(data_path, temp)
            print(f"normalized data{i}.npy")

    def do_per_pixel_spectral_normalization_fn(cube: np.ndarray) -> np.ndarray:
        cube = cube.astype(np.float32, copy=False)

        spec_min = cube.min(axis=2, keepdims=True)
        spec_max = cube.max(axis=2, keepdims=True)
        denom = spec_max - spec_min

        # Avoid divide-by-zero for flat spectra
        denom = np.where(denom < 1e-6, 1.0, denom)

        cube = (cube - spec_min) / denom
        return cube.astype(np.float32, copy=False)

    if do_per_pixel_spectral_normalization:
        print("Doing per-pixel spectral min-max normalization ..")

        for i in range(total):
            data_path = out / f"data{i}.npy"
            temp = np.load(data_path).astype(np.float32)
            temp = do_per_pixel_spectral_normalization_fn(temp)
            np.save(data_path, temp)
            print(f"per-pixel normalized data{i}.npy")

    
    def calculate_pca() -> np.ndarray:
        x = []
        for i in train_ids:
            data_path = out / (f"data{i}.npy")
            temp = np.load(data_path).astype(np.float32)
            H, W, B = temp.shape
            x.append(temp.reshape(-1, B))
        x = np.vstack(x)
        return x
    
    if do_pca:
        print("Doing pca ..")
        X = calculate_pca()
        pca = PCA(n_components=pca_k, whiten=pca_whiten, random_state=0).fit(X)
    
        for i in range(total):
            data_path = out / f"data{i}.npy"
            temp = np.load(data_path).astype(np.float32)
            H, W, B = temp.shape
            temp = pca.transform(temp.reshape(-1, B)).reshape(H, W, pca_k).astype(np.float32)
            np.save(data_path, temp)
            print(f"pca data{i}.npy")



    print("Done")

if __name__ == "__main__":
    main()