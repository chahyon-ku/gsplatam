import os
import imageio.v2 as imageio
from glob import glob
from natsort import natsorted
from tqdm import tqdm


if __name__ == '__main__':
    # main
    datasets =  ['tum', 'replica']
    backends = ['orig', 'gsplat']
    sizes = ['tum', 'base', 'small', 'tiny']
    dataset_paths = {
        'replica': 'experiments-main/Replica/{backend}_{size}-room0-seed0/eval/plots/*.png',
        'tum': 'experiments-main/TUM/{backend}_{size}-freiburg1_desk-seed0/eval/plots/*.png'
    }

    # ablation gs
    datasets = ['replica']
    backends = ['gsplat', 'gsplat_2dgs']
    sizes = ['tiny', 'tiny_aniso', 'small', 'small_aniso']
    dataset_paths = {
        'replica': 'experiments/Replica/{backend}_{size}-room0-seed0/eval/plots/*.png',
    }

    # default
    datasets = ['replica']
    backends = ['gsplat']
    sizes = ['tiny']
    dataset_paths = {
        'replica': 'experiments/Replica/{backend}_{size}-room0-seed0/eval/plots/*.png',
    }

    os.makedirs('videos', exist_ok=True)
    for dataset in datasets:
        table = []
        for size in sizes:
            for backend in backends:
                image_paths = natsorted(glob(dataset_paths[dataset].format(backend=backend, size=size)))
                if len(image_paths):
                    with imageio.get_writer('videos/' + f'{dataset}-{backend}_{size}.mp4', fps=30) as writer:
                        for image_path in tqdm(image_paths, desc=f'Creating {dataset}-{backend}_{size}.mp4'):
                            image = imageio.imread(image_path)
                            writer.append_data(image[:, :image.shape[1] * 2 // 3, :])
