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
    image_dirs = [
        # 'experiments/Replica/gsplat-tiny-isotropic-room0-seed0',
        'experiments/TUM/gsplat-small-isotropic-freiburg1_desk-seed0',
    ]
    image_dirs = glob('experiments/*/*')

    os.makedirs('videos', exist_ok=True)
    for image_dir in image_dirs:
        video_path = os.path.join(f'videos/{os.path.basename(image_dir)}.mp4')
        image_paths = natsorted(glob(f'{image_dir}/eval/plots/*.png'))
        with imageio.get_writer(video_path, fps=30) as writer:
            for image_path in tqdm(image_paths, desc=f'Creating {video_path}.mp4'):
                image = imageio.imread(image_path)
                writer.append_data(image[:, :image.shape[1] * 2 // 3, :])
