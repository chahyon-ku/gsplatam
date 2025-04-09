


import os


if __name__ == '__main__':
    datasets = ['replica', 'tum']
    backends = ['orig', 'gsplat']
    sizes = ['tum', 'base', 'small', 'tiny']
    backend_names = {
        'orig': 'reproduced',
        'gsplat': 'ours',
    }

    for dataset in datasets:
        table = []
        for size in sizes:
            for backend in backends:
                path = f'logs-main/{backend}_{size}-{dataset}.log'
                if not os.path.exists(path):
                    continue

                metrics = {
                    'track_time': 0,
                    'map_time': 0,
                    'total_time': 0,
                    'num_gaussians': 0,
                    'ate_rmse': 0,
                    'psnr': 0,
                    'depth_l1': 0,
                }
                with open(path, 'r') as f:
                    lines = f.readlines()

                for line in lines:
                    if 'Average Tracking/Frame Time' in line:
                        metric = line.split()[-2]
                        metrics['track_time'] = float(metric)
                    elif 'Average Mapping/Frame Time' in line:
                        metric = line.split()[-2]
                        metrics['map_time'] = float(metric)
                    elif '100%|██████████| 592/592' in line or '100%|██████████| 2000/2000' in line:
                        if metrics['track_time'] == 0:
                            metric = line.split()[3]
                            # 1.44s/it -> 0.6944
                            if metric[-5] == 's':
                                metrics['total_time'] = float(metric[:-5])
                            else:
                                metrics['total_time'] = 1 / float(metric[:-5])
                            if 'num_gaussians' in line:
                                metric = line.split()[5][:-1]
                                metrics['num_gaussians'] = int(metric)
                    elif 'Total Number of Gaussians' in line:
                        metric = line.split()[-1]
                        metrics['num_gaussians'] = int(metric)
                    elif 'Final Average ATE RMSE' in line:
                        metric = line.split()[-2]
                        metrics['ate_rmse'] = float(metric)
                    elif 'Average PSNR' in line:
                        metric = line.split()[-1]
                        metrics['psnr'] = float(metric)
                    elif 'Average Depth L1' in line:
                        metric = line.split()[-2]
                        metrics['depth_l1'] = float(metric)

                table.append(f'{size} & {backend_names[backend]} & {metrics["track_time"]:.2f} & {metrics["map_time"]:.2f} & {metrics["total_time"]:.2f} & {metrics["num_gaussians"]:,} & {metrics["ate_rmse"]:.2f} & {metrics["psnr"]:.2f} & {metrics["depth_l1"]:.2f} \\\\')
        print('\n'.join(table))
        print()