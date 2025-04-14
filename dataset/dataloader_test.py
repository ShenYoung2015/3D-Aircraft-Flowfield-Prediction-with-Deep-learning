import numpy as np
import pandas as pd
from pyinstrument import Profiler
profiler = Profiler()
profiler.start()
for i in range(5):
    shape_name = 'shape_001'
    surf_file = f'D:\database\WR_fl\\face_pcd\{shape_name}.txt'
    volume_file = f'D:\database\WR_fl\\volume_pcd\{shape_name}.txt'
    data_surf = pd.read_csv(surf_file, header=None).to_numpy().astype(np.float32)
    data_volume = pd.read_csv(volume_file, header=None).to_numpy().astype(np.float32)
    
profiler.stop()
profiler.print()
