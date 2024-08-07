from dataclasses import dataclass
from typing import Iterable

import numpy as np

from disentangle.data_loader.multifile_dset import MultiFileDset


@dataclass
class AlgebTuple:
    """A tuple class that supports addition and subtraction."""
    data: tuple

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

    def __add__(self, other: Iterable): 
        return AlgebTuple(tuple([self[i] + other[i] for i in range(len(self))]))

    def __sub__(self, other: Iterable): 
        return AlgebTuple(tuple([self[i] - other[i] for i in range(len(self))]))

# from disentangle.analysis.stitch_prediction import * 
def stitch_predictions(predictions, dset):
    """
    Args:
        smoothening_pixelcount: number of pixels which can be interpolated
    """
    if isinstance(dset, MultiFileDset):
        cum_count = 0
        output = []
        for dset in dset.dsets:
            cnt = dset.idx_manager.total_grid_count()
            output.append(
                stitch_predictions(predictions[cum_count:cum_count + cnt], dset))
            cum_count += cnt
        return output

    else:
        mng = dset.idx_manager
        
        # if there are more channels, use all of them.
        shape = list(dset.get_data_shape())
        shape[-1] = max(shape[-1], predictions.shape[1])

        output = np.zeros(shape, dtype=predictions.dtype)
        # frame_shape = dset.get_data_shape()[:-1]
        for dset_idx in range(predictions.shape[0]):
            # loc = get_location_from_idx(dset, dset_idx, predictions.shape[-2], predictions.shape[-1])
            # grid start, grid end
            gs = AlgebTuple(mng.get_location_from_dataset_idx(dset_idx))
            ge = gs + mng.grid_shape

            # patch start, patch end
            ps = gs - mng.patch_offset()
            pe = ps + mng.patch_shape
            # print('PS')
            # print(ps)
            # print(pe)

            # valid grid start, valid grid end
            vgs = AlgebTuple([max(0,x) for x in gs])
            vge = AlgebTuple([min(x,y) for x,y in zip(ge, mng.data_shape)])
            # print('VGS')
            # print(gs)
            # print(ge)

            # relative start, relative end. This will be used on pred_tiled
            rs = vgs - ps
            re = rs + ( vge - vgs)
            # print('RS')
            # print(rs)
            # print(re)
            
            # print(output.shape)
            # print(predictions.shape)
            for ch_idx in range(predictions.shape[1]):
                if len(output.shape) == 4:
                    # channel dimension is the last one.
                    output[vgs[0]:vge[0],
                        vgs[1]:vge[1],
                        vgs[2]:vge[2],
                        ch_idx] = predictions[dset_idx][ch_idx,rs[1]:re[1], rs[2]:re[2]]
                elif len(output.shape) == 5:
                    # channel dimension is the last one.
                    output[vgs[0],
                        vgs[1]:vge[1],
                        vgs[2]:vge[2],
                        vgs[3]:vge[3],
                        ch_idx] = predictions[dset_idx][ch_idx, rs[1]:re[1], rs[2]:re[2], rs[3]:re[3]]
                else:
                    raise ValueError(f'Unsupported shape {output.shape}')
                
        return output
