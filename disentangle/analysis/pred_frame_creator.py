"""
Here, we filter and club together the predicted patches to form the predicted frame.
"""
import numpy as np
import os

class PredFrameCreator:
    def __init__(self, grid_index_manager, frame_t, dump_dir=None) -> None:
        self._grid_index_manager = grid_index_manager
        _,H,W,C = self._grid_index_manager.get_data_shape()
        self.frame = np.zeros((C,H,W), dtype=np.int32)
        self.target_frame = np.zeros((C,H,W), dtype=np.int32)
        self._frame_t = frame_t
        self._dump_dir = dump_dir
        print(f'{self.__class__.__name__} frame_t:{self._frame_t}')
    
    def _update(self, predictions, indices, output_frame):
        for i, index in enumerate(indices):
            h,w,t = self._grid_index_manager.hwt_from_idx(index)
            if t != self._frame_t:
                continue
            sz = predictions[i].shape[-1]
            output_frame[:,h:h+sz,w:w+sz] = predictions[i]
    
    def update(self, predictions, indices):
        self._update(predictions, indices, self.frame)
    
    def update_target(self, target, indices):
        self._update(target, indices, self.target_frame)

    def reset(self):
        self.frame = np.zeros_like(self.frame)

    def dump_target(self):
        assert self._dump_dir is not None
        data_dict = {f'ch_{ch_idx}': self.target_frame[ch_idx] for ch_idx in range(self.target_frame.shape[0])}
        np.savez_compressed(os.path.join(self._dump_dir, f"tar_frame_t_{self._frame_t}.npz"), **data_dict)

        # for ch_idx in range(self.target_frame.shape[0]):
        #     np.save(os.path.join(self._dump_dir, f"target_frame_{self._frame_t}_ch_{ch_idx}.npy"), self.target_frame[ch_idx])

    def dump(self,epoch):
        assert self._dump_dir is not None
        data_dict = {f'ch_{ch_idx}': self.frame[ch_idx] for ch_idx in range(self.frame.shape[0])}
        np.savez_compressed(os.path.join(self._dump_dir, f"pred_frame_{epoch}_t_{self._frame_t}.npz"), **data_dict)
        