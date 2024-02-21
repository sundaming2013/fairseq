import logging
import os
import mne  
import numpy as np  
import torch
from fairseq.data import FairseqDataset  
from fairseq.data.text_compressor import TextCompressor, TextCompressionLevel

logger = logging.getLogger(__name__)


class EEGDataset(FairseqDataset):  
    def __init__(
            self,
            manifest_path,
            max_sample_size=None, 
            min_sample_size=0, 
            normalize=True, 
            shuffle=True,
            pad=False,
            text_compression_level=TextCompressionLevel.none,
            **mask_compute_kwargs,
        ):  
        super().__init__()  

        self.fnames = []
        self.normalize = normalize  
        self.shuffle = shuffle  
        self.sizes = []
        self.max_sample_size = max_sample_size if max_sample_size is not None else np.inf  
        self.min_sample_size = min_sample_size  
        self.pad = pad

        self.text_compressor = TextCompressor(level=text_compression_level)

        skipped = 0
        self.fnames = []
        sizes = []
        self.skipped_indices = set()

        with open(manifest_path, "r") as f:
            self.root_dir = f.readline().strip()
            for i, line in enumerate(f):
                items = line.strip().split("\t")
                assert len(items) == 2, line
                sz = int(items[1])
                if min_sample_size is not None and sz < min_sample_size:
                    skipped += 1
                    self.skipped_indices.add(i)
                    continue
                self.fnames.append(self.text_compressor.compress(items[0]))
                sizes.append(sz)
        logger.info(f"loaded {len(self.fnames)}, skipped {skipped} samples")

        self.sizes = np.array(sizes, dtype=np.int64)

        try:
            import pyarrow

            self.fnames = pyarrow.array(self.fnames)
        except:
            logger.debug(
                "Could not create a pyarrow array. Please install pyarrow for better performance"
            )
            pass

    def __len__(self):  
        return len(self.sizes)  

    def __getitem__(self, index):
        fn = self.fnames[index]
        fn = fn if isinstance(self.fnames, list) else fn.as_py()
        fn = self.text_compressor.decompress(fn)
        file_path = os.path.join(self.root_dir, fn)
        # Load raw data    
        raw = mne.io.read_raw(file_path, preload=True)    

        # Define the desired channel order  
        desired_channels = ['Fp1', 'Fpz', 'Fp2', 'F7', 'F3', 'Fz', 'F4',  
                            'F8', 'C3', 'Cz', 'C4', 'T3', 'T5', 'T7',  
                            'T4', 'T6', 'T8', 'P3', 'Pz', 'P4',  
                            'O1', 'Oz', 'O2'] 

        # Create a new info structure with the desired channels  
        new_info = mne.create_info(desired_channels, raw.info['sfreq'], 'eeg', None)  
        
        # Initialize a new RawArray with zeros  
        new_data = np.zeros((len(desired_channels), raw.n_times))  
        
        # Map existing channels to the new data array  
        channel_indices = {ch: i for i, ch in enumerate(raw.info['ch_names'])}  
        for i, ch_name in enumerate(desired_channels):  
            if ch_name in channel_indices:  
                new_data[i, :] = raw._data[channel_indices[ch_name], :]  
        
        # Create the new RawArray with the desired channel order and zero padding  
        new_raw = mne.io.RawArray(new_data, new_info)  
        
        # Use 'new_raw' for further processing  
        raw = new_raw  

        # Get data from Raw object
        data = raw.get_data()

        # Normalize data after padding
        valid_data = data != 0
        mean = np.mean(data[valid_data], axis=0, keepdims=True)
        std = np.std(data[valid_data], axis=0, keepdims=True)
        data = (data - mean) / std
            
        # 修改部分：将标准后的数据返回raw
        raw._data = data

        return {"id": index, "source": torch.from_numpy(data).float()}

    @staticmethod
    def _bucket_tensor(tensor, num_pad, value):
        return F.pad(tensor, (0, num_pad), value=value)
    
    def crop_to_max_size(self, t, target_size, dim=0):
        size = t.size(dim)
        diff = size - target_size
        if diff <= 0:
            return t

        start = np.random.randint(0, diff + 1)
        end = size - diff + start

        # 创建一个多维的切片对象，用于裁剪指定维度上的数据  
        slices = [slice(None)] * t.ndim  
        slices[dim] = slice(start, end)

        return t[tuple(slices)]

    def collater(self, samples):  
        samples = [s for s in samples if s["source"] is not None]  
        if len(samples) == 0:  
            return {}  
    
        sources = [s["source"] for s in samples]  
        # Change here: Get the length of the time dimension (second dimension)  
        sizes = [s.size(1) for s in sources]  
    
        if self.pad:  
            target_size = min(max(sizes), self.max_sample_size)  
        else:  
            target_size = min(min(sizes), self.max_sample_size)  
    
        collated_sources = sources[0].new_zeros((len(sources), sources[0].shape[0], target_size))  
        # Change here: Padding mask should only be for the time dimension  
        padding_mask = (torch.BoolTensor(len(sources), target_size).fill_(False) if self.pad else None)  
        for i, (source, size) in enumerate(zip(sources, sizes)):  
            diff = size - target_size  
            if diff == 0:  
                collated_sources[i] = source  
            elif diff < 0:  
                assert self.pad  
                # Change here: Padding should be applied to the second dimension (time)  
                padding = source.new_zeros((source.shape[0], -diff))  
                collated_sources[i] = torch.cat([source, padding], dim=1)  
                if padding_mask is not None:  
                    padding_mask[i, size:] = True  
            else:  
                collated_sources[i] = self.crop_to_max_size(source, target_size, dim=1)  
    
        input = {"source": collated_sources}  
        out = {"id": torch.LongTensor([s["id"] for s in samples])}  
        if self.pad:  
            input["padding_mask"] = padding_mask  
    
        # The following bucketing code seems to be okay, but ensure that the bucketing is applied correctly to the time dimension.  
        if hasattr(self, "num_buckets") and self.num_buckets > 0:  
            assert self.pad, "Cannot bucket without padding first."  
            bucket = max(self._bucketed_sizes[s["id"]] for s in samples)  
            num_pad = bucket - collated_sources.size(-1)  
            if num_pad:  
                input["source"] = self._bucket_tensor(collated_sources, num_pad, 0)  
                input["padding_mask"] = self._bucket_tensor(padding_mask, num_pad, True)  
    
        out["net_input"] = input  
        return out  


    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        if self.pad:
            return self.sizes[index]
        return min(self.sizes[index], self.max_sample_size)

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""

        if self.shuffle:
            order = [np.random.permutation(len(self))]
            order.append(
                np.minimum(
                    np.array(self.sizes),
                    self.max_sample_size,
                )
            )
            return np.lexsort(order)[::-1]
        else:
            return np.arange(len(self))