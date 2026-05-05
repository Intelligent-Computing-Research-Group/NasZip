import numpy as np

class Dfloat:
    def __init__(self):
        None

    # Dfloat
    def mask_fp32_mantissa(self, arr, n_bits):
        # Always operate on a private copy so masking never mutates
        # views into query/index storage in place.
        arr = np.array(arr, dtype=np.float32, copy=True)
        arr = np.ascontiguousarray(np.atleast_1d(arr))
        
        # Convert float32 to uint32 for bitwise operations.
        uint_view = arr.view(np.uint32)
        
        # Create a mask: keep the top (23 - n_bits) mantissa bits,
        # and set the lower n_bits to 0.
        # float32 has a 23-bit mantissa (bit positions 0-22).
        mantissa_mask = 0xFFFFFFFF ^ ((1 << n_bits) - 1)  # Lower n_bits are zeroed.
        
        # Apply the mask.
        uint_view &= mantissa_mask
        
        # Convert back to float32.
        return uint_view.view(np.float32).reshape(arr.shape)
