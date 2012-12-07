
import Image
import numpy as np



def create_composite_grid_image(image_arrays):
    n = int(np.ceil(np.sqrt(len(image_arrays))))
    shapes = np.array([arr.shape for arr in image_arrays])
    sy, sx, _ = 5 + shapes.max(axis=0)
    comp = Image.new('RGB', (n*sx, n*sy), 'white')
    for si,arr in enumerate(image_arrays):
        img = Image.fromarray(arr.astype(np.uint8))
        i,j = divmod(si, n)
        comp.paste(img, (j*sx, i*sy))
    return comp
