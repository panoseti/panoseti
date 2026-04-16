# read the first N frames of a PFF file.
# compute the x and 1-x quantiles of the pixels


from utils import pff


def get_values(file: str, image_size: int, bytes_per_pixel: int, nframes: int = 100) -> list[int]:
    fin = open(file, "rb")
    values: list[int] = []
    for _i in range(nframes):
        x = pff.read_json(fin)
        if x is None:
            break
        img = pff.read_image(fin, image_size, bytes_per_pixel)
        if img is None:
            break
        for j in range(image_size*image_size):
            values.append(img[j])
    fin.close()
    return values

def get_quantiles(file: str, img_size: int, bytes_per_pixel: int, x: float) -> list[int]:
    values = get_values(file, img_size, bytes_per_pixel)
    n = len(values)
    values.sort()
    return [values[int(n*x)], values[int(n*(1-x))] ]
