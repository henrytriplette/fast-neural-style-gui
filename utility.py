from pathlib import Path, PureWindowsPath
from PIL import Image

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def openSingleAndCheck(path, resized_path, resize=False):
    """
    Check if path leads to valid image
    :param img: PIL image object
    :return: PIL image object
    """
    image = Image.open(path)

    if image.mode != 'RGB':
        image = image.convert('RGB')

    if resize != False:
        max_size = resize
        original_size = max(image.size[0], image.size[1])
        if original_size >= max_size:
            if (image.size[0] > image.size[1]):
                resized_width = max_size
                resized_height = int(round((max_size/float(image.size[0]))*image.size[1]))
            else:
                resized_height = max_size
                resized_width = int(round((max_size/float(image.size[1]))*image.size[0]))

            image = image.resize((resized_width, resized_height), Image.ANTIALIAS)

    image.save(resized_path, 'JPEG')

    output = Path(resized_path).absolute()

    return PureWindowsPath(output)
