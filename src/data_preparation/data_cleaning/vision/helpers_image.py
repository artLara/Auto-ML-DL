import math

import cv2
import numpy as np
from pathlib import Path

FONT_SCALE = 2e-3  # Adjust for larger font size in all images


def read_image(file_path: str,
               color: str = 'rgb') -> np.ndarray:
    """Load image from file
    
    Keyword arguments:
    file_path (str): file path of image.
    color (str): rgb load image with 3 channels and
    gray just one channel. Default rgb.

    Returns np.ndarray
    """
    file_path = Path(file_path)
    assert file_path.exists(), f'file {file_path} does not exists'

    if color == 'rgb': 
        color = cv2.IMREAD_COLOR

    if color == 'gray':
        color = cv2.IMREAD_GRAYSCALE

    return cv2.imread(file_path, color)


def write_image(img: np.ndarray,
                output_dir: str) -> None:
    """Write image in jpg file
    
    Keyword arguments:
    img (np.ndarray).
    output_dir (str): output file.

    Returns None
    """    
    img = np.array(img)
    cv2.imwrite(output_dir, img)


def resize_image(img: np.ndarray,
                 height: int,
                 width: int,
                 aspect_ratio:bool = False) -> np.ndarray:
    """Resize image 
    
    Keyword arguments:
    img (np.ndarray): image to resize.
    height (int): output height.
    width (int): output witdh.

    Returns np.ndarray
    """  
    if aspect_ratio:
        return resize_image_aspect_ratio(img=img,
                                         height=height,
                                         width=width)
    return cv2.resize(img, (height, width))

def padding_zeros(img: np.ndarray,
                  width: int,
                  height: int,
                  color = None):
    """
    Add padding of zeros to image

    Keyword arguments:
    img (np.ndarray): image to resize.
    height (int): output height.
    width (int): output witdh.
    color (np.ndarray): rgb array of color. Default None: zeros (black color)

    Returns np.ndarray
    """
    channels = 1 if len(img.shape) == 2 else img.shape[2]
    if color == None:
        color = np.zeros([channels], dtype=np.uint8)

    (h, w) = img.shape[:2]
    dim = (height,width,channels)
    if channels == 1:
        dim = (height,width)

    image_pad = np.full(dim, color, dtype=np.uint8)
    offsetHeight1 = (height-h)//2
    offsetHeight2 = (height-h)//2 + h
    offsetWidth1 = (width-w)//2
    offsetWidth2 = (width-w)//2 + w

    if channels == 1:
        image_pad[offsetHeight1:offsetHeight2, offsetWidth1:offsetWidth2] = img

    else:
        image_pad[offsetHeight1:offsetHeight2, offsetWidth1:offsetWidth2,:] = img

    return image_pad


def resize_image_aspect_ratio(img: np.ndarray,
                              height: int,
                              width: int,
                              inter=cv2.INTER_AREA) -> np.ndarray:
    """Resize image 
    
    Keyword arguments:
    img (np.ndarray): image to resize.
    height (int): output height.
    width (int): output witdh.
    inter (int): cv2 interpolation. Default cv2.INTER_AREA

    Returns np.ndarray
    """  
    (h, w) = img.shape[:2]
    #Case 1: height is longer than width
    if h > w:
        relationAspect = height / float(h)
        dim = (int(w * relationAspect), height)

    else:
        relationAspect = width / float(w)
        dim = (width, int(h * relationAspect))

    resized = cv2.resize(img, dim, interpolation = inter)
    resized = padding_zeros(resized, 
                            width = width,
                            height = height)
    return resized


def flip_image(img: np.ndarray,
               flip_code: int):
    """Flip image
    Keyword arguments:
    img (np.ndarray): image to resize.
    flip_code (int): 0 horizonally, 1 vertically, -1 both.
    
    Return: np.ndarray
    """
    return cv2.flip(img, flip_code)


def normalize(img: np.ndarray):
    return img/255.0

def draw_point(img: np.ndarray, 
               x: float, 
               y: float, 
               color: tuple[int, int, int] = (255,0,0), 
               rescale: bool = True) -> np.ndarray:
    """Dibuja un punto de color en la coordenada (x,y)
    
    Keyword arguments:
    img (np.ndarray) -- Imagen donde se dibuja.
    x (float) -- coordenada X del punto
    y (float) -- coordenada Y del punto
    color (tuple) -- Color en el cual se dibuja la informacion 
    (default (255,0,0)).
    rescale (bool) -- Reescala los valores normalizados de (x,y) respecto al 
    tamanio de la imagen (defaul True).
    """
    if rescale:
        x = x * img.shape[1]
        y = y * img.shape[0]

    x, y = int(x), int(y)
    pointsize = int(max(img.shape) * 0.01)
    return cv2.circle(img, (x,y), radius=pointsize, color=color, thickness=-1)

def draw_text(img: np.ndarray, 
              x: float, 
              y: float, 
              probs: float,
              font_scale: float, 
              color = (255,0,0),
              thickness = 1,
              font = cv2.FONT_HERSHEY_SIMPLEX, 
              rescale=True) -> np.ndarray:
    
    """Dibuja en la coordenada (x,y) la probabilidad (texto)
    
    Keyword arguments:
    img (np.ndarray) -- Imagen donde se dibuja.
    x (float) -- coordenada X del punto
    y (float) -- coordenada Y del punto
    probs (float) -- probabilidad a dibujar
    font_scale (float): tamanio de la fuente
    color (tuple) -- Color en el cual se dibuja la informacion 
    (default (255,0,0)).
    thickness (float): thickness.
    font (): fuente del texto
    rescale (bool) -- Reescala los valores normalizados de (x,y) respecto al 
    tamanio de la imagen (defaul True).
    """
    x, y = int(x), int(y)
    text = '{}'.format(int(probs*100))
    return cv2.putText(img,
                       text, 
                       (x,y), 
                       font,
                       font_scale, 
                       color,
                       thickness,
                       cv2.LINE_AA, 
                       False)

def draw_info(img: np.ndarray, 
              kps: np.ndarray,  
              probs: np.ndarray,  
              type_model: int,
              color = (255,0,0),
              thickness = 1,
              font = cv2.FONT_HERSHEY_SIMPLEX, 
              rescale=True) -> np.ndarray:
    
    """Dibuja la información necesaria segun el tipo de modelo
    
    Keyword arguments:
    img (np.ndarray) -- Imagen donde se dibuja.
    kps_pred (np.ndarray) -- Keypoints encontrados.
    probs (np.ndarray) -- probabilidades de ser esquinas
    type_model (int) -- Tipo del modelo utilizado. 0 para solo regresión,
    1 para regresión y clasificación de 2 heads, 
    2 para regresión y clasificación de 4 heads,
    3 para segmentar y regresar la mascara binaria,
    4 para segmentar y regresar los kps calculados a partir de la mascara.
    5 para operar con la mascara de segmentacion.
    color (tuple) -- Color en el cual se dibuja la informacion 
    (default (255,0,0)).
    """
    if type_model == 3 or\
        type_model == 5:
        return img
    
    height, width, _ = img.shape
    font_scale = min(width, height) * FONT_SCALE
    thickness = math.ceil(min(width, height) * 0.01)
    for i in range(0, len(kps), 2):
        img = draw_point(img, 
                         kps[i], 
                         kps[i+1],
                         color = color,
                         rescale = rescale)
        
        if type_model:
            img = draw_text(img, 
                            kps[i],
                            kps[i+1],
                            probs[i//2],
                            font_scale,
                            thickness = thickness,
                            color = color)
            
    return img