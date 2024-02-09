from PIL import Image
import cv2

# cv2
def read_image_as_bgr(filename:str):
    return cv2.imread(filename)

def read_image_as_gray(filename:str):
    return cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

def read_image_as_rgb(filename:str):
    img_bgr = cv2.imread(filename)
    return convert_from_bgr_to_rgb(img_bgr)

def get_width_and_heigt(img):
    (height, width, *colors) = img.shape
    if len(colors) == 0 :
        colors.append(1)
    return (height,width,*colors)

def convert_from_bgr_to_rgb(img_bgr):
    return cv2.cvtColor(img_bgr,cv2.COLOR_BGR2RGB)

# PIL


def get_image_as_rgb(filename:str):
    img = Image.open(filename)
    return img.convert('RGB')