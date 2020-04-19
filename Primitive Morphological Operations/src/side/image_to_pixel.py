from PIL import Image
import numpy as np 

def extractPixels(): 
    im = Image.open("../input/f3.png", 'r')
    pix_val = [im.getdata()]
    pix_val_flat = [x for sets in pix_val for x in sets]
    with open('pixeled.txt', 'w') as file:
        file.write(str(pix_val_flat))