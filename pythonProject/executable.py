import pandas
import pytesseract

filepath = 'Capture.png'
from PIL import Image
img = Image.open(filepath)
d = pytesseract.image_to_data(img)





