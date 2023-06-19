from PIL import Image, ImageEnhance, ImageFilter
import os

# Test script for contour tracing
# batch transform
def imageTrace(path,output_path):
    for f in os.listdir(path):
        img = Image.open(f"{path}/{f}")

        edit_gray = img.filter(ImageFilter.SHARPEN).convert('L')
        edit_smooth = edit_gray.filter(ImageFilter.SMOOTH)
        edit_enhance = edit_smooth.filter(ImageFilter.EDGE_ENHANCE)
        edit_edges = edit_enhance.filter(ImageFilter.FIND_EDGES)
        contrast = ImageEnhance.Contrast(edit_edges)
        edit_contrast = contrast.enhance(1.5)

        clean_name = os.path.splitext(f)[0]

        edit_contrast.save(f'.{output_path}/{clean_name}_edited.jpg')
        return "Output written to filepath"


# batch transform
def imageContour(path, output_path):
    for f in os.listdir(path):
        img = Image.open(f"{path}/{f}")

        edit_gray = img.filter(ImageFilter.SHARPEN).convert('L')
        edit_smooth = edit_gray.filter(ImageFilter.SMOOTH)
        edit_enhance = edit_smooth.filter(ImageFilter.EDGE_ENHANCE_MORE)
        edit_edges = edit_enhance.filter(ImageFilter.FIND_EDGES)
        contrast = ImageEnhance.Contrast(edit_edges)
        edit_contrast = contrast.enhance(1.5)

        clean_name = os.path.splitext(f)[0]

        edit_contrast.save(f'.{output_path}/{clean_name}_edited.jpg')
        return "Output written to filepath"

path = './Images'
output_path = '/editedImages'

print('Do you want to trace or contour?')
user_in = input().lower()

if user_in == 'trace':
    imageTrace(path, output_path)
else:
    imageContour(path, output_path)
