import os
import cv2
import numpy as np
from skimage import io
from PIL import Image

def file_list(dir_path, ext=None):
    # ディレクトリ内のファイル一覧を取得
    files = os.listdir(dir_path)
    files = [f for f in files if os.path.isfile(os.path.join(dir_path, f))]
    if ext is not None:
        files = [f for f in files if os.path.splitext(f)[1] == ext]
    return files

def read_image(file_path):
    # 画像を読み込む
    image = io.imread(file_path)
    return image

def rgba2rgb(image_rgba):
    # RGBA画像をRGBに変換
    if image_rgba.shape[-1] == 4:
        image_rgb = image_rgba[:, :, :3]
    else:
        image_rgb = image_rgba
    return image_rgb

def save_image(file_path, image):
    # 画像を保存
    io.imsave(file_path, image)

def dpi_change(file_path, dpi):
    # DPIを変更して保存
    image = Image.open(file_path)
    image.save(file_path, dpi=(dpi, dpi))

def image_info(image):
    return {
        'shape': image.shape,
        'dtype': image.dtype,
        'min': image.min(),
        'max': image.max()
    }

def rename_file(file_path, new_name):
    # ファイル名を変更
    os.rename(file_path, new_name)

def bicubic_interpolation(image, scale):
    height, width = image.shape[:2]
    new_dimensions = (int(width * scale), int(height * scale))
    return cv2.resize(image, new_dimensions, interpolation=cv2.INTER_CUBIC)

def bilinear_interpolation(image, scale):
    height, width = image.shape[:2]
    new_dimensions = (int(width * scale), int(height * scale))
    return cv2.resize(image, new_dimensions, interpolation=cv2.INTER_LINEAR)

def nearest_interpolation(image, scale):
    height, width = image.shape[:2]
    new_dimensions = (int(width * scale), int(height * scale))
    return cv2.resize(image, new_dimensions, interpolation=cv2.INTER_NEAREST)

def resize_image(image, scale, method='bicubic'):
    # 画像を拡大する
    if method == 'bicubic':
        return bicubic_interpolation(image, scale)
    elif method == 'bilinear':
        return bilinear_interpolation(image, scale)
    elif method == 'nearest':
        return nearest_interpolation(image, scale)
    else:
        raise ValueError('Invalid method')
    
def resize_canvas(image, tmargin, lmargin, bmargin, rmargin, color=(0, 0, 0, 0)):
    # キャンバスをリサイズ
    height, width, channels = image.shape
    new_height = height + tmargin + bmargin
    new_width = width + lmargin + rmargin
    new_image = np.zeros((new_height, new_width, channels), dtype=image.dtype)
    new_image[:, :] = color
    new_image[tmargin:tmargin + height, lmargin:lmargin + width] = image
    return new_image

def main():
    dpi = 96  # DPIを指定
    dpi2 = 300  # DPIを指定
    ext = '.bmp'  # 対象の拡張子を指定
    output_ext = '.png'  # 出力の拡張子を指定
    scale = dpi2 / dpi  # 拡大率を計算
    xmargin = 10  # キャンバスの余白を指定
    ymargin = 10  # キャンバスの余白を指定

    # ファイル一覧を取得
    dir_path = 'input'
    files = file_list(dir_path, ext)

    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for f in files:
        print('Processing:', f)
        file_path = os.path.join(dir_path, f)
        output_path = os.path.join(output_dir, os.path.splitext(f)[0] + output_ext)
        image = read_image(file_path)
        image = resize_canvas(image, ymargin, xmargin, ymargin, xmargin, color=(255, 255, 255, 0))
        image_rgb = rgba2rgb(image)
        save_image(output_path, image_rgb)
        dpi_change(output_path, dpi)
        info = image_info(image_rgb)
        new_name = os.path.splitext(output_path)[0] + f'_{info["dtype"]}_{dpi}dpi{output_ext}'
        rename_file(output_path, new_name)

        # resized_output_path = os.path.join(output_dir, os.path.splitext(f)[0] + f'_bicubic_{scale}x{output_ext}')
        # resized_image = resize_image(image_rgb, scale, method='bicubic')
        # save_image(resized_output_path, resized_image)
        # info = image_info(resized_image)
        # dpi_change(resized_output_path, dpi2)
        # new_name = os.path.splitext(resized_output_path)[0] + f'_{info["dtype"]}_{dpi2}dpi{output_ext}'
        # rename_file(resized_output_path, new_name)

        # resized_output_path = os.path.join(output_dir, os.path.splitext(f)[0] + f'_bilinear_{scale}x{output_ext}')
        # resized_image = resize_image(image_rgb, scale, method='bilinear')
        # save_image(resized_output_path, resized_image)
        # info = image_info(resized_image)
        # dpi_change(resized_output_path, dpi2)
        # new_name = os.path.splitext(resized_output_path)[0] + f'_{info["dtype"]}_{dpi2}dpi{output_ext}'
        # rename_file(resized_output_path, new_name)

        # resized_output_path = os.path.join(output_dir, os.path.splitext(f)[0] + f'_nearest_{scale}x{output_ext}')
        resized_output_path = os.path.join(output_dir, os.path.splitext(f)[0] + f'{output_ext}')
        resized_image = resize_image(image_rgb, scale, method='nearest')
        save_image(resized_output_path, resized_image)
        info = image_info(resized_image)
        dpi_change(resized_output_path, dpi2)
        new_name = os.path.splitext(resized_output_path)[0] + f'_{info["dtype"]}_{dpi2}dpi{output_ext}'
        rename_file(resized_output_path, new_name)

if __name__ == '__main__':
    main()