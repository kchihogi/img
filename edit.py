import os
import cv2
import re
import math
import numpy as np
from skimage import io
from PIL import Image, ImageDraw
from PIL.Image import Resampling

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

def draw_border(image, top, left, height, width, color=(0, 0, 0, 0), thickness=1):
    """
    指定された画像にボーダーを描画します。

    :param image: Imageオブジェクト
    :param top: ボーダーの開始位置（上からのピクセル数）
    :param left: ボーダーの開始位置（左からのピクセル数）
    :param height: ボーダーの高さ
    :param width: ボーダーの幅
    :param color: ボーダーの色（RGBA形式のタプル）
    :param thickness: ボーダーの太さ
    :return: 画像にボーダーが描画されたImageオブジェクト
    """

    image = Image.fromarray(image)

    draw = ImageDraw.Draw(image)
    
    # 上辺
    draw.rectangle([left, top, left + width, top + thickness], fill=color)
    # 下辺
    draw.rectangle([left, top + height - thickness, left + width, top + height], fill=color)
    # 左辺
    draw.rectangle([left, top, left + thickness, top + height], fill=color)
    # 右辺
    draw.rectangle([left + width - thickness, top, left + width, top + height], fill=color)
    return np.array(image)

def brightness_contrast(image, alpha, beta):
    # 明るさとコントラストを調整
    new_image = np.clip(alpha * image + beta, 0, 255).astype(np.uint8)
    return new_image

def negative_image(image):
    # ネガティブ画像を作成
    new_image = 255 - image
    return new_image

def draw_line(image, start, end, color=(0, 0, 0, 0), thickness=1):
    # 直線を描画
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    draw.line(start + end, fill=color, width=thickness)
    return np.array(image)

def rotate_image(image, angle, expand=True):
    """
    画像を回転させ、キャンバスをリサイズして回転後の画像全体を収める。

    :param image: 回転するPIL.Imageオブジェクト
    :param angle: 回転角度（度数法で指定）
    :param expand: 回転後の画像が全て収まるようにキャンバスを拡張するかどうか（デフォルトはTrue）
    :return: 回転後の画像
    """

    image = Image.fromarray(image)

    if expand:
        # 元の画像サイズ
        w, h = image.size
        
        # 角度をラジアンに変換
        angle_rad = math.radians(angle)
        
        # 回転後の画像サイズを計算
        new_w = abs(w * math.cos(angle_rad)) + abs(h * math.sin(angle_rad))
        new_h = abs(h * math.cos(angle_rad)) + abs(w * math.sin(angle_rad))
        
        # 新しいサイズのキャンバスを作成
        new_image = Image.new("RGB", (int(math.ceil(new_w)), int(math.ceil(new_h))), (255, 255, 255))
        
        # 元の画像をキャンバスの中心に貼り付ける
        new_image.paste(image, ((new_image.size[0] - w) // 2, (new_image.size[1] - h) // 2))
        
        # 新しいキャンバス上で回転
        return np.array(new_image.rotate(angle, resample=Resampling.BICUBIC, expand=False))
    else:
        # キャンバスのリサイズなしで回転する
        return np.array(image.rotate(angle, resample=Resampling.BICUBIC, expand=True))

def shear_image_with_angle(image, angle_degrees, direction='horizontal', shear_direction='right'):
    """
    画像にシアー変換を指定された角度で適用します。

    :param image: PIL.Imageオブジェクト
    :param angle_degrees: シアー角度（度数法で指定）
    :param direction: シアーの方向 ('horizontal' または 'vertical')
    :return: シアー変換されたPIL.Imageオブジェクト
    """

    image = Image.fromarray(image)

    # シアー角度のタンジェントを計算
    shear_factor = math.tan(math.radians(angle_degrees))

    if shear_direction == 'left':
        shear_factor = -shear_factor
    
    width, height = image.size
    
    if direction == 'horizontal':
        # 水平方向のシアー
        xshift = abs(shear_factor) * height
        new_width = width + int(math.ceil(xshift))
        
        # アフィン変換行列
        matrix = (1, shear_factor, -xshift if shear_factor > 0 else 0, 0, 1, 0)
        # 変換
        image = image.transform((new_width, height), Image.Transform.AFFINE, matrix, resample=Resampling.BICUBIC)
    
    elif direction == 'vertical':
        # 垂直方向のシアー
        yshift = abs(shear_factor) * width
        new_height = height + int(math.ceil(yshift))
        
        # アフィン変換行列
        matrix = (1, 0, 0, shear_factor, 1, -yshift if shear_factor > 0 else 0)
        # 変換
        image = image.transform((width, new_height), Image.Transform.AFFINE, matrix, resample=Resampling.BICUBIC)
    
    else:
        raise ValueError("Invalid direction: choose 'horizontal' or 'vertical'")
    
    return np.array(image)

def main():
    dpi = 300  # DPIを指定
    ext = '.png'  # 対象の拡張子を指定

    # ファイル一覧を取得
    dir_path = 'edit_input'
    files = file_list(dir_path, ext)

    output_dir = 'edit_output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for f in files:
        print('Processing:', f)
        file_path = os.path.join(dir_path, f)
        output_path = os.path.join(output_dir, f)
        image = read_image(file_path)
        # 枠線を描画
        image_border = draw_border(image, 25, 30, 50, 860, color=(0, 0, 0), thickness=2)
        save_image(output_path, image_border)
        new_name = os.path.splitext(output_path)[0] + f'_border{ext}'
        rename_file(output_path, new_name)
        # 太い枠線を描画
        image_border_thick = draw_border(image, 20, 30, 60, 860, color=(0, 0, 0), thickness=6)
        save_image(output_path, image_border_thick)
        new_name = os.path.splitext(output_path)[0] + f'_border_thick{ext}'
        rename_file(output_path, new_name)
        # 明るいコントラストへ変換
        image_bc = brightness_contrast(image, 1.5, 50)
        save_image(output_path, image_bc)
        new_name = os.path.splitext(output_path)[0] + f'_bright{ext}'
        rename_file(output_path, new_name)
        # もっと明るいコントラストへ変換
        image_bc = brightness_contrast(image, 2.0, 100)
        save_image(output_path, image_bc)
        new_name = os.path.splitext(output_path)[0] + f'_more_bright{ext}'
        rename_file(output_path, new_name)
        # 暗いコントラストへ変換
        image_bc = brightness_contrast(image, 0.5, -50)
        save_image(output_path, image_bc)
        new_name = os.path.splitext(output_path)[0] + f'_dark{ext}'
        rename_file(output_path, new_name)
        # もっと暗いコントラストへ変換
        image_bc = brightness_contrast(image, 0.5, -100)
        save_image(output_path, image_bc)
        new_name = os.path.splitext(output_path)[0] + f'_more_dark{ext}'
        rename_file(output_path, new_name)
        # ネガティブ画像を作成
        image_negative = negative_image(image)
        save_image(output_path, image_negative)
        new_name = os.path.splitext(output_path)[0] + f'_negative{ext}'
        rename_file(output_path, new_name)
        # 高解像度に変換
        image_resize = resize_image(image, 1200/dpi, method='nearest')
        save_image(output_path, image_resize)
        dpi_change(output_path, 1200)
        new_name = re.sub(r'_\d{3}dpi', '_1200dpi', output_path)
        rename_file(output_path, new_name)
        # 低解像度に変換
        image_resize = resize_image(image, 72/dpi, method='bicubic')
        save_image(output_path, image_resize)
        dpi_change(output_path, 72)
        new_name = re.sub(r'_\d{3}dpi', '_72dpi', output_path)
        rename_file(output_path, new_name)
        # ラインを描画
        image_line = draw_line(image, (50, 50), (870, 50), color=(0, 0, 0), thickness=2)
        save_image(output_path, image_line)
        new_name = os.path.splitext(output_path)[0] + f'_line{ext}'
        rename_file(output_path, new_name)
        # スラッシュを描画
        image_line = draw_line(image, (0, 0), (925, 100), color=(0, 0, 0), thickness=4)
        save_image(output_path, image_line)
        new_name = os.path.splitext(output_path)[0] + f'_slash{ext}'
        rename_file(output_path, new_name)
        # 90度系回転
        for angle in [90, 180, 270]:
            image_rotate = rotate_image(image, angle, expand=False)
            save_image(output_path, image_rotate)
            new_name = os.path.splitext(output_path)[0] + f'_rotate{angle}{ext}'
            rename_file(output_path, new_name)

        # 1-10度の回転
        for angle in range(1, 11):
            image_rotate = rotate_image(image, angle)
            save_image(output_path, image_rotate)
            new_name = os.path.splitext(output_path)[0] + f'_rotate{angle}{ext}'
            rename_file(output_path, new_name)

        # 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0度のシアー変換 (right)
        for angle in [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]:
            image_shear = shear_image_with_angle(image, angle, direction='horizontal', shear_direction='right')
            save_image(output_path, image_shear)
            new_name = os.path.splitext(output_path)[0] + f'_shear{angle}_horizontal_right{ext}'
            rename_file(output_path, new_name)

        # 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0度のシアー変換 (left)
        for angle in [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]:
            image_shear = shear_image_with_angle(image, angle, direction='horizontal', shear_direction='left')
            save_image(output_path, image_shear)
            new_name = os.path.splitext(output_path)[0] + f'_shear{angle}_horizontal_left{ext}'
            rename_file(output_path, new_name)

        # rotate and shear
        angles = range(1, 11)
        shear_angles = [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]

        for angle in angles:
            for shear_angle in shear_angles:
                image_rotate = rotate_image(image, angle)
                image_shear = shear_image_with_angle(image_rotate, shear_angle, direction='horizontal', shear_direction='right')
                save_image(output_path, image_shear)
                new_name = os.path.splitext(output_path)[0] + f'_rotate{angle}_shear{shear_angle}_horizontal_right{ext}'
                rename_file(output_path, new_name)
                image_shear = shear_image_with_angle(image_rotate, shear_angle, direction='horizontal', shear_direction='left')
                save_image(output_path, image_shear)
                new_name = os.path.splitext(output_path)[0] + f'_rotate{angle}_shear{shear_angle}_horizontal_left{ext}'
                rename_file(output_path, new_name)

        print('Done')

if __name__ == '__main__':
    main()