import numpy as np
from PIL import ImageDraw, Image
# Draw sample

def Draw_img_bbox(image,result):

    for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
        box = [round(i, 2) for i in box.tolist()]

    image_with_boxes = image.copy()
    draw = ImageDraw.Draw(image_with_boxes)

    for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        x, y, x2, y2 = tuple(box)
        draw.rectangle((x, y, x2, y2), outline="red", width=1)

        draw.text((x, y), f" [ {score.item():.2f} ]", fill="blue")

    return image_with_boxes

def Draw_histogram(path):
    fig, ax = plt.subplots(1,2,figsize=(20,10))

    try:
        # 1. Pillow로 이미지 열기
        src = Image.open(path)

        # 2. 이미지를 NumPy 배열로 변환
        img_array = np.array(src)

        # 3. NumPy 배열에서 R, G, B 채널 분리 (슬라이싱 이용)
        # img_array는 (높이, 너비, 채널) 형태의 3차원 배열입니다.
        red = img_array[:, :, 0]
        green = img_array[:, :, 1]
        blue = img_array[:, :, 2]

        # 원본 이미지 시각화 (NumPy 배열을 직접 사용)
        ax[0].imshow(img_array)
        ax[0].set_title('Sample Image (RGB)')
        ax[0].axis('off')

        # 각 채널의 히스토그램 시각화
        ax[1].hist(red.flatten(), bins=256, color='red', alpha=0.5, label='Red Channel')
        ax[1].hist(green.flatten(), bins=256, color='green', alpha=0.5, label='Green Channel')
        ax[1].hist(blue.flatten(), bins=256, color='blue', alpha=0.5, label='Blue Channel')
        ax[1].set_title('Pixel Intensity Histogram')
        ax[1].legend() # 각 히스토그램의 라벨을 표시

        plt.show()

    except FileNotFoundError:
        print(f"'{path}' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")