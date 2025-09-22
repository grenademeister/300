# main file to solve problem 1

from p1.utils.wrappers import YOLOWrapper


def main():
    model = YOLOWrapper(method="nms", use_tta=True)
    sample_image_path = "data/VS_KS/K3_CHN_20130308050237_30.jpg"
    model.predict([sample_image_path, sample_image_path, sample_image_path])


if __name__ == "__main__":
    main()
