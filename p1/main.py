# main file to solve problem 1

from p1.utils.wrappers import YOLOWrapper, RFWrapper


def main():
    model = RFWrapper(method="nms", use_tta=True)
    sample_image_path = "data/VS_KS/K3_CHN_20130308050237_30.jpg"


if __name__ == "__main__":
    main()
