# main file to solve problem 1

from p1.utils.wrappers import YOLOWrapper, RFWrapper, EnsembleWrapper


def main():
    model_RF = RFWrapper(method="nms", use_tta=True)
    model_YOLO = YOLOWrapper(method="nms", use_tta=True)
    ensemble_model = EnsembleWrapper([model_RF, model_YOLO], method="wbf", use_tta=True)
    sample_image_path = "data/VS_KS/K3_CHN_20130308050237_30.jpg"
    results_RF = model_RF.predict(
        [sample_image_path, sample_image_path, sample_image_path]
    )
    results_YOLO = model_YOLO.predict(
        [sample_image_path, sample_image_path, sample_image_path]
    )
    print("Predictions RF:", results_RF)
    print("Predictions YOLO:", results_YOLO)
    results_ensemble = ensemble_model.predict(
        [sample_image_path, sample_image_path, sample_image_path]
    )
    print("Predictions Ensemble:", results_ensemble)


if __name__ == "__main__":
    main()
