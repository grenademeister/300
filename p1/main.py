# main file to solve problem 1

from p1.utils.wrappers import YOLOWrapper, RFWrapper, EnsembleWrapper


def main():
    model_RF = RFWrapper(method="nms", use_tta=True)
    model_YOLO = YOLOWrapper(method="nms", use_tta=True)
    ensemble_model = EnsembleWrapper([model_RF, model_YOLO], method="wbf", use_tta=True)
    sample_image_path_1 = "data/VS_KS/K3_CHN_20130308050237_30.jpg"
    sample_image_path_2 = "data/VS_KS/K3_CHN_20130308050353_17.jpg"
    sample_image_path_3 = "data/VS_KS/K3A_CHN_20240329055658_26.jpg"
    results_RF = model_RF.predict(
        [sample_image_path_1, sample_image_path_2, sample_image_path_3]
    )
    results_YOLO = model_YOLO.predict(
        [sample_image_path_1, sample_image_path_2, sample_image_path_3]
    )
    print("Predictions RF:", results_RF)
    print("Predictions YOLO:", results_YOLO)
    results_ensemble = ensemble_model.predict(
        [sample_image_path_1, sample_image_path_2, sample_image_path_3]
    )
    print("Predictions Ensemble:", results_ensemble)


if __name__ == "__main__":
    main()
