# Advanced Model Analysis Notes
This folder contains post-training analysis figures for the multi-task multi-head residual MLP weather model.

## Important interpretation
The model is a regression model. Therefore, confusion matrices, ROC curves and Precision-Recall curves are not native evaluation metrics for the four atmospheric outputs. To satisfy the requirement honestly, the script creates derived classification views: binned regression confusion matrices and a high-wind event classifier based on predicted wind speed.

## Training behavior
Best validation epoch: 292
Possible overfitting start: 29
Comment: A possible overfitting region was detected by trend heuristic.

## Held-out regression metrics
- temperature_k: MAE=1.1233, RMSE=1.5092, P95 abs error=3.0683, R²=0.9976
- pressure_pa: MAE=498.1449, RMSE=809.4811, P95 abs error=1797.1967, R²=0.9994
- wind_u: MAE=1.8516, RMSE=2.5020, P95 abs error=5.0069, R²=0.9749
- wind_v: MAE=1.7279, RMSE=2.3957, P95 abs error=4.7671, R²=0.9124

## Recommended figures for the project book
1. `architecture/model_architecture_block_diagram.png`
2. `training_diagnostics/01_loss_with_overfitting_marker.png`
3. `training_diagnostics/03_learning_rate_schedule.png` and `04_learning_rate_vs_loss.png`
4. `prediction_quality/predicted_vs_actual_*.png` and `residual_histogram_*.png`
5. `physical_error_analysis/error_vs_altitude_m_*.png`
6. `feature_importance/permutation_importance_top15_*.png`
7. `weights/weights_distribution_all_kernels.png`
8. `classification_views/high_wind_confusion_matrix.png`, `high_wind_roc_curve.png`, and `high_wind_precision_recall_curve.png`
