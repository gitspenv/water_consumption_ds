import matplotlib.pyplot as plt

def plot_residuals_vs_predicted(y_pred, residuals):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--', linewidth=1)
    plt.title('Residuals vs Predicted')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.show()

def plot_feature_importance(feature_importances, feature_names):
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, feature_importances)
    plt.title('Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.show()

def plot_mean_error_by_feature(test_data, feature, error_column='Prediction_Error'):
    grouped_error = test_data.groupby(feature)[error_column].mean()
    plt.figure(figsize=(10, 6))
    plt.bar(grouped_error.index, grouped_error.values)
    plt.title(f'Mean Prediction Error by {feature.capitalize()}')
    plt.xlabel(feature.capitalize())
    plt.ylabel('Mean Error')
    plt.xticks(rotation=45)
    plt.show()
