import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# Scatter Plot Function
def scatter_plot(y_train, y_train_preds, y_test, y_test_preds, title='Catboost Model', xlim=(-5, 11), ylim=(-5, 11), save_path='scatter_plot_catboost.png'):
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)
    # sns.set(font_scale=2)

    f, ax = plt.subplots(figsize=(8, 8))
    plt.scatter(y_train, y_train_preds, color='#2d4d85', s=30, label='Train dataset', alpha=1)
    plt.scatter(y_test, y_test_preds, color='#951d6d', s=30, label='Test dataset', alpha=1)
    plt.plot(y_test_preds, y_test_preds, color='#444444', linewidth=2)

    # Customize axis
    plt.gca().spines['bottom'].set_color('black')
    plt.gca().spines['bottom'].set_linewidth(3)
    plt.gca().spines['left'].set_color('black')
    plt.gca().spines['left'].set_linewidth(3)

    plt.title(title)
    plt.xlabel('Actual data')
    plt.ylabel('Predicted data')
    plt.legend(loc='upper left')
    plt.xlim(xlim)
    plt.ylim(ylim)

    # Save the figure with transparency
    plt.savefig(save_path, transparent=True)
    plt.show()


# Feature Importance Plot Function
def feature_importance_plot(model, X_train, cols, title='Feature Importance', save_path='feature_importance.png'):
    feature_importance = model.feature_importances_
    feature_names = X_train.columns
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importance})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    custom_palette = sns.color_palette("Blues_r", n_colors=len(feature_importance_df))
    selected_feature_importance = feature_importance_df[feature_importance_df['Feature'].isin(cols)]

    # Plotting feature importance with custom color gradient
    plt.figure(figsize=(8, 8))
    sns.barplot(x='Importance', y='Feature', data=selected_feature_importance, palette=custom_palette)
    plt.title(title)
    plt.xlabel('Importance')
    plt.ylabel('Features')
    plt.tight_layout()

    # Save the plot
    plt.savefig(save_path, transparent=True)
    plt.show()

# SHAP Summary Plot Function
def shap_summary_plot(model, X_train, cols, save_path='shap_summary_plot.png'):
    X_importance = X_train
    selected_X_importance = X_importance[cols]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_importance)
    selected_shap_values = shap_values[:, X_importance.columns.isin(cols)]

    # Create the SHAP summary plot
    shap.summary_plot(selected_shap_values, selected_X_importance, show=False)
    plt.gcf().set_size_inches(8, 8)
    plt.tight_layout()  # Ensures plots are properly arranged

    plt.savefig(save_path, transparent=True)
    plt.show()