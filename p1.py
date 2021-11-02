from sklearn import metrics
import matplotlib.pyplot as plt

# y_true = [[1, 0], [1, 0], [0, 1], [0, 1], [1, 0], [1, 0], [0, 1], [0, 1], [1, 0], [0, 1]]
# y_pred = [[0.61, 0.39], [0.03, 0.97], [0.68, 0.32], [0.31, 0.69], [0.45, 0.55],
#           [0.09, 0.91], [0.38, 0.62], [0.05, 0.95], [0.01, 0.99], [0.04, 0.96]]

# y_true = [1, 0, 1, 0, 0, 0, 0, 0, 0, 0]
y_true = [1, 1, 0, 0, 1, 1, 0, 0, 1, 0]
y_pred = [0.61, 0.03, 0.32, 0.69, 0.45, 0.09, 0.62, 0.95, 0.01, 0.96]
f_pred, t_pred, _ = metrics.roc_curve(y_true, y_pred)
# t_pred = [0.61]
# print(f_pred)
# print(t_pred)
plt.title("ROC Curve")
plt.xlabel("False Prediction Rate")
plt.ylabel("Ture Prediction Rate")
plt.plot(f_pred, t_pred)
plt.show()
