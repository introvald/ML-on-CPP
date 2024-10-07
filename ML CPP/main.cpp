#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

// Function to perform linear regression
VectorXd linearRegression(const MatrixXd& X, const VectorXd& y) {
    // Calculate the coefficients (theta) using the normal equation
    VectorXd theta = (X.transpose() * X).inverse() * X.transpose() * y;
    return theta;
}

int main() {
    // Sample data: X (features) and y (target)
    MatrixXd X(4, 2);
    X << 1, 1,
         1, 2,
         2, 2,
         2, 3;
    VectorXd y(4);
    y << 6, 8, 9, 11;

    // Add a column of ones to X for the intercept term
    MatrixXd X_b = MatrixXd::Ones(X.rows(), X.cols() + 1);
    X_b.block(0, 1, X.rows(), X.cols()) = X;

    // Perform linear regression
    VectorXd theta = linearRegression(X_b, y);

    // Output the result
    cout << "Coefficients: " << theta.transpose() << endl;

    return 0;
}
