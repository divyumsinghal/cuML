#include <LinearRegression.cuh>
#include <iostream>
#include <vector>

int main() {
  int m = 5, n = 3;  // 5 samples, 3 features

  /*
   *   X = |  1.0 |  5.0 |  9.0 |
   *       |  2.0 |  6.0 | 10.0 |
   *       |  3.0 |  7.0 | 11.0 |
   *       |  4.0 |  8.0 | 12.0 |
   *       |  5.0 |  9.0 | 13.0 |
   *
   *   Y = | 18.0 | 21.0 | 24.0 | 27.0 | 30.0 |
   */

  const std::vector<float> X = {1.0f, 2.0f,  3.0f,  4.0f,  5.0f,
                                5.0f, 6.0f,  7.0f,  8.0f,  9.0f,
                                9.0f, 10.0f, 11.0f, 12.0f, 13.0f};
  const std::vector<float> Y = {18.0f, 21.0f, 24.0f, 27.0f, 30.0f};

  /*
   *   W = coefficients to be learned
   */
  std::vector<float> W(n, 0.0f);  // n coefficients, not m*n

  LinearRegression<float> lr;
  lr.fit(X, Y, W, m, n);  // 5 samples, 3 features

  std::cout << "Learned coefficients: ";
  for (float coeff : W) {  // Use float, not int
    std::cout << coeff << " ";
  }

  std::cout << std::endl;

  return 0;
}
