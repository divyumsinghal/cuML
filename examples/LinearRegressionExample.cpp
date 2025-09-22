#include <LinearRegression.cuh>
#include <iostream>
#include <vector>

int main() {
  int m = 4, n = 4;

  /*
   *   A = |  1.0 |  5.0 |  9.0 | 13.0 |
   *       |  2.0 |  6.0 | 10.0 | 14.0 |
   *       |  3.0 |  7.0 | 11.0 | 15.0 |
   *       |  4.0 |  8.0 | 12.0 | 16.0 |
   *
   *   B = |  1.0 |  2.0 |  3.0 |  4.0 |
   */

  const std::vector<float> A = {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,
                                7.0f,  8.0f,  9.0f,  10.0f, 11.0f, 12.0f,
                                13.0f, 14.0f, 15.0f, 16.0f};
  const std::vector<float> B = {1.0f, 2.0f, 3.0f, 4.0f};

  /*
   *   C = | 276.0 | 304.0 | 332.0 |
   */
  std::vector<float> C(m * n, 0.0f);

  LinearRegression<float> lr;
  lr.fit(A, B, C, 4, 3);

  for (int i : C) {
    std::cout << i << " ";
  }
  std::cout << std::endl;

  return 0;
}
