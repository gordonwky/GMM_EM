#include <iostream>
#include <vector>

double calculateRandIndex(const std::vector<int> &clustering1,
                          const std::vector<int> &clustering2) {
  size_t n = clustering1.size();
  if (n != clustering2.size()) {
    std::cerr << "Error: Clustering sizes do not match.\n";
    return -1.0; // Error code
  }

  int a = 0, b = 0;

  // Compare each pair of elements
  for (size_t i = 0; i < n; ++i) {
    for (size_t j = i + 1; j < n; ++j) {
      if (clustering1[i] == clustering1[j] &&
          clustering2[i] == clustering2[j]) {
        // Both clusterings group the pair in the same cluster
        ++a;
      } else if (clustering1[i] != clustering1[j] &&
                 clustering2[i] != clustering2[j]) {
        // Both clusterings group the pair in different clusters
        ++b;
      }
      // If they are in different clusters in one clustering and in the same
      // cluster in the other, we don't need to do anything, as it does not
      // contribute to a or b.
    }
  }

  // Calculate Rand Index
  double randIndex = static_cast<double>(a + b) / (n * (n - 1) / 2);
  return randIndex;
}

int main() {
  // Example clusterings
  std::vector<int> clustering1 = {0, 0, 1, 1, 1};
  std::vector<int> clustering2 = {1, 1, 0, 0, 0};

  // Calculate Rand Index
  double randIndex = calculateRandIndex(clustering1, clustering2);

  // Output result
  if (randIndex >= 0) {
    std::cout << "Rand Index: " << randIndex << std::endl;
  } else {
    std::cerr << "Error calculating Rand Index.\n";
  }

  return 0;
}
