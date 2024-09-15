#include <torch/torch.h>
#include <iostream>
#include <array>
#include <vector>
#include <cmath>    // For std::abs
#include <iomanip>  // For std::setprecision
#include "tests.h"
#include "debug.h"

// Existing regret_match function
std::array<double, TEST_MAX_ACTIONS> new_regret_match(const torch::Tensor& logits, int n_acts) {
    auto relu_logits = torch::relu(logits);
    
    double logits_sum = relu_logits.sum().item<double>();
    
    std::array<double, TEST_MAX_ACTIONS> strat;
    
    // If the sum is positive, calculate the strategy
    if (logits_sum > 0) {
        auto denominator = logits_sum;
        auto strategy_tensor = relu_logits / denominator;
        // Ensure strategy_tensor is detached and on CPU
        strategy_tensor = strategy_tensor.detach().cpu();
        // Copy to strat array
        for (int i = 0; i < n_acts; ++i) {
            strat[i] = static_cast<double>(strategy_tensor[i].item<float>());
        }
        // Fill remaining actions with 0.0 if n_acts < TEST_MAX_ACTIONS
        for (int i = n_acts; i < TEST_MAX_ACTIONS; ++i) {
            strat[i] = 0.0;
        }
    } 
    // If the sum is zero or negative, return a one-hot vector for the max logit
    else {
        auto max_index = torch::argmax(relu_logits).item<int>();
        std::fill(strat.begin(), strat.end(), 0.0);
        if (max_index < TEST_MAX_ACTIONS) { // Ensure max_index is within bounds
            strat[max_index] = 1.0;
        }
    }
    return strat;
}

// Function to check if two doubles are approximately equal
bool approx_equal(double a, double b, double tol = 1e-6) {
    return std::abs(a - b) < tol;
}

// Function to print a tensor (for debugging)
void print_tensor(const torch::Tensor& tensor) {
    std::cout << "[";
    for (int i = 0; i < tensor.size(0); ++i) {
        std::cout << tensor[i].item<float>();
        if (i != tensor.size(0) - 1) std::cout << ", ";
    }
    std::cout << "]";
}

// Test function
void test_regret_match() {
    std::cout << "Starting test_regret_match...\n\n";

    // Define test cases as vectors of logits
    struct TestCase {
        torch::Tensor logits;
        int n_acts;
        std::string description;
    };

    std::vector<TestCase> test_cases = {
        // Test Case 1: All positive logits
        {torch::tensor({1.0, 2.0, 3.0}), 3, "All positive logits"},
        // Test Case 2: Some zero logits
        {torch::tensor({0.0, 2.0, 3.0}), 3, "Some zero logits"},
        // Test Case 3: Negative logits
        {torch::tensor({-1.0, 2.0, 3.0}), 3, "Negative logits present"},
        // Test Case 4: All logits zero
        {torch::tensor({0.0, 0.0, 0.0}), 3, "All logits zero"},
        // Test Case 5: All logits negative
        {torch::tensor({-1.0, -2.0, -3.0}), 3, "All logits negative"},
        // Test Case 6: Single action
        {torch::tensor({5.0}), 1, "Single action"},
        // Test Case 7: Large number of actions (n_acts < TEST_MAX_ACTIONS)
        {torch::tensor({1.0, 2.0, 3.0, 4.0, 5.0}), 5, "Five actions"},
        // Test Case 8: n_acts less than TEST_MAX_ACTIONS with padding
        {torch::tensor({1.0, 0.0, 3.0}), 3, "Three actions with padding"},
        // Test Case 9: Logits leading to denominator zero in some actions
        {torch::tensor({1.0, 1.0, 2.0}), 3, "Denominator zero scenario"},
        // Test Case 10: Mixed positive, zero, and negative logits
        {torch::tensor({-1.0, 0.0, 2.0, 3.0, -2.0}), 5, "Mixed logits"}
    };

    int passed = 0;
    int failed = 0;

    for (size_t i = 0; i < test_cases.size(); ++i) {
        const auto& tc = test_cases[i];
        std::cout << "Test Case " << i + 1 << ": " << tc.description << "\n";
        std::cout << "Logits: ";
        print_tensor(tc.logits);
        std::cout << "\nNumber of Actions (n_acts): " << tc.n_acts << "\n";

        // Call regret_match
        std::array<double, TEST_MAX_ACTIONS> strat = new_regret_match(tc.logits, tc.n_acts);

        // Print strat
        std::cout << "Strategy (strat): [";
        for (int j = 0; j < tc.n_acts; ++j) {
            std::cout << std::fixed << std::setprecision(6) << strat[j];
            if (j != tc.n_acts - 1) std::cout << ", ";
        }
        std::cout << "]\n";

        // Check if strat is a probability distribution
        double sum = 0.0;
        for (int j = 0; j < tc.n_acts; ++j) {
            sum += strat[j];
        }
        bool is_prob_dist = approx_equal(sum, 1.0);

        std::cout << "Sum of strat: " << sum << " --> " << (is_prob_dist ? "PASS" : "FAIL") << "\n";

        // Additional checks based on the formula
        bool formula_correct = true;
        auto relu_logits = torch::relu(tc.logits);
        double logits_sum = relu_logits.sum().item<double>();

        if (logits_sum > 0) {
            // Calculate expected strat
            for (int j = 0; j < tc.n_acts; ++j) {
                double denominator = logits_sum;
                double expected;
                if (approx_equal(denominator, 0.0)) {
                    expected = 0.0; // Avoid division by zero
                } else {
                    expected = relu_logits[j].item<double>() / denominator;
                }
                // Compare expected and actual strat
                if (!approx_equal(strat[j], expected)) {
                    formula_correct = false;
                    std::cout << "  Mismatch at action " << j << ": expected " << expected 
                              << ", got " << strat[j] << "\n";
                }
            }
        } else {
            // One-hot encoding: only the max index should be 1.0
            int max_index = torch::argmax(relu_logits).item<int>();
            for (int j = 0; j < tc.n_acts; ++j) {
                double expected = (j == max_index) ? 1.0 : 0.0;
                if (!approx_equal(strat[j], expected)) {
                    formula_correct = false;
                    std::cout << "  Mismatch at action " << j << ": expected " << expected 
                              << ", got " << strat[j] << "\n";
                }
            }
        }

        if (is_prob_dist && formula_correct) {
            std::cout << "Result: PASS\n\n";
            passed++;
        } else {
            std::cout << "Result: FAIL\n\n";
            failed++;
        }
    }

    std::cout << "Test Summary:\n";
    std::cout << "Passed: " << passed << "\n";
    std::cout << "Failed: " << failed << "\n";

    if (failed == 0) {
        std::cout << "All test cases passed successfully!\n";
    } else {
        std::cout << "Some test cases failed. Please review the failures.\n";
    }
}

// Sample an action according to the strategy probabilities
int test_sample_action(const std::array<double, 5>& strat, int n_acts) {
    double r = static_cast<double>(rand()) / RAND_MAX;
    double cumulative = 0.0;
    for (int i = 0; i < n_acts; ++i) {
        cumulative += strat[i];
        if (r <= cumulative) {
            return i;
        }
    }
    return n_acts - 1; // Return last valid action if none selected
}

/*
int main() {
    std::array<double, 5> strat = {0.1, 0.1, 0.5, 0.2, 0.1};
    std::array<double, 5> sampled = {0.0, 0.0, 0.0, 0.0, 0.0};
    int n_acts = 5;
    int n_samples = 10000;
    for (size_t i=0; i<n_samples; ++i) {
        int s = test_sample_action(strat, n_acts);
        sampled[s] += 1.0;
    }
    for (size_t i=0; i<5; ++i) {
        sampled[i] /= static_cast<double>(n_samples);
        DEBUG_INFO("idx " << i << " = " << sampled[i]);
    }
    return 0;
}

// Example main function to run the test
int main() {
    test_regret_match();
    return 0;
}
*/
