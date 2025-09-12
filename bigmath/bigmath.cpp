// Gneretated with Chat-GPT
// https://chatgpt.com/share/66e5100e-1548-8011-bbbe-46481157c6aa
//
// For 256-bit, this is much more complex
//
// https://chatgpt.com/share/66e51324-172c-8011-b75c-0297b8c87873

#include <iostream>
#include <algorithm> // For std::reverse
#include <string>    // For std::string

// Function to convert a string to __int128
__int128 stringToInt128(const std::string& str) {
    __int128 result = 0;
    bool isNegative = false;
    size_t i = 0;

    if (str[0] == '-') {
        isNegative = true;
        i = 1;
    }

    for (; i < str.length(); ++i) {
        result = result * 10 + (str[i] - '0');
    }

    if (isNegative) {
        result = -result;
    }

    return result;
}

// Function to print a __int128 number
void print128(__int128 num) {
    if (num == 0) {
        std::cout << "0";
        return;
    }

    __int128 temp = num;
    if (temp < 0) {
        std::cout << "-";
        temp = -temp;
    }

    std::string result;
    while (temp > 0) {
        result += '0' + temp % 10;
        temp /= 10;
    }

    std::reverse(result.begin(), result.end());
    std::cout << result;
}

int main() {
    // Use strings to represent large numbers
    std::string a_str = "12345678901234567890123456789012345";
    std::string b_str = "9876543210987654321098765432109876";

    // Convert strings to __int128
    __int128 a = stringToInt128(a_str);
    __int128 b = stringToInt128(b_str);

    // Perform arithmetic operations
    __int128 sum = a + b;
    __int128 diff = a - b;
    __int128 product = a * b;
    __int128 quotient = a / b;
    __int128 remainder = a % b;

    // Print results
    std::cout << "a = "; print128(a); std::cout << std::endl;
    std::cout << "b = "; print128(b); std::cout << std::endl;

    std::cout << "Sum: "; print128(sum); std::cout << std::endl;
    std::cout << "Difference: "; print128(diff); std::cout << std::endl;
    std::cout << "Product: "; print128(product); std::cout << std::endl;
    std::cout << "Quotient: "; print128(quotient); std::cout << std::endl;
    std::cout << "Remainder: "; print128(remainder); std::cout << std::endl;

    return 0;
}

