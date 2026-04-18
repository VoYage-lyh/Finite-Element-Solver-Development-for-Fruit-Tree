#pragma once

#include <complex>
#include <cstddef>
#include <vector>

namespace orchard {

class DenseMatrix {
public:
    DenseMatrix() = default;
    DenseMatrix(std::size_t rows, std::size_t cols);

    [[nodiscard]] std::size_t rows() const noexcept;
    [[nodiscard]] std::size_t cols() const noexcept;

    double& operator()(std::size_t row, std::size_t col);
    const double& operator()(std::size_t row, std::size_t col) const;

    void add(std::size_t row, std::size_t col, double value);

private:
    std::size_t rows_ {0};
    std::size_t cols_ {0};
    std::vector<double> data_;
};

[[nodiscard]] std::vector<std::complex<double>> solveComplexLinearSystem(
    std::vector<std::vector<std::complex<double>>> matrix,
    std::vector<std::complex<double>> rhs
);

[[nodiscard]] std::vector<double> solveLinearSystem(
    std::vector<std::vector<double>> matrix,
    std::vector<double> rhs
);

} // namespace orchard
