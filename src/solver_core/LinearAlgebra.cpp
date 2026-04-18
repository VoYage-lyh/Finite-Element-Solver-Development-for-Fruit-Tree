#include "orchard_solver/solver_core/LinearAlgebra.h"

#include <cmath>
#include <stdexcept>

namespace orchard {

DenseMatrix::DenseMatrix(const std::size_t rows, const std::size_t cols)
    : rows_(rows), cols_(cols), data_(rows * cols, 0.0) {
}

std::size_t DenseMatrix::rows() const noexcept {
    return rows_;
}

std::size_t DenseMatrix::cols() const noexcept {
    return cols_;
}

double& DenseMatrix::operator()(const std::size_t row, const std::size_t col) {
    return data_.at((row * cols_) + col);
}

const double& DenseMatrix::operator()(const std::size_t row, const std::size_t col) const {
    return data_.at((row * cols_) + col);
}

void DenseMatrix::add(const std::size_t row, const std::size_t col, const double value) {
    (*this)(row, col) += value;
}

std::vector<std::complex<double>> solveComplexLinearSystem(
    std::vector<std::vector<std::complex<double>>> matrix,
    std::vector<std::complex<double>> rhs
) {
    const std::size_t n = matrix.size();
    if (n == 0U) {
        return {};
    }

    for (const auto& row : matrix) {
        if (row.size() != n) {
            throw std::runtime_error("Linear solve requires a square matrix");
        }
    }

    if (rhs.size() != n) {
        throw std::runtime_error("Linear solve rhs size mismatch");
    }

    for (std::size_t pivot = 0; pivot < n; ++pivot) {
        std::size_t best_row = pivot;
        double best_value = std::abs(matrix[pivot][pivot]);

        for (std::size_t row = pivot + 1; row < n; ++row) {
            const double candidate = std::abs(matrix[row][pivot]);
            if (candidate > best_value) {
                best_value = candidate;
                best_row = row;
            }
        }

        if (best_value < 1.0e-14) {
            throw std::runtime_error("Linear system is singular or ill-conditioned");
        }

        if (best_row != pivot) {
            std::swap(matrix[pivot], matrix[best_row]);
            std::swap(rhs[pivot], rhs[best_row]);
        }

        for (std::size_t row = pivot + 1; row < n; ++row) {
            const auto factor = matrix[row][pivot] / matrix[pivot][pivot];
            if (std::abs(factor) < 1.0e-20) {
                continue;
            }

            for (std::size_t col = pivot; col < n; ++col) {
                matrix[row][col] -= factor * matrix[pivot][col];
            }
            rhs[row] -= factor * rhs[pivot];
        }
    }

    std::vector<std::complex<double>> solution(n, std::complex<double> {0.0, 0.0});
    for (std::size_t i = n; i-- > 0;) {
        auto sum = rhs[i];
        for (std::size_t col = i + 1; col < n; ++col) {
            sum -= matrix[i][col] * solution[col];
        }
        solution[i] = sum / matrix[i][i];
    }

    return solution;
}

std::vector<double> solveLinearSystem(
    std::vector<std::vector<double>> matrix,
    std::vector<double> rhs
) {
    const std::size_t n = matrix.size();
    if (n == 0U) {
        return {};
    }

    for (const auto& row : matrix) {
        if (row.size() != n) {
            throw std::runtime_error("Linear solve requires a square matrix");
        }
    }

    if (rhs.size() != n) {
        throw std::runtime_error("Linear solve rhs size mismatch");
    }

    for (std::size_t pivot = 0; pivot < n; ++pivot) {
        std::size_t best_row = pivot;
        double best_value = std::abs(matrix[pivot][pivot]);

        for (std::size_t row = pivot + 1; row < n; ++row) {
            const double candidate = std::abs(matrix[row][pivot]);
            if (candidate > best_value) {
                best_value = candidate;
                best_row = row;
            }
        }

        if (best_value < 1.0e-14) {
            throw std::runtime_error("Linear system is singular or ill-conditioned");
        }

        if (best_row != pivot) {
            std::swap(matrix[pivot], matrix[best_row]);
            std::swap(rhs[pivot], rhs[best_row]);
        }

        for (std::size_t row = pivot + 1; row < n; ++row) {
            const double factor = matrix[row][pivot] / matrix[pivot][pivot];
            if (std::abs(factor) < 1.0e-20) {
                continue;
            }

            for (std::size_t col = pivot; col < n; ++col) {
                matrix[row][col] -= factor * matrix[pivot][col];
            }
            rhs[row] -= factor * rhs[pivot];
        }
    }

    std::vector<double> solution(n, 0.0);
    for (std::size_t i = n; i-- > 0;) {
        double sum = rhs[i];
        for (std::size_t col = i + 1; col < n; ++col) {
            sum -= matrix[i][col] * solution[col];
        }
        solution[i] = sum / matrix[i][i];
    }

    return solution;
}

} // namespace orchard
