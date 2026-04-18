#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "orchard_solver/Common.h"
#include "orchard_solver/discretization/BeamElement.h"
#include "orchard_solver/solver_core/DynamicSystem.h"
#include "orchard_solver/solver_core/LinearAlgebra.h"

namespace verification {

using Dense = std::vector<std::vector<double>>;

inline void check(const bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

inline void checkClose(const double actual, const double expected, const double tolerance, const std::string& message) {
    if (std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(message + " actual=" + std::to_string(actual) + " expected=" + std::to_string(expected));
    }
}

inline Dense zeros(const int size) {
    return Dense(static_cast<std::size_t>(size), std::vector<double>(static_cast<std::size_t>(size), 0.0));
}

inline Dense identity(const int size) {
    Dense result = zeros(size);
    for (int i = 0; i < size; ++i) {
        result[static_cast<std::size_t>(i)][static_cast<std::size_t>(i)] = 1.0;
    }
    return result;
}

inline std::vector<double> zerosVector(const int size) {
    return std::vector<double>(static_cast<std::size_t>(size), 0.0);
}

inline Dense transpose(const Dense& matrix) {
    const int rows = static_cast<int>(matrix.size());
    const int cols = rows == 0 ? 0 : static_cast<int>(matrix.front().size());
    Dense result(static_cast<std::size_t>(cols), std::vector<double>(static_cast<std::size_t>(rows), 0.0));

    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            result[static_cast<std::size_t>(col)][static_cast<std::size_t>(row)] = matrix[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)];
        }
    }

    return result;
}

inline Dense multiply(const Dense& left, const Dense& right) {
    const int rows = static_cast<int>(left.size());
    const int inner = rows == 0 ? 0 : static_cast<int>(left.front().size());
    const int cols = right.empty() ? 0 : static_cast<int>(right.front().size());
    Dense result(static_cast<std::size_t>(rows), std::vector<double>(static_cast<std::size_t>(cols), 0.0));

    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            for (int k = 0; k < inner; ++k) {
                result[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)] += left[static_cast<std::size_t>(row)][static_cast<std::size_t>(k)] * right[static_cast<std::size_t>(k)][static_cast<std::size_t>(col)];
            }
        }
    }

    return result;
}

inline std::vector<double> multiply(const Dense& matrix, const std::vector<double>& vector) {
    std::vector<double> result(matrix.size(), 0.0);
    for (std::size_t row = 0; row < matrix.size(); ++row) {
        for (std::size_t col = 0; col < matrix[row].size(); ++col) {
            result[row] += matrix[row][col] * vector[col];
        }
    }
    return result;
}

inline Dense choleskyLower(const Dense& matrix) {
    const int n = static_cast<int>(matrix.size());
    Dense lower = zeros(n);

    for (int row = 0; row < n; ++row) {
        for (int col = 0; col <= row; ++col) {
            double sum = matrix[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)];
            for (int k = 0; k < col; ++k) {
                sum -= lower[static_cast<std::size_t>(row)][static_cast<std::size_t>(k)] * lower[static_cast<std::size_t>(col)][static_cast<std::size_t>(k)];
            }

            if (row == col) {
                if (sum <= 0.0) {
                    throw std::runtime_error("Cholesky decomposition requires SPD matrix");
                }
                lower[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)] = std::sqrt(sum);
            } else {
                lower[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)] = sum / lower[static_cast<std::size_t>(col)][static_cast<std::size_t>(col)];
            }
        }
    }

    return lower;
}

inline Dense inverseLower(const Dense& lower) {
    const int n = static_cast<int>(lower.size());
    Dense inverse = zeros(n);

    for (int col = 0; col < n; ++col) {
        inverse[static_cast<std::size_t>(col)][static_cast<std::size_t>(col)] = 1.0 / lower[static_cast<std::size_t>(col)][static_cast<std::size_t>(col)];
        for (int row = col + 1; row < n; ++row) {
            double sum = 0.0;
            for (int k = col; k < row; ++k) {
                sum += lower[static_cast<std::size_t>(row)][static_cast<std::size_t>(k)] * inverse[static_cast<std::size_t>(k)][static_cast<std::size_t>(col)];
            }
            inverse[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)] = -sum / lower[static_cast<std::size_t>(row)][static_cast<std::size_t>(row)];
        }
    }

    return inverse;
}

struct EigenResult {
    std::vector<double> eigenvalues;
    Dense eigenvectors;
};

inline EigenResult jacobiEigenDecomposition(Dense matrix, const double tolerance = 1.0e-12, const int max_iterations = -1) {
    const int n = static_cast<int>(matrix.size());
    Dense eigenvectors = identity(n);
    const int iteration_limit = max_iterations > 0 ? max_iterations : std::max(200, 50 * n * n);

    for (int iteration = 0; iteration < iteration_limit; ++iteration) {
        int pivot_row = 0;
        int pivot_col = 1;
        double max_off_diagonal = 0.0;

        for (int row = 0; row < n; ++row) {
            for (int col = row + 1; col < n; ++col) {
                const double value = std::abs(matrix[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)]);
                if (value > max_off_diagonal) {
                    max_off_diagonal = value;
                    pivot_row = row;
                    pivot_col = col;
                }
            }
        }

        if (max_off_diagonal < tolerance) {
            break;
        }

        const double app = matrix[static_cast<std::size_t>(pivot_row)][static_cast<std::size_t>(pivot_row)];
        const double aqq = matrix[static_cast<std::size_t>(pivot_col)][static_cast<std::size_t>(pivot_col)];
        const double apq = matrix[static_cast<std::size_t>(pivot_row)][static_cast<std::size_t>(pivot_col)];
        const double tau = (aqq - app) / (2.0 * apq);
        const double t = (tau >= 0.0 ? 1.0 : -1.0) / (std::abs(tau) + std::sqrt(1.0 + (tau * tau)));
        const double c = 1.0 / std::sqrt(1.0 + (t * t));
        const double s = t * c;

        for (int k = 0; k < n; ++k) {
            if (k == pivot_row || k == pivot_col) {
                continue;
            }

            const double aik = matrix[static_cast<std::size_t>(pivot_row)][static_cast<std::size_t>(k)];
            const double akq = matrix[static_cast<std::size_t>(pivot_col)][static_cast<std::size_t>(k)];
            matrix[static_cast<std::size_t>(pivot_row)][static_cast<std::size_t>(k)] = (c * aik) - (s * akq);
            matrix[static_cast<std::size_t>(k)][static_cast<std::size_t>(pivot_row)] = matrix[static_cast<std::size_t>(pivot_row)][static_cast<std::size_t>(k)];
            matrix[static_cast<std::size_t>(pivot_col)][static_cast<std::size_t>(k)] = (s * aik) + (c * akq);
            matrix[static_cast<std::size_t>(k)][static_cast<std::size_t>(pivot_col)] = matrix[static_cast<std::size_t>(pivot_col)][static_cast<std::size_t>(k)];
        }

        matrix[static_cast<std::size_t>(pivot_row)][static_cast<std::size_t>(pivot_row)] = (c * c * app) - (2.0 * s * c * apq) + (s * s * aqq);
        matrix[static_cast<std::size_t>(pivot_col)][static_cast<std::size_t>(pivot_col)] = (s * s * app) + (2.0 * s * c * apq) + (c * c * aqq);
        matrix[static_cast<std::size_t>(pivot_row)][static_cast<std::size_t>(pivot_col)] = 0.0;
        matrix[static_cast<std::size_t>(pivot_col)][static_cast<std::size_t>(pivot_row)] = 0.0;

        for (int k = 0; k < n; ++k) {
            const double vip = eigenvectors[static_cast<std::size_t>(k)][static_cast<std::size_t>(pivot_row)];
            const double viq = eigenvectors[static_cast<std::size_t>(k)][static_cast<std::size_t>(pivot_col)];
            eigenvectors[static_cast<std::size_t>(k)][static_cast<std::size_t>(pivot_row)] = (c * vip) - (s * viq);
            eigenvectors[static_cast<std::size_t>(k)][static_cast<std::size_t>(pivot_col)] = (s * vip) + (c * viq);
        }
    }

    double residual = 0.0;
    for (int row = 0; row < n; ++row) {
        for (int col = row + 1; col < n; ++col) {
            residual = std::max(residual, std::abs(matrix[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)]));
        }
    }
    if (residual >= tolerance * 10.0) {
        throw std::runtime_error("Jacobi eigen decomposition did not converge");
    }

    EigenResult result;
    result.eigenvalues.reserve(static_cast<std::size_t>(n));
    result.eigenvectors = std::move(eigenvectors);
    for (int i = 0; i < n; ++i) {
        result.eigenvalues.push_back(matrix[static_cast<std::size_t>(i)][static_cast<std::size_t>(i)]);
    }

    std::vector<int> ordering(static_cast<std::size_t>(n), 0);
    std::iota(ordering.begin(), ordering.end(), 0);
    std::sort(ordering.begin(), ordering.end(), [&](const int left, const int right) {
        return result.eigenvalues[static_cast<std::size_t>(left)] < result.eigenvalues[static_cast<std::size_t>(right)];
    });

    std::vector<double> sorted_values;
    Dense sorted_vectors(static_cast<std::size_t>(n), std::vector<double>(static_cast<std::size_t>(n), 0.0));
    sorted_values.reserve(static_cast<std::size_t>(n));
    for (int sorted_index = 0; sorted_index < n; ++sorted_index) {
        const int source = ordering[static_cast<std::size_t>(sorted_index)];
        sorted_values.push_back(result.eigenvalues[static_cast<std::size_t>(source)]);
        for (int row = 0; row < n; ++row) {
            sorted_vectors[static_cast<std::size_t>(row)][static_cast<std::size_t>(sorted_index)] = result.eigenvectors[static_cast<std::size_t>(row)][static_cast<std::size_t>(source)];
        }
    }

    result.eigenvalues = std::move(sorted_values);
    result.eigenvectors = std::move(sorted_vectors);
    return result;
}

inline Dense reducedMatrix(const Dense& matrix, const std::vector<int>& free_dofs) {
    Dense reduced(static_cast<std::size_t>(free_dofs.size()), std::vector<double>(free_dofs.size(), 0.0));
    for (std::size_t row = 0; row < free_dofs.size(); ++row) {
        for (std::size_t col = 0; col < free_dofs.size(); ++col) {
            reduced[row][col] = matrix[static_cast<std::size_t>(free_dofs[row])][static_cast<std::size_t>(free_dofs[col])];
        }
    }
    return reduced;
}

inline std::vector<double> reducedVector(const std::vector<double>& vector, const std::vector<int>& free_dofs) {
    std::vector<double> reduced;
    reduced.reserve(free_dofs.size());
    for (const auto dof : free_dofs) {
        reduced.push_back(vector[static_cast<std::size_t>(dof)]);
    }
    return reduced;
}

inline std::vector<int> complementDofs(const int total_dofs, const std::vector<int>& fixed_dofs) {
    std::vector<bool> is_fixed(static_cast<std::size_t>(total_dofs), false);
    for (const auto dof : fixed_dofs) {
        is_fixed[static_cast<std::size_t>(dof)] = true;
    }

    std::vector<int> free_dofs;
    for (int dof = 0; dof < total_dofs; ++dof) {
        if (!is_fixed[static_cast<std::size_t>(dof)]) {
            free_dofs.push_back(dof);
        }
    }
    return free_dofs;
}

inline std::vector<double> solveStaticSystem(
    const Dense& stiffness,
    const std::vector<double>& force,
    const std::vector<int>& fixed_dofs
) {
    const auto free_dofs = complementDofs(static_cast<int>(force.size()), fixed_dofs);
    const auto reduced_stiffness = reducedMatrix(stiffness, free_dofs);
    const auto reduced_force = reducedVector(force, free_dofs);
    const auto reduced_solution = orchard::solveLinearSystem(reduced_stiffness, reduced_force);

    std::vector<double> full_solution(force.size(), 0.0);
    for (std::size_t i = 0; i < free_dofs.size(); ++i) {
        full_solution[static_cast<std::size_t>(free_dofs[i])] = reduced_solution[i];
    }
    return full_solution;
}

inline std::vector<double> generalizedFrequencies(
    const Dense& stiffness,
    const Dense& mass,
    const std::vector<int>& fixed_dofs,
    const int count
) {
    const auto free_dofs = complementDofs(static_cast<int>(stiffness.size()), fixed_dofs);
    const auto reduced_stiffness = reducedMatrix(stiffness, free_dofs);
    const auto reduced_mass = reducedMatrix(mass, free_dofs);

    const auto lower = choleskyLower(reduced_mass);
    const auto inverse_lower = inverseLower(lower);
    const auto transformed = multiply(inverse_lower, multiply(reduced_stiffness, transpose(inverse_lower)));
    const auto eigen = jacobiEigenDecomposition(transformed);

    std::vector<double> frequencies;
    for (const auto eigenvalue : eigen.eigenvalues) {
        if (eigenvalue <= 1.0e-8) {
            continue;
        }
        frequencies.push_back(std::sqrt(eigenvalue) / (2.0 * 3.14159265358979323846));
        if (static_cast<int>(frequencies.size()) == count) {
            break;
        }
    }

    if (static_cast<int>(frequencies.size()) < count) {
        throw std::runtime_error("Unable to extract requested number of physical frequencies");
    }

    return frequencies;
}

struct PlanarBeamSystem {
    Dense stiffness;
    Dense mass;
};

inline std::array<std::array<double, 4>, 4> extractPlanarBlock(const orchard::Matrix12& matrix) {
    constexpr std::array<int, 4> indices = {1, 5, 7, 11};
    std::array<std::array<double, 4>, 4> result {};
    for (std::size_t row = 0; row < 4; ++row) {
        for (std::size_t col = 0; col < 4; ++col) {
            result[row][col] = matrix[static_cast<std::size_t>(indices[row])][static_cast<std::size_t>(indices[col])];
        }
    }
    return result;
}

inline PlanarBeamSystem buildUniformPlanarBeam(
    const int num_elements,
    const double length,
    const double youngs_modulus,
    const double density,
    const double area,
    const double inertia
) {
    const int nodes = num_elements + 1;
    const int dofs = 2 * nodes;
    PlanarBeamSystem system {zeros(dofs), zeros(dofs)};

    const orchard::BeamElementProperties properties {
        youngs_modulus,
        youngs_modulus / (2.0 * (1.0 + 0.3)),
        area,
        inertia,
        inertia,
        2.0 * inertia,
        density,
        length / static_cast<double>(num_elements)
    };

    const auto local_stiffness = extractPlanarBlock(orchard::BeamElement::buildLocalStiffnessMatrix(properties));
    const auto local_mass = extractPlanarBlock(orchard::BeamElement::buildLocalMassMatrix(properties));

    for (int element = 0; element < num_elements; ++element) {
        const std::array<int, 4> element_dofs = {
            2 * element,
            (2 * element) + 1,
            (2 * (element + 1)),
            (2 * (element + 1)) + 1
        };

        for (std::size_t row = 0; row < 4; ++row) {
            for (std::size_t col = 0; col < 4; ++col) {
                system.stiffness[static_cast<std::size_t>(element_dofs[row])][static_cast<std::size_t>(element_dofs[col])] += local_stiffness[row][col];
                system.mass[static_cast<std::size_t>(element_dofs[row])][static_cast<std::size_t>(element_dofs[col])] += local_mass[row][col];
            }
        }
    }

    return system;
}

inline double l2Norm(const std::vector<double>& values) {
    double sum = 0.0;
    for (const auto value : values) {
        sum += value * value;
    }
    return std::sqrt(sum);
}

inline double estimateSteadyAmplitude(const orchard::TimeHistoryResult& response) {
    const std::size_t start = response.points.size() / 2;
    double amplitude = 0.0;
    for (std::size_t i = start; i < response.points.size(); ++i) {
        amplitude = std::max(amplitude, std::abs(response.points[i].observation_values.front()));
    }
    return amplitude;
}

} // namespace verification
