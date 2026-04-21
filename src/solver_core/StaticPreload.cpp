#include "orchard_solver/solver_core/StaticPreload.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "orchard_solver/solver_core/LinearAlgebra.h"

namespace orchard {

namespace {

std::vector<std::vector<double>> copyDenseMatrix(const DenseMatrix& matrix) {
    std::vector<std::vector<double>> result(
        matrix.rows(),
        std::vector<double>(matrix.cols(), 0.0)
    );
    for (std::size_t row = 0; row < matrix.rows(); ++row) {
        for (std::size_t col = 0; col < matrix.cols(); ++col) {
            result[row][col] = matrix(row, col);
        }
    }
    return result;
}

std::vector<double> multiply(const DenseMatrix& matrix, const std::vector<double>& vector) {
    std::vector<double> result(matrix.rows(), 0.0);
    for (std::size_t row = 0; row < matrix.rows(); ++row) {
        for (std::size_t col = 0; col < matrix.cols(); ++col) {
            result[row] += matrix(row, col) * vector[col];
        }
    }
    return result;
}

std::array<double, 12> localDisplacement(
    const Matrix12& transformation,
    const std::vector<double>& displacement,
    const std::array<int, 12>& dofs
) {
    std::array<double, 12> global_element {};
    for (std::size_t index = 0; index < dofs.size(); ++index) {
        global_element[index] = displacement.at(static_cast<std::size_t>(dofs[index]));
    }

    std::array<double, 12> local {};
    for (std::size_t row = 0; row < 12; ++row) {
        for (std::size_t col = 0; col < 12; ++col) {
            local[row] += transformation[row][col] * global_element[col];
        }
    }
    return local;
}

} // namespace

std::unordered_map<std::string, std::vector<double>> computeGravityAxialForces(
    const AssembledModel& assembled,
    const std::vector<double>& gravity_load
) {
    if (gravity_load.size() != assembled.system.dof_labels.size()) {
        throw std::runtime_error("gravity_load size must match the assembled DOF count");
    }

    const auto stiffness = copyDenseMatrix(assembled.system.stiffness);
    const auto displacement = solveLinearSystem(stiffness, gravity_load);
    const auto residual = multiply(assembled.system.stiffness, displacement);

    double residual_norm = 0.0;
    for (std::size_t index = 0; index < gravity_load.size(); ++index) {
        residual_norm = std::max(residual_norm, std::abs(gravity_load[index] - residual[index]));
    }
    if (residual_norm > 1.0e-5) {
        throw std::runtime_error("Gravity preload solve residual is too large: " + std::to_string(residual_norm));
    }

    std::unordered_map<std::string, std::vector<double>> axial_forces;
    for (const auto& [branch_id, elements] : assembled.branch_elements) {
        auto& branch_forces = axial_forces[branch_id];
        branch_forces.reserve(elements.size());

        for (const auto& element : elements) {
            const auto local = localDisplacement(
                element.transformation,
                displacement,
                element.dofs
            );
            branch_forces.push_back(
                element.axial_rigidity * ((local[6] - local[0]) / element.length)
            );
        }
    }

    return axial_forces;
}

} // namespace orchard
