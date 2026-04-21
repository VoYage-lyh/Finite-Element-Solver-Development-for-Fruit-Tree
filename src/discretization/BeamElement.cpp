#include "orchard_solver/discretization/BeamElement.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace orchard {

namespace {

Matrix12 transpose(const Matrix12& matrix) {
    Matrix12 result = zeroMatrix12();
    for (std::size_t row = 0; row < 12; ++row) {
        for (std::size_t col = 0; col < 12; ++col) {
            result[row][col] = matrix[col][row];
        }
    }
    return result;
}

Matrix12 multiply(const Matrix12& left, const Matrix12& right) {
    Matrix12 result = zeroMatrix12();
    for (std::size_t row = 0; row < 12; ++row) {
        for (std::size_t col = 0; col < 12; ++col) {
            for (std::size_t inner = 0; inner < 12; ++inner) {
                result[row][col] += left[row][inner] * right[inner][col];
            }
        }
    }
    return result;
}

void addSubmatrix(
    Matrix12& target,
    const std::array<int, 4>& indices,
    const std::array<std::array<double, 4>, 4>& source
) {
    for (std::size_t row = 0; row < 4; ++row) {
        for (std::size_t col = 0; col < 4; ++col) {
            target[static_cast<std::size_t>(indices[row])][static_cast<std::size_t>(indices[col])] += source[row][col];
        }
    }
}

std::array<std::array<double, 4>, 4> buildPlaneBendingStiffness(const double flexural_rigidity, const double length) {
    const double l2 = length * length;
    const double l3 = l2 * length;
    const double scale = flexural_rigidity / l3;

    return {{
        {{12.0 * scale, 6.0 * length * scale, -12.0 * scale, 6.0 * length * scale}},
        {{6.0 * length * scale, 4.0 * l2 * scale, -6.0 * length * scale, 2.0 * l2 * scale}},
        {{-12.0 * scale, -6.0 * length * scale, 12.0 * scale, -6.0 * length * scale}},
        {{6.0 * length * scale, 2.0 * l2 * scale, -6.0 * length * scale, 4.0 * l2 * scale}}
    }};
}

std::array<std::array<double, 4>, 4> buildPlaneBendingMass(const double density_area, const double length) {
    const double l2 = length * length;
    const double scale = density_area * length / 420.0;

    return {{
        {{156.0 * scale, 22.0 * length * scale, 54.0 * scale, -13.0 * length * scale}},
        {{22.0 * length * scale, 4.0 * l2 * scale, 13.0 * length * scale, -3.0 * l2 * scale}},
        {{54.0 * scale, 13.0 * length * scale, 156.0 * scale, -22.0 * length * scale}},
        {{-13.0 * length * scale, -3.0 * l2 * scale, -22.0 * length * scale, 4.0 * l2 * scale}}
    }};
}

} // namespace

Matrix12 zeroMatrix12() {
    Matrix12 result {};
    for (auto& row : result) {
        row.fill(0.0);
    }
    return result;
}

Matrix12 BeamElement::buildLocalStiffnessMatrix(const BeamElementProperties& properties) {
    if (properties.length <= 0.0) {
        throw std::runtime_error("Beam element length must be positive");
    }

    Matrix12 matrix = zeroMatrix12();
    const double axial = properties.youngs_modulus * properties.area / properties.length;
    const double torsion = properties.shear_modulus * properties.torsion_constant / properties.length;

    matrix[0][0] = axial;
    matrix[0][6] = -axial;
    matrix[6][0] = -axial;
    matrix[6][6] = axial;

    matrix[3][3] = torsion;
    matrix[3][9] = -torsion;
    matrix[9][3] = -torsion;
    matrix[9][9] = torsion;

    const auto bending_z = buildPlaneBendingStiffness(properties.youngs_modulus * properties.iz, properties.length);
    addSubmatrix(matrix, {1, 5, 7, 11}, bending_z);

    const auto bending_y_base = buildPlaneBendingStiffness(properties.youngs_modulus * properties.iy, properties.length);
    const std::array<double, 4> sign = {1.0, -1.0, 1.0, -1.0};
    std::array<std::array<double, 4>, 4> bending_y {};
    for (std::size_t row = 0; row < 4; ++row) {
        for (std::size_t col = 0; col < 4; ++col) {
            bending_y[row][col] = sign[row] * bending_y_base[row][col] * sign[col];
        }
    }
    addSubmatrix(matrix, {2, 4, 8, 10}, bending_y);

    return matrix;
}

Matrix12 BeamElement::buildLocalGeometricStiffnessMatrix(const double axial_force, const double length) {
    if (length <= 0.0) {
        throw std::runtime_error("Beam element length must be positive");
    }

    Matrix12 matrix = zeroMatrix12();
    const double scale = axial_force / (30.0 * length);
    const double l2 = length * length;
    const std::array<std::array<double, 4>, 4> base {{
        {{36.0 * scale, 3.0 * length * scale, -36.0 * scale, 3.0 * length * scale}},
        {{3.0 * length * scale, 4.0 * l2 * scale, -3.0 * length * scale, -1.0 * l2 * scale}},
        {{-36.0 * scale, -3.0 * length * scale, 36.0 * scale, -3.0 * length * scale}},
        {{3.0 * length * scale, -1.0 * l2 * scale, -3.0 * length * scale, 4.0 * l2 * scale}}
    }};
    addSubmatrix(matrix, {1, 5, 7, 11}, base);

    const std::array<double, 4> sign = {1.0, -1.0, 1.0, -1.0};
    std::array<std::array<double, 4>, 4> bending_y {};
    for (std::size_t row = 0; row < 4; ++row) {
        for (std::size_t col = 0; col < 4; ++col) {
            bending_y[row][col] = sign[row] * base[row][col] * sign[col];
        }
    }
    addSubmatrix(matrix, {2, 4, 8, 10}, bending_y);

    return matrix;
}

Matrix12 BeamElement::buildLocalMassMatrix(const BeamElementProperties& properties) {
    if (properties.length <= 0.0) {
        throw std::runtime_error("Beam element length must be positive");
    }

    Matrix12 matrix = zeroMatrix12();
    const double density_area = properties.density * properties.area;
    const double axial_scale = density_area * properties.length / 6.0;
    const double torsional_scale = properties.density * properties.torsion_constant * properties.length / 6.0;

    matrix[0][0] = 2.0 * axial_scale;
    matrix[0][6] = axial_scale;
    matrix[6][0] = axial_scale;
    matrix[6][6] = 2.0 * axial_scale;

    matrix[3][3] = 2.0 * torsional_scale;
    matrix[3][9] = torsional_scale;
    matrix[9][3] = torsional_scale;
    matrix[9][9] = 2.0 * torsional_scale;

    const auto bending_z = buildPlaneBendingMass(density_area, properties.length);
    addSubmatrix(matrix, {1, 5, 7, 11}, bending_z);

    const auto bending_y_base = buildPlaneBendingMass(density_area, properties.length);
    const std::array<double, 4> sign = {1.0, -1.0, 1.0, -1.0};
    std::array<std::array<double, 4>, 4> bending_y {};
    for (std::size_t row = 0; row < 4; ++row) {
        for (std::size_t col = 0; col < 4; ++col) {
            bending_y[row][col] = sign[row] * bending_y_base[row][col] * sign[col];
        }
    }
    addSubmatrix(matrix, {2, 4, 8, 10}, bending_y);

    return matrix;
}

Matrix12 BeamElement::buildTransformationMatrix(
    const Vec3& start,
    const Vec3& end,
    Vec3 preferred_up
) {
    const Vec3 local_x = normalize(end - start);
    preferred_up = normalize(preferred_up);

    if (std::abs(dot(local_x, preferred_up)) > 0.95) {
        preferred_up = Vec3 {0.0, 1.0, 0.0};
        if (std::abs(dot(local_x, preferred_up)) > 0.95) {
            preferred_up = Vec3 {1.0, 0.0, 0.0};
        }
    }

    const Vec3 local_y = normalize(cross(preferred_up, local_x));
    const Vec3 local_z = normalize(cross(local_x, local_y));

    const std::array<std::array<double, 3>, 3> rotation {{
        {{local_x.x, local_x.y, local_x.z}},
        {{local_y.x, local_y.y, local_y.z}},
        {{local_z.x, local_z.y, local_z.z}}
    }};

    Matrix12 transformation = zeroMatrix12();
    for (int block = 0; block < 4; ++block) {
        const int offset = block * 3;
        for (std::size_t row = 0; row < 3; ++row) {
            for (std::size_t col = 0; col < 3; ++col) {
                transformation[static_cast<std::size_t>(offset) + row][static_cast<std::size_t>(offset) + col] = rotation[row][col];
            }
        }
    }

    return transformation;
}

Matrix12 BeamElement::transformToGlobal(const Matrix12& local_matrix, const Matrix12& transformation) {
    return multiply(transpose(transformation), multiply(local_matrix, transformation));
}

} // namespace orchard
