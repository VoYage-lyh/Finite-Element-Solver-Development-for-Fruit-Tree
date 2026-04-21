#pragma once

#include <array>

#include "orchard_solver/Common.h"

namespace orchard {

using Matrix12 = std::array<std::array<double, 12>, 12>;

struct BeamElementProperties {
    double youngs_modulus {0.0};
    double shear_modulus {0.0};
    double area {0.0};
    double iy {0.0};
    double iz {0.0};
    double torsion_constant {0.0};
    double density {0.0};
    double length {0.0};
};

class BeamElement {
public:
    [[nodiscard]] static Matrix12 buildLocalStiffnessMatrix(const BeamElementProperties& properties);
    [[nodiscard]] static Matrix12 buildLocalGeometricStiffnessMatrix(double axial_force, double length);
    [[nodiscard]] static Matrix12 buildLocalMassMatrix(const BeamElementProperties& properties);
    [[nodiscard]] static Matrix12 buildTransformationMatrix(
        const Vec3& start,
        const Vec3& end,
        Vec3 preferred_up = Vec3 {0.0, 0.0, 1.0}
    );
    [[nodiscard]] static Matrix12 transformToGlobal(const Matrix12& local_matrix, const Matrix12& transformation);
};

[[nodiscard]] Matrix12 zeroMatrix12();

} // namespace orchard
