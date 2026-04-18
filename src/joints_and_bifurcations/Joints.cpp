#include "orchard_solver/joints_and_bifurcations/Joints.h"

#include <cmath>

namespace orchard {

PolynomialRotationalJointLaw::PolynomialRotationalJointLaw(const double linear_scale, const double cubic_scale)
    : linear_scale_(linear_scale), cubic_scale_(cubic_scale) {
}

double PolynomialRotationalJointLaw::stiffnessScale(const double generalized_rotation) const {
    return linear_scale_ + (cubic_scale_ * generalized_rotation * generalized_rotation);
}

JointLawParameters PolynomialRotationalJointLaw::parameters() const {
    return JointLawParameters {
        JointLawKind::Cubic,
        linear_scale_,
        cubic_scale_,
        linear_scale_,
        0.0
    };
}

GapFrictionJointLaw::GapFrictionJointLaw(const double closed_scale, const double open_scale, const double gap_threshold)
    : closed_scale_(closed_scale), open_scale_(open_scale), gap_threshold_(gap_threshold) {
}

double GapFrictionJointLaw::stiffnessScale(const double generalized_rotation) const {
    return std::abs(generalized_rotation) < gap_threshold_ ? closed_scale_ : open_scale_;
}

JointLawParameters GapFrictionJointLaw::parameters() const {
    return JointLawParameters {
        JointLawKind::Gap,
        closed_scale_,
        0.0,
        open_scale_,
        gap_threshold_
    };
}

JointLawParameters JointComponent::parameters() const {
    if (law) {
        return law->parameters();
    }

    return JointLawParameters {};
}

double JointComponent::linearizedScale() const {
    return linear_stiffness_scale * parameters().linear_scale;
}

double JointComponent::effectiveScale(const double generalized_rotation) const {
    if (law) {
        return linear_stiffness_scale * law->stiffnessScale(generalized_rotation);
    }

    return linear_stiffness_scale;
}

} // namespace orchard
