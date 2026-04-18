#pragma once

#include <memory>
#include <string>

namespace orchard {

enum class JointLawKind {
    None,
    Cubic,
    Gap
};

struct JointLawParameters {
    JointLawKind kind {JointLawKind::None};
    double linear_scale {1.0};
    double cubic_scale {0.0};
    double open_scale {1.0};
    double gap_threshold {0.0};
};

class JointConstitutiveLaw {
public:
    virtual ~JointConstitutiveLaw() = default;
    [[nodiscard]] virtual double stiffnessScale(double generalized_rotation) const = 0;
    [[nodiscard]] virtual JointLawParameters parameters() const = 0;
};

class PolynomialRotationalJointLaw final : public JointConstitutiveLaw {
public:
    PolynomialRotationalJointLaw(double linear_scale, double cubic_scale);

    [[nodiscard]] double stiffnessScale(double generalized_rotation) const override;
    [[nodiscard]] JointLawParameters parameters() const override;

private:
    double linear_scale_ {1.0};
    double cubic_scale_ {0.0};
};

class GapFrictionJointLaw final : public JointConstitutiveLaw {
public:
    GapFrictionJointLaw(double closed_scale, double open_scale, double gap_threshold);

    [[nodiscard]] double stiffnessScale(double generalized_rotation) const override;
    [[nodiscard]] JointLawParameters parameters() const override;

private:
    double closed_scale_ {1.0};
    double open_scale_ {0.0};
    double gap_threshold_ {0.0};
};

struct JointComponent {
    std::string id;
    std::string parent_branch_id;
    std::string child_branch_id;
    double linear_stiffness_scale {1.0};
    std::shared_ptr<JointConstitutiveLaw> law;

    [[nodiscard]] JointLawParameters parameters() const;
    [[nodiscard]] double linearizedScale() const;
    [[nodiscard]] double effectiveScale(double generalized_rotation = 0.0) const;
};

} // namespace orchard
