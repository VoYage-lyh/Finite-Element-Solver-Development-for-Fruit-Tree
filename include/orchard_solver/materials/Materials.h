#pragma once

#include <array>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "orchard_solver/Common.h"

namespace orchard {

struct MaterialProperties {
    std::string id;
    TissueType tissue {TissueType::Xylem};
    double density {0.0};
    double youngs_modulus {0.0};
    double damping_ratio {0.0};
    double nonlinear_alpha {0.0};
    bool orthotropic_enabled {false};
    std::array<double, 3> orthotropic_moduli {0.0, 0.0, 0.0};
};

class MaterialBase {
public:
    explicit MaterialBase(MaterialProperties properties);
    virtual ~MaterialBase() = default;

    [[nodiscard]] const MaterialProperties& properties() const noexcept;
    [[nodiscard]] virtual double tangentModulus(double generalized_strain) const = 0;

protected:
    MaterialProperties properties_;
};

class ElasticMaterial final : public MaterialBase {
public:
    using MaterialBase::MaterialBase;

    [[nodiscard]] double tangentModulus(double generalized_strain) const override;
};

class NonlinearElasticMaterial final : public MaterialBase {
public:
    using MaterialBase::MaterialBase;

    [[nodiscard]] double tangentModulus(double generalized_strain) const override;
};

class OrthotropicMaterialAdapter final : public MaterialBase {
public:
    using MaterialBase::MaterialBase;

    [[nodiscard]] double tangentModulus(double generalized_strain) const override;
};

class MaterialLibrary {
public:
    void addLinearElastic(const MaterialProperties& properties);
    void addNonlinearElastic(const MaterialProperties& properties);
    void addOrthotropicPlaceholder(const MaterialProperties& properties);

    [[nodiscard]] bool contains(const std::string& material_id) const noexcept;
    [[nodiscard]] const MaterialBase& require(const std::string& material_id) const;
    [[nodiscard]] std::vector<std::string> ids() const;

private:
    std::unordered_map<std::string, std::shared_ptr<MaterialBase>> materials_;
};

class SpatialMaterialField {
public:
    explicit SpatialMaterialField(const MaterialLibrary& materials);

    [[nodiscard]] const MaterialBase& resolve(const std::string& material_id, double station) const;

private:
    const MaterialLibrary* materials_ {nullptr};
};

} // namespace orchard
