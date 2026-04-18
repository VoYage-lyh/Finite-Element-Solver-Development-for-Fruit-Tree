#include "orchard_solver/materials/Materials.h"

#include <algorithm>
#include <stdexcept>

namespace orchard {

MaterialBase::MaterialBase(MaterialProperties properties)
    : properties_(std::move(properties)) {
}

const MaterialProperties& MaterialBase::properties() const noexcept {
    return properties_;
}

double ElasticMaterial::tangentModulus(const double /*generalized_strain*/) const {
    return properties_.youngs_modulus;
}

double NonlinearElasticMaterial::tangentModulus(const double generalized_strain) const {
    return properties_.youngs_modulus * (1.0 + (properties_.nonlinear_alpha * generalized_strain * generalized_strain));
}

double OrthotropicMaterialAdapter::tangentModulus(const double /*generalized_strain*/) const {
    if (properties_.orthotropic_enabled) {
        return std::max({properties_.orthotropic_moduli[0], properties_.orthotropic_moduli[1], properties_.orthotropic_moduli[2]});
    }

    return properties_.youngs_modulus;
}

void MaterialLibrary::addLinearElastic(const MaterialProperties& properties) {
    materials_[properties.id] = std::make_shared<ElasticMaterial>(properties);
}

void MaterialLibrary::addNonlinearElastic(const MaterialProperties& properties) {
    materials_[properties.id] = std::make_shared<NonlinearElasticMaterial>(properties);
}

void MaterialLibrary::addOrthotropicPlaceholder(const MaterialProperties& properties) {
    materials_[properties.id] = std::make_shared<OrthotropicMaterialAdapter>(properties);
}

bool MaterialLibrary::contains(const std::string& material_id) const noexcept {
    return materials_.contains(material_id);
}

const MaterialBase& MaterialLibrary::require(const std::string& material_id) const {
    const auto it = materials_.find(material_id);
    if (it == materials_.end()) {
        throw std::runtime_error("Material not found: " + material_id);
    }

    return *(it->second);
}

std::vector<std::string> MaterialLibrary::ids() const {
    std::vector<std::string> result;
    result.reserve(materials_.size());

    for (const auto& [material_id, _] : materials_) {
        result.push_back(material_id);
    }

    return result;
}

SpatialMaterialField::SpatialMaterialField(const MaterialLibrary& materials)
    : materials_(&materials) {
}

const MaterialBase& SpatialMaterialField::resolve(const std::string& material_id, const double /*station*/) const {
    return materials_->require(material_id);
}

} // namespace orchard
