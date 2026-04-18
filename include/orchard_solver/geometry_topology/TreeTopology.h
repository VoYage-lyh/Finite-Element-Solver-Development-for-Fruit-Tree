#pragma once

#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "orchard_solver/Common.h"

namespace orchard {

class BranchPath {
public:
    BranchPath() = default;
    BranchPath(Vec3 start, Vec3 end);

    [[nodiscard]] const Vec3& start() const noexcept;
    [[nodiscard]] const Vec3& end() const noexcept;
    [[nodiscard]] double length() const noexcept;
    [[nodiscard]] Vec3 pointAt(double station) const noexcept;
    [[nodiscard]] Vec3 direction() const;

private:
    Vec3 start_ {};
    Vec3 end_ {};
    double length_ {0.0};
};

struct ObservationPoint {
    std::string id;
    std::string target_type;
    std::string target_id;
    std::string target_node {"tip"};
    std::string target_component {"ux"};
};

struct TopologyNode {
    std::string branch_id;
    std::optional<std::string> parent_branch_id;
    int level {0};
    BranchPath path;
    std::vector<std::string> child_branch_ids;
};

class TreeTopology {
public:
    void addBranch(
        const std::string& branch_id,
        std::optional<std::string> parent_branch_id,
        int level,
        const BranchPath& path
    );

    void rebuildChildLinks();

    [[nodiscard]] bool contains(const std::string& branch_id) const noexcept;
    [[nodiscard]] const TopologyNode& requireNode(const std::string& branch_id) const;
    [[nodiscard]] std::vector<std::string> roots() const;
    [[nodiscard]] std::vector<std::string> traversalOrder() const;
    [[nodiscard]] bool validate(std::string& error_message) const;

private:
    std::unordered_map<std::string, TopologyNode> nodes_;
};

} // namespace orchard
