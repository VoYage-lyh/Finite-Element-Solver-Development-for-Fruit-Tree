#include "orchard_solver/geometry_topology/TreeTopology.h"

#include <functional>
#include <queue>
#include <stdexcept>
#include <unordered_set>

namespace orchard {

BranchPath::BranchPath(const Vec3 start, const Vec3 end)
    : start_(start), end_(end), length_(distance(start, end)) {
}

const Vec3& BranchPath::start() const noexcept {
    return start_;
}

const Vec3& BranchPath::end() const noexcept {
    return end_;
}

double BranchPath::length() const noexcept {
    return length_;
}

void TreeTopology::addBranch(
    const std::string& branch_id,
    std::optional<std::string> parent_branch_id,
    const int level,
    const BranchPath& path
) {
    nodes_[branch_id] = TopologyNode {branch_id, std::move(parent_branch_id), level, path, {}};
}

void TreeTopology::rebuildChildLinks() {
    for (auto& [_, node] : nodes_) {
        node.child_branch_ids.clear();
    }

    for (auto& [branch_id, node] : nodes_) {
        if (node.parent_branch_id.has_value()) {
            const auto parent_it = nodes_.find(*node.parent_branch_id);
            if (parent_it != nodes_.end()) {
                parent_it->second.child_branch_ids.push_back(branch_id);
            }
        }
    }
}

bool TreeTopology::contains(const std::string& branch_id) const noexcept {
    return nodes_.contains(branch_id);
}

const TopologyNode& TreeTopology::requireNode(const std::string& branch_id) const {
    const auto it = nodes_.find(branch_id);
    if (it == nodes_.end()) {
        throw std::runtime_error("Topology node not found for branch: " + branch_id);
    }

    return it->second;
}

std::vector<std::string> TreeTopology::roots() const {
    std::vector<std::string> result;
    result.reserve(nodes_.size());

    for (const auto& [branch_id, node] : nodes_) {
        if (!node.parent_branch_id.has_value()) {
            result.push_back(branch_id);
        }
    }

    return result;
}

std::vector<std::string> TreeTopology::traversalOrder() const {
    std::vector<std::string> order;
    std::queue<std::string> queue;
    std::unordered_set<std::string> visited;

    for (const auto& root : roots()) {
        queue.push(root);
    }

    while (!queue.empty()) {
        const std::string current = queue.front();
        queue.pop();

        if (visited.contains(current)) {
            continue;
        }

        visited.insert(current);
        order.push_back(current);

        const auto& node = requireNode(current);
        for (const auto& child : node.child_branch_ids) {
            queue.push(child);
        }
    }

    return order;
}

bool TreeTopology::validate(std::string& error_message) const {
    if (nodes_.empty()) {
        error_message = "Tree topology is empty";
        return false;
    }

    if (roots().empty()) {
        error_message = "Tree topology has no root branch";
        return false;
    }

    for (const auto& [branch_id, node] : nodes_) {
        if (node.parent_branch_id.has_value()) {
            if (*node.parent_branch_id == branch_id) {
                error_message = "Branch cannot be its own parent: " + branch_id;
                return false;
            }

            if (!contains(*node.parent_branch_id)) {
                error_message = "Missing parent branch '" + *node.parent_branch_id + "' for branch '" + branch_id + "'";
                return false;
            }
        }
    }

    enum class VisitState { Unvisited, Visiting, Visited };
    std::unordered_map<std::string, VisitState> states;
    for (const auto& [branch_id, _] : nodes_) {
        states[branch_id] = VisitState::Unvisited;
    }

    std::function<bool(const std::string&)> visit = [&](const std::string& branch_id) -> bool {
        states[branch_id] = VisitState::Visiting;
        const auto& node = requireNode(branch_id);

        for (const auto& child : node.child_branch_ids) {
            if (states[child] == VisitState::Visiting) {
                error_message = "Cycle detected around branch: " + child;
                return false;
            }

            if (states[child] == VisitState::Unvisited && !visit(child)) {
                return false;
            }
        }

        states[branch_id] = VisitState::Visited;
        return true;
    };

    for (const auto& root : roots()) {
        if (states[root] == VisitState::Unvisited && !visit(root)) {
            return false;
        }
    }

    return true;
}

} // namespace orchard
