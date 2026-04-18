#pragma once

#include <map>
#include <string>
#include <variant>
#include <vector>

namespace orchard {

class JsonValue {
public:
    using object_t = std::map<std::string, JsonValue>;
    using array_t = std::vector<JsonValue>;
    using value_t = std::variant<std::nullptr_t, bool, double, std::string, array_t, object_t>;

    JsonValue() = default;
    explicit JsonValue(value_t value);

    [[nodiscard]] bool isNull() const;
    [[nodiscard]] bool isBool() const;
    [[nodiscard]] bool isNumber() const;
    [[nodiscard]] bool isString() const;
    [[nodiscard]] bool isArray() const;
    [[nodiscard]] bool isObject() const;

    [[nodiscard]] bool asBool() const;
    [[nodiscard]] double asNumber() const;
    [[nodiscard]] const std::string& asString() const;
    [[nodiscard]] const array_t& asArray() const;
    [[nodiscard]] const object_t& asObject() const;

private:
    value_t value_ {nullptr};
};

class JsonParser {
public:
    [[nodiscard]] JsonValue parse(const std::string& text) const;
};

[[nodiscard]] JsonValue parseJsonFile(const std::string& file_path);

} // namespace orchard
