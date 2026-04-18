#include "orchard_solver/io/SimpleJson.h"

#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace orchard {

JsonValue::JsonValue(value_t value)
    : value_(std::move(value)) {
}

bool JsonValue::isNull() const {
    return std::holds_alternative<std::nullptr_t>(value_);
}

bool JsonValue::isBool() const {
    return std::holds_alternative<bool>(value_);
}

bool JsonValue::isNumber() const {
    return std::holds_alternative<double>(value_);
}

bool JsonValue::isString() const {
    return std::holds_alternative<std::string>(value_);
}

bool JsonValue::isArray() const {
    return std::holds_alternative<array_t>(value_);
}

bool JsonValue::isObject() const {
    return std::holds_alternative<object_t>(value_);
}

bool JsonValue::asBool() const {
    return std::get<bool>(value_);
}

double JsonValue::asNumber() const {
    return std::get<double>(value_);
}

const std::string& JsonValue::asString() const {
    return std::get<std::string>(value_);
}

const JsonValue::array_t& JsonValue::asArray() const {
    return std::get<array_t>(value_);
}

const JsonValue::object_t& JsonValue::asObject() const {
    return std::get<object_t>(value_);
}

namespace {

class ParserImpl {
public:
    explicit ParserImpl(const std::string& text)
        : text_(text) {
    }

    JsonValue parse() {
        const JsonValue value = parseValue();
        skipWhitespace();
        if (position_ != text_.size()) {
            throw std::runtime_error("Unexpected trailing JSON content");
        }
        return value;
    }

private:
    JsonValue parseValue() {
        skipWhitespace();
        if (position_ >= text_.size()) {
            throw std::runtime_error("Unexpected end of JSON input");
        }

        const char ch = text_[position_];
        if (ch == '{') {
            return parseObject();
        }
        if (ch == '[') {
            return parseArray();
        }
        if (ch == '"') {
            return JsonValue(parseString());
        }
        if (ch == 't') {
            consumeKeyword("true");
            return JsonValue(true);
        }
        if (ch == 'f') {
            consumeKeyword("false");
            return JsonValue(false);
        }
        if (ch == 'n') {
            consumeKeyword("null");
            return JsonValue(nullptr);
        }
        if (ch == '-' || std::isdigit(static_cast<unsigned char>(ch))) {
            return JsonValue(parseNumber());
        }

        throw std::runtime_error(std::string("Unexpected character in JSON: ") + ch);
    }

    JsonValue parseObject() {
        consume('{');
        JsonValue::object_t object;
        skipWhitespace();
        if (tryConsume('}')) {
            return JsonValue(object);
        }

        while (true) {
            skipWhitespace();
            const std::string key = parseString();
            skipWhitespace();
            consume(':');
            object[key] = parseValue();
            skipWhitespace();
            if (tryConsume('}')) {
                break;
            }
            consume(',');
        }

        return JsonValue(object);
    }

    JsonValue parseArray() {
        consume('[');
        JsonValue::array_t array;
        skipWhitespace();
        if (tryConsume(']')) {
            return JsonValue(array);
        }

        while (true) {
            array.push_back(parseValue());
            skipWhitespace();
            if (tryConsume(']')) {
                break;
            }
            consume(',');
        }

        return JsonValue(array);
    }

    std::string parseString() {
        consume('"');
        std::string result;

        while (position_ < text_.size()) {
            const char ch = text_[position_++];
            if (ch == '"') {
                return result;
            }

            if (ch == '\\') {
                if (position_ >= text_.size()) {
                    throw std::runtime_error("Incomplete JSON escape sequence");
                }

                const char escape = text_[position_++];
                switch (escape) {
                case '"':
                case '\\':
                case '/':
                    result.push_back(escape);
                    break;
                case 'b':
                    result.push_back('\b');
                    break;
                case 'f':
                    result.push_back('\f');
                    break;
                case 'n':
                    result.push_back('\n');
                    break;
                case 'r':
                    result.push_back('\r');
                    break;
                case 't':
                    result.push_back('\t');
                    break;
                default:
                    throw std::runtime_error("Unsupported JSON escape sequence");
                }
                continue;
            }

            result.push_back(ch);
        }

        throw std::runtime_error("Unterminated JSON string");
    }

    double parseNumber() {
        const std::size_t start = position_;

        if (text_[position_] == '-') {
            ++position_;
        }

        consumeDigits();

        if (position_ < text_.size() && text_[position_] == '.') {
            ++position_;
            consumeDigits();
        }

        if (position_ < text_.size() && (text_[position_] == 'e' || text_[position_] == 'E')) {
            ++position_;
            if (position_ < text_.size() && (text_[position_] == '+' || text_[position_] == '-')) {
                ++position_;
            }
            consumeDigits();
        }

        return std::stod(text_.substr(start, position_ - start));
    }

    void consumeDigits() {
        const std::size_t start = position_;
        while (position_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[position_]))) {
            ++position_;
        }
        if (position_ == start) {
            throw std::runtime_error("Expected digit in JSON number");
        }
    }

    void consumeKeyword(const std::string& keyword) {
        if (text_.substr(position_, keyword.size()) != keyword) {
            throw std::runtime_error("Unexpected JSON keyword");
        }
        position_ += keyword.size();
    }

    void consume(const char expected) {
        skipWhitespace();
        if (position_ >= text_.size() || text_[position_] != expected) {
            throw std::runtime_error(std::string("Expected JSON character: ") + expected);
        }
        ++position_;
    }

    bool tryConsume(const char expected) {
        skipWhitespace();
        if (position_ < text_.size() && text_[position_] == expected) {
            ++position_;
            return true;
        }
        return false;
    }

    void skipWhitespace() {
        while (position_ < text_.size() && std::isspace(static_cast<unsigned char>(text_[position_]))) {
            ++position_;
        }
    }

    const std::string& text_;
    std::size_t position_ {0};
};

} // namespace

JsonValue JsonParser::parse(const std::string& text) const {
    ParserImpl parser(text);
    return parser.parse();
}

JsonValue parseJsonFile(const std::string& file_path) {
    std::ifstream stream(file_path);
    if (!stream) {
        throw std::runtime_error("Unable to open JSON file: " + file_path);
    }

    std::ostringstream buffer;
    buffer << stream.rdbuf();

    JsonParser parser;
    return parser.parse(buffer.str());
}

} // namespace orchard
