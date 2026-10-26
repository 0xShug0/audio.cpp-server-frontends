#include "frontend.h"

#include "common.h"

#include <cstdlib>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace minitts::server {
namespace {

constexpr const char * token_env = "AUDIOCPP_BEARER_AUTH_TOKEN";
constexpr const char * public_routes_env = "AUDIOCPP_BEARER_AUTH_PUBLIC_ROUTES";
constexpr const char * allow_routes_env = "AUDIOCPP_BEARER_AUTH_ALLOW_ROUTES";
constexpr const char * deny_routes_env = "AUDIOCPP_BEARER_AUTH_DENY_ROUTES";
constexpr const char * default_public_routes = "GET /health";

constexpr std::string_view whitespace = " \t\r\n";

std::string_view trim(std::string_view text) {
    while (!text.empty() && whitespace.find(text.front()) != std::string_view::npos) {
        text.remove_prefix(1);
    }
    while (!text.empty() && whitespace.find(text.back()) != std::string_view::npos) {
        text.remove_suffix(1);
    }
    return text;
}

// An HTTP method token as core compares it: exact and case-sensitive, so
// "post" would never match a request and is rejected rather than ignored.
bool valid_method(std::string_view method) {
    if (method == "*") {
        return true;
    }
    if (method.empty()) {
        return false;
    }
    for (const char ch : method) {
        if (ch < 'A' || ch > 'Z') {
            return false;
        }
    }
    return true;
}

// One "METHOD /path" entry. Method "*" matches any method; a path ending in
// "*" matches by prefix. Everything else is an exact, case-sensitive match,
// the same comparison the core dispatcher uses to select a route.
struct RoutePattern {
    std::string method;
    std::string path;
    bool prefix = false;

    bool matches(const HttpRequest & request) const {
        if (method != "*" && method != request.method) {
            return false;
        }
        if (prefix) {
            return request.path.compare(0, path.size(), path) == 0;
        }
        return request.path == path;
    }
};

[[noreturn]] void bad_route(const char * env_name, std::string_view entry, std::string_view why) {
    throw std::runtime_error(
        std::string(env_name) + ": " + std::string(why) +
        " in '" + std::string(entry) + "'; entries are 'METHOD /path' separated by commas");
}

// Strict on purpose: a rule that parses but can never match is a deny rule
// that silently does nothing, so anything that is not exactly
// "METHOD /path" fails startup instead.
std::vector<RoutePattern> parse_routes(const char * env_name, std::string_view text) {
    std::vector<RoutePattern> routes;
    while (!text.empty()) {
        const auto comma = text.find(',');
        const auto entry = trim(text.substr(0, comma));
        text = comma == std::string_view::npos ? std::string_view() : text.substr(comma + 1);
        if (entry.empty()) {
            continue;
        }

        std::vector<std::string_view> tokens;
        for (auto rest = entry; !rest.empty();) {
            const auto start = rest.find_first_not_of(whitespace);
            if (start == std::string_view::npos) {
                break;
            }
            rest.remove_prefix(start);
            const auto end = rest.find_first_of(whitespace);
            tokens.push_back(rest.substr(0, end));
            rest = end == std::string_view::npos ? std::string_view() : rest.substr(end);
        }
        if (tokens.size() > 2) {
            bad_route(env_name, entry, "more than one route (missing comma?)");
        }
        if (tokens.size() < 2) {
            bad_route(env_name, entry, "expected a method and a path");
        }

        const auto method = tokens[0];
        auto path = tokens[1];
        if (!valid_method(method)) {
            bad_route(env_name, entry, "method must be '*' or an uppercase HTTP method");
        }
        if (path.front() != '/') {
            bad_route(env_name, entry, "path must start with '/'");
        }
        RoutePattern route;
        route.method = std::string(method);
        if (path.back() == '*') {
            path.remove_suffix(1);
            route.prefix = true;
        }
        if (path.find('*') != std::string_view::npos) {
            bad_route(env_name, entry, "'*' is only allowed at the end of a path");
        }
        route.path = std::string(path);
        routes.push_back(std::move(route));
    }
    return routes;
}

bool any_route_matches(const std::vector<RoutePattern> & routes, const HttpRequest & request) {
    for (const auto & route : routes) {
        if (route.matches(request)) {
            return true;
        }
    }
    return false;
}

// Running time depends only on the configured token length, never on where
// the presented token first differs or on its length.
bool tokens_equal(const std::string & expected, std::string_view presented) {
    volatile unsigned char diff = expected.size() == presented.size() ? 0 : 1;
    for (size_t i = 0; i < expected.size(); ++i) {
        const unsigned char other = presented.empty()
            ? 0
            : static_cast<unsigned char>(presented[i % presented.size()]);
        diff = static_cast<unsigned char>(diff | (static_cast<unsigned char>(expected[i]) ^ other));
    }
    return diff == 0;
}

std::optional<std::string_view> bearer_token(const HttpRequest & request) {
    const auto it = request.headers.find("authorization");
    if (it == request.headers.end()) {
        return std::nullopt;
    }
    const auto value = trim(it->second);
    constexpr std::string_view scheme = "bearer";
    if (value.size() <= scheme.size() ||
        frontends::lower_ascii(std::string(value.substr(0, scheme.size()))) != scheme ||
        (value[scheme.size()] != ' ' && value[scheme.size()] != '\t')) {
        return std::nullopt;
    }
    const auto token = trim(value.substr(scheme.size()));
    if (token.empty()) {
        return std::nullopt;
    }
    return token;
}

HttpResponse unauthorized_response(bool token_presented) {
    auto response = error_response(
        401,
        token_presented ? "invalid bearer token" : "missing bearer token",
        "authentication_error");
    response.headers["WWW-Authenticate"] = token_presented
        ? "Bearer realm=\"audiocpp_server\", error=\"invalid_token\""
        : "Bearer realm=\"audiocpp_server\"";
    return response;
}

class BearerAuthModule final : public ServerFrontendModule {
public:
    BearerAuthModule() {
        const char * token = std::getenv(token_env);
        if (token == nullptr || trim(token).empty()) {
            throw std::runtime_error(
                std::string("bearer_auth frontend requires ") + token_env + " to be set to a non-empty token");
        }
        token_ = std::string(trim(token));

        // Unset keeps the health check public; an empty value makes every
        // route require the token.
        const char * public_routes = std::getenv(public_routes_env);
        public_routes_ = parse_routes(
            public_routes_env, public_routes == nullptr ? default_public_routes : public_routes);
        if (const char * allow_routes = std::getenv(allow_routes_env)) {
            allow_routes_ = parse_routes(allow_routes_env, allow_routes);
            allowlist_enabled_ = true;
        }
        if (const char * deny_routes = std::getenv(deny_routes_env)) {
            deny_routes_ = parse_routes(deny_routes_env, deny_routes);
        }
    }

    std::string_view name() const override { return "bearer_auth"; }

    std::optional<FrontendPreContract> pre_contract() const override {
        return FrontendPreContract{
            frontend_contracts::any,
            frontend_contracts::any,
            frontend_contracts::any,
            frontend_contracts::any,
        };
    }

    void pre_process(ServerFrontendContext & context, ServerFrontendRequest & request) override {
        (void) context;
        if (any_route_matches(public_routes_, request.request)) {
            return;
        }

        const auto token = bearer_token(request.request);
        if (!token.has_value() || !tokens_equal(token_, *token)) {
            request.response = unauthorized_response(token.has_value());
            return;
        }

        if (any_route_matches(deny_routes_, request.request) ||
            (allowlist_enabled_ && !any_route_matches(allow_routes_, request.request))) {
            request.response = error_response(
                403,
                "route is not permitted by this server: " + request.request.method + " " + request.request.path,
                "permission_error");
        }
    }

private:
    std::string token_;
    std::vector<RoutePattern> public_routes_;
    std::vector<RoutePattern> allow_routes_;
    std::vector<RoutePattern> deny_routes_;
    bool allowlist_enabled_ = false;
};

std::unique_ptr<ServerFrontendModule> make_bearer_auth_module() {
    return std::make_unique<BearerAuthModule>();
}

} // namespace

void register_bearer_auth_module(ServerFrontendRegistry & registry) {
    registry.add(make_bearer_auth_module);
}

} // namespace minitts::server
