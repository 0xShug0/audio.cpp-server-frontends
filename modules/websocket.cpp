#include "frontend.h"

#include "engine/framework/io/json.h"

#include "httplib.h"

#include <atomic>
#include <chrono>
#include <cctype>
#include <exception>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>

namespace minitts::server {
namespace {

std::string lower_ascii(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    for (const unsigned char ch : text) {
        out.push_back(static_cast<char>(std::tolower(ch)));
    }
    return out;
}

std::string raw_query_from_target(const httplib::Request & request) {
    const auto marker = request.target.find('?');
    if (marker == std::string::npos) {
        return {};
    }
    return request.target.substr(marker + 1);
}

HttpRequest to_http_request(const httplib::Request & request) {
    HttpRequest out;
    out.method = request.method;
    out.path = request.path;
    out.query = raw_query_from_target(request);
    out.body = request.body;
    for (const auto & header : request.headers) {
        out.headers[lower_ascii(header.first)] = header.second;
    }
    return out;
}

class HttplibStreamWriter final : public HttpStreamWriter {
public:
    explicit HttplibStreamWriter(httplib::DataSink & sink)
        : sink_(sink) {}

    void write(std::string_view data) override {
        if (data.empty()) {
            return;
        }
        if (!sink_.write(data.data(), data.size())) {
            throw std::runtime_error("WebSocket frontend HTTP client disconnected while writing streamed response");
        }
    }

private:
    httplib::DataSink & sink_;
};

void copy_response_headers(const HttpResponse & source, httplib::Response & target) {
    for (const auto & [key, value] : source.headers) {
        const auto normalized = lower_ascii(key);
        if (normalized == "content-length" ||
            normalized == "transfer-encoding" ||
            normalized == "content-type") {
            continue;
        }
        target.set_header(key, value);
    }
}

std::string stream_error_event(const std::exception & ex) {
    return
        "data: {\"type\":\"error\",\"error\":{\"message\":" +
        engine::io::json::stringify_string(ex.what()) +
        "}}\n\n";
}

void send_response(HttpResponse response, httplib::Response & target) {
    target.status = response.status;
    copy_response_headers(response, target);
    if (!response.stream_body) {
        target.set_content(response.body, response.content_type);
        return;
    }

    auto stream_body = std::make_shared<std::function<void(HttpStreamWriter &)>>(
        std::move(response.stream_body));
    auto sent = std::make_shared<std::atomic_bool>(false);
    const auto content_type = response.content_type;
    target.set_chunked_content_provider(
        content_type,
        [stream_body, sent, content_type](size_t, httplib::DataSink & sink) {
            if (sent->exchange(true)) {
                return false;
            }
            HttplibStreamWriter writer(sink);
            try {
                (*stream_body)(writer);
            } catch (const std::exception & ex) {
                if (content_type.rfind("text/event-stream", 0) == 0) {
                    writer.write(stream_error_event(ex));
                } else {
                    std::cerr << "audiocpp_server WebSocket frontend HTTP streaming response failed: " << ex.what() << "\n";
                }
            }
            sink.done();
            return true;
        });
}

void handle_request(
    const httplib::Request & request,
    httplib::Response & response,
    IHttpHandler & handler,
    uint64_t max_request_body_bytes) {
    if (request.body.size() > max_request_body_bytes) {
        send_response(
            error_response(413, "request body exceeds max_request_body_bytes", "request_too_large"),
            response);
        return;
    }
    send_response(handler.handle(to_http_request(request)), response);
}

bool has_websocket_envelope_field(const engine::io::json::Value & root) {
    return root.find("method") != nullptr ||
        root.find("query") != nullptr ||
        root.find("headers") != nullptr ||
        root.find("body") != nullptr ||
        root.find("body_json") != nullptr;
}

HttpRequest request_from_ws_message(const httplib::Request & upgrade_request, const std::string & message) {
    HttpRequest out = to_http_request(upgrade_request);
    out.method = "POST";
    out.body.clear();
    out.body_stream = nullptr;
    out.headers.erase("connection");
    out.headers.erase("upgrade");
    out.headers.erase("sec-websocket-key");
    out.headers.erase("sec-websocket-version");
    out.headers.erase("sec-websocket-protocol");
    out.headers.erase("sec-websocket-extensions");

    const auto root = engine::io::json::parse(message);
    if (!root.is_object()) {
        throw std::runtime_error("WebSocket request must be a JSON object");
    }
    if (root.find("path") != nullptr) {
        throw std::runtime_error("WebSocket request path is selected by the upgrade URL, not by a frame field");
    }

    if (const auto * headers = root.find("headers")) {
        if (!headers->is_object()) {
            throw std::runtime_error("WebSocket request field 'headers' must be an object");
        }
        for (const auto & [key, value] : headers->as_object()) {
            if (!value.is_string()) {
                throw std::runtime_error("WebSocket request headers must be string values");
            }
            out.headers[lower_ascii(key)] = value.as_string();
        }
    }

    if (const auto * method = root.find("method")) {
        if (!method->is_string()) {
            throw std::runtime_error("WebSocket request field 'method' must be a string");
        }
        out.method = method->as_string();
    }
    if (const auto * query = root.find("query")) {
        if (!query->is_string()) {
            throw std::runtime_error("WebSocket request field 'query' must be a string");
        }
        out.query = query->as_string();
    }

    if (const auto * body = root.find("body")) {
        if (!body->is_string()) {
            throw std::runtime_error("WebSocket request field 'body' must be a string");
        }
        out.body = body->as_string();
    } else if (const auto * body_json = root.find("body_json")) {
        out.body = engine::io::json::stringify(*body_json);
        out.headers["content-type"] = "application/json";
    } else if (!has_websocket_envelope_field(root)) {
        out.body = message;
        out.headers["content-type"] = "application/json";
    }
    if (!out.body.empty() && out.headers.find("content-length") == out.headers.end()) {
        out.headers["content-length"] = std::to_string(out.body.size());
    }
    return out;
}

std::string headers_json(const std::unordered_map<std::string, std::string> & headers) {
    std::ostringstream out;
    out << "{";
    bool first = true;
    for (const auto & [key, value] : headers) {
        if (!first) {
            out << ",";
        }
        first = false;
        out << engine::io::json::stringify_string(key)
            << ":"
            << engine::io::json::stringify_string(value);
    }
    out << "}";
    return out.str();
}

std::string ws_message_id_json(const std::string & request_message) {
    const auto root = engine::io::json::parse(request_message);
    if (!root.is_object()) {
        return "null";
    }
    const auto * id = root.find("id");
    if (id == nullptr || id->is_null()) {
        return "null";
    }
    if (id->is_string() || id->is_number()) {
        return engine::io::json::stringify(*id);
    }
    return "null";
}

bool send_ws(httplib::ws::WebSocket & ws, const std::string & payload) {
    return ws.send(payload);
}

std::string response_json(std::string_view id_json, const HttpResponse & response) {
    return
        "{\"type\":\"response\",\"id\":" + std::string(id_json) +
        ",\"status\":" + std::to_string(response.status) +
        ",\"content_type\":" + engine::io::json::stringify_string(response.content_type) +
        ",\"headers\":" + headers_json(response.headers) +
        ",\"body\":" + engine::io::json::stringify_string(response.body) +
        "}";
}

std::string response_start_json(std::string_view id_json, const HttpResponse & response) {
    return
        "{\"type\":\"response.start\",\"id\":" + std::string(id_json) +
        ",\"status\":" + std::to_string(response.status) +
        ",\"content_type\":" + engine::io::json::stringify_string(response.content_type) +
        ",\"headers\":" + headers_json(response.headers) +
        "}";
}

std::string response_body_json(std::string_view id_json, std::string_view body) {
    return
        "{\"type\":\"response.body\",\"id\":" + std::string(id_json) +
        ",\"body\":" + engine::io::json::stringify_string(body) +
        "}";
}

std::string response_done_json(std::string_view id_json) {
    return "{\"type\":\"response.done\",\"id\":" + std::string(id_json) + "}";
}

std::string error_json(std::string_view id_json, const std::exception & ex) {
    return
        "{\"type\":\"error\",\"id\":" + std::string(id_json) +
        ",\"error\":{\"message\":" + engine::io::json::stringify_string(ex.what()) +
        ",\"type\":\"server_error\"}}";
}

class WebSocketStreamWriter final : public HttpStreamWriter {
public:
    WebSocketStreamWriter(httplib::ws::WebSocket & ws, std::string id_json)
        : ws_(ws),
          id_json_(std::move(id_json)) {}

    void write(std::string_view data) override {
        if (data.empty()) {
            return;
        }
        if (!send_ws(ws_, response_body_json(id_json_, data))) {
            throw std::runtime_error("WebSocket client disconnected while writing streamed response");
        }
    }

private:
    httplib::ws::WebSocket & ws_;
    std::string id_json_;
};

void handle_ws_message(
    httplib::ws::WebSocket & ws,
    IHttpHandler & handler,
    uint64_t max_request_body_bytes,
    const httplib::Request & upgrade_request,
    const std::string & message) {
    std::string id_json = "null";
    try {
        id_json = ws_message_id_json(message);
        auto request = request_from_ws_message(upgrade_request, message);
        if (request.body.size() > max_request_body_bytes) {
            auto response = error_response(413, "request body exceeds max_request_body_bytes", "request_too_large");
            send_ws(ws, response_json(id_json, response));
            return;
        }
        HttpResponse response;
        try {
            response = handler.handle(request);
        } catch (const std::exception & ex) {
            response = error_response(500, ex.what(), "server_error");
        }
        if (!response.stream_body) {
            send_ws(ws, response_json(id_json, response));
            return;
        }

        send_ws(ws, response_start_json(id_json, response));
        WebSocketStreamWriter writer(ws, id_json);
        response.stream_body(writer);
        send_ws(ws, response_done_json(id_json));
    } catch (const std::exception & ex) {
        send_ws(ws, error_json(id_json, ex));
    }
}

class WebSocketFrontendListener final : public ServerFrontendListener {
public:
    std::string_view name() const override {
        return "websocket";
    }

    void serve(
        const std::string & host,
        int port,
        IHttpHandler & handler,
        ShutdownRequested shutdown_requested,
        uint64_t max_request_body_bytes,
        const ServerFrontendOptions & options) override {
        (void) options;
        httplib::Server server;
        server.set_payload_max_length(static_cast<size_t>(max_request_body_bytes));
        server.set_idle_interval(std::chrono::milliseconds(250));
        server.set_websocket_ping_interval(15);

        server.WebSocket(
            R"(.*)",
            [&handler, max_request_body_bytes](const httplib::Request & upgrade_request, httplib::ws::WebSocket & ws) {
                std::string message;
                while (ws.is_open()) {
                    const auto read = ws.read(message);
                    if (read == httplib::ws::Fail) {
                        break;
                    }
                    if (read == httplib::ws::Binary) {
                        send_ws(
                            ws,
                            "{\"type\":\"error\",\"id\":null,\"error\":{\"message\":\"binary WebSocket messages are not supported\",\"type\":\"invalid_request_error\"}}");
                        continue;
                    }
                    handle_ws_message(ws, handler, max_request_body_bytes, upgrade_request, message);
                }
            });

        const auto route = [&handler, max_request_body_bytes](const httplib::Request & request, httplib::Response & response) {
            try {
                handle_request(request, response, handler, max_request_body_bytes);
            } catch (const std::exception & ex) {
                send_response(error_response(500, ex.what(), "server_error"), response);
            }
        };
        server.Get(R"(.*)", route);
        server.Post(R"(.*)", route);
        server.Put(R"(.*)", route);
        server.Patch(R"(.*)", route);
        server.Delete(R"(.*)", route);
        server.Options(R"(.*)", route);

        if (!server.bind_to_port(host, port)) {
            throw std::runtime_error("could not bind WebSocket frontend server on " + host + ":" + std::to_string(port));
        }

        std::thread server_thread([&server] {
            if (!server.listen_after_bind()) {
                std::cerr << "audiocpp_server WebSocket frontend listener stopped before accepting requests\n";
            }
        });
        std::cout << "audiocpp_server WebSocket frontend listening on ws://" << host << ":" << port << "/<endpoint>" << "\n";
        while (!shutdown_requested()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
        }
        server.stop();
        if (server_thread.joinable()) {
            server_thread.join();
        }
        std::cout << "audiocpp_server stopped\n";
    }
};

std::unique_ptr<ServerFrontendListener> make_websocket_listener() {
    return std::make_unique<WebSocketFrontendListener>();
}

} // namespace

void register_websocket_listener(ServerFrontendRegistry & registry) {
    registry.add_listener("websocket", make_websocket_listener);
}

} // namespace minitts::server
