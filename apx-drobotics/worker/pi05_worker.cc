// Pi05 LIBERO inference worker for the D-Robotics S600.
//
// The S600 runtime is a C++ library, while the network service is Python. This
// process sits between them: it loads the model once through the official
// OellmRuntime API, then answers one request at a time on stdin/stdout. The
// Python side (apx_drobotics.runtime) owns the contract checks and all error reporting.
//
// Requests arrive on stdin. Replies go to the file descriptor named by
// --reply-fd, never to stdout: the SDK libraries print to stdout as soon as they
// are loaded, before main() runs, so stdout cannot carry a binary protocol.
// stdout and stderr are left to the runtime's logs.
//
// Every frame, in both directions, is:
//   uint32 little-endian JSON length, JSON header,
//   uint32 little-endian payload length, payload bytes.
//
// Worker -> front end, once at start-up:
//   {"event": "ready", "oellm_version": "2.0.0"}            payload: empty
//   {"event": "init_failed", "error_code": N}                payload: empty
// Front end -> worker, per request:
//   {"op": "infer", "prompt": "...", "state": [8 doubles],
//    "images": [{"width": 224, "height": 224}, ...]}         payload: CHW uint8 RGB images
// Worker -> front end, per request:
//   {"ok": true, "action_shape": [1, 10, 7], "timing_ms": {...}}  payload: fp64 actions
//   {"ok": false, "error": "...", "error_code": N}                 payload: empty

#include <fcntl.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "oellm_runtime_basic/oellm_runtime.h"
#include "oellm_runtime_basic/oellm_runtime_error_code.h"
#include "oellm_runtime_basic/oellm_runtime_version.h"
#include "vla/oellm_vla_types.h"

namespace {

using Json = nlohmann::json;

constexpr uint32_t kMaxJsonBytes = 1u << 20;
constexpr uint32_t kMaxPayloadBytes = 64u << 20;

bool ReadExact(int fd, void* buffer, size_t size) {
  auto* out = static_cast<char*>(buffer);
  size_t done = 0;
  while (done < size) {
    ssize_t got = ::read(fd, out + done, size - done);
    if (got == 0) {
      return false;
    }
    if (got < 0) {
      if (errno == EINTR) {
        continue;
      }
      return false;
    }
    done += static_cast<size_t>(got);
  }
  return true;
}

bool WriteExact(int fd, void const* buffer, size_t size) {
  auto const* in = static_cast<char const*>(buffer);
  size_t done = 0;
  while (done < size) {
    ssize_t put = ::write(fd, in + done, size - done);
    if (put < 0) {
      if (errno == EINTR) {
        continue;
      }
      return false;
    }
    done += static_cast<size_t>(put);
  }
  return true;
}

bool ReadU32(int fd, uint32_t& value) {
  unsigned char bytes[4];
  if (!ReadExact(fd, bytes, sizeof(bytes))) {
    return false;
  }
  value = static_cast<uint32_t>(bytes[0]) | (static_cast<uint32_t>(bytes[1]) << 8) |
          (static_cast<uint32_t>(bytes[2]) << 16) | (static_cast<uint32_t>(bytes[3]) << 24);
  return true;
}

bool WriteU32(int fd, uint32_t value) {
  unsigned char bytes[4] = {
      static_cast<unsigned char>(value & 0xff),
      static_cast<unsigned char>((value >> 8) & 0xff),
      static_cast<unsigned char>((value >> 16) & 0xff),
      static_cast<unsigned char>((value >> 24) & 0xff)};
  return WriteExact(fd, bytes, sizeof(bytes));
}

// Returns false on end of input or a malformed frame; the caller exits.
bool ReadFrame(int fd, Json& header, std::string& payload) {
  uint32_t json_size = 0;
  if (!ReadU32(fd, json_size) || json_size > kMaxJsonBytes) {
    return false;
  }
  std::string text(json_size, '\0');
  if (!ReadExact(fd, text.data(), json_size)) {
    return false;
  }
  uint32_t payload_size = 0;
  if (!ReadU32(fd, payload_size) || payload_size > kMaxPayloadBytes) {
    return false;
  }
  payload.assign(payload_size, '\0');
  if (!ReadExact(fd, payload.data(), payload_size)) {
    return false;
  }
  header = Json::parse(text, nullptr, false);
  return !header.is_discarded() && header.is_object();
}

bool WriteFrame(int fd, Json const& header, void const* payload, size_t payload_size) {
  std::string text = header.dump();
  return WriteU32(fd, static_cast<uint32_t>(text.size())) &&
         WriteExact(fd, text.data(), text.size()) &&
         WriteU32(fd, static_cast<uint32_t>(payload_size)) &&
         (payload_size == 0 || WriteExact(fd, payload, payload_size));
}

bool WriteError(int fd, std::string const& message, int error_code) {
  return WriteFrame(fd, Json{{"ok", false}, {"error", message}, {"error_code", error_code}},
                    nullptr, 0);
}

// Builds the request exactly as the official demo's raw-image path does.
bool BuildRequest(Json const& header, std::string const& payload,
                  oellm::OellmVlaRequestData& request, std::string& error) {
  if (!header.contains("prompt") || !header["prompt"].is_string() ||
      !header.contains("state") || !header["state"].is_array() ||
      !header.contains("images") || !header["images"].is_array()) {
    error = "request header needs prompt, state and images";
    return false;
  }
  size_t offset = 0;
  for (auto const& image : header["images"]) {
    if (!image.contains("width") || !image.contains("height")) {
      error = "each image needs width and height";
      return false;
    }
    int32_t const width = image["width"].get<int32_t>();
    int32_t const height = image["height"].get<int32_t>();
    if (width <= 0 || height <= 0) {
      error = "image width and height must be positive";
      return false;
    }
    size_t const bytes = static_cast<size_t>(3) * width * height;
    if (offset + bytes > payload.size()) {
      error = "payload is shorter than the images it declares";
      return false;
    }
    oellm::OellmVlaImageInput input;
    input.width = width;
    input.height = height;
    input.layout = oellm::ImageLayout::kChw;
    input.type = oellm::ImageType::kRgb;
    input.dtype = oellm::VlaImageDtype::kUint8;
    input.data.assign(payload.begin() + static_cast<std::ptrdiff_t>(offset),
                      payload.begin() + static_cast<std::ptrdiff_t>(offset + bytes));
    request.images.push_back(std::move(input));
    offset += bytes;
  }
  if (offset != payload.size()) {
    error = "payload is longer than the images it declares";
    return false;
  }
  request.prompt = header["prompt"].get<std::string>();
  request.token_ids.reset();
  request.state = header["state"].get<std::vector<double>>();
  request.state_raw = true;
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  std::string config_path;
  int protocol_out = -1;
  for (int i = 1; i < argc; ++i) {
    std::string const arg = argv[i];
    if (arg == "--config" && i + 1 < argc) {
      config_path = argv[++i];
    } else if (arg == "--reply-fd" && i + 1 < argc) {
      protocol_out = std::atoi(argv[++i]);
    } else {
      config_path.clear();
      break;
    }
  }
  if (config_path.empty() || protocol_out <= STDERR_FILENO || ::fcntl(protocol_out, F_GETFD) < 0) {
    std::cerr << "usage: " << argv[0] << " --config <model_config.json> --reply-fd <fd>"
              << std::endl;
    return 64;
  }

  oellm::OellmRuntime runtime;
  oellm::OellmErrorCode const init = runtime.Init(config_path);
  if (init != oellm::OellmErrorCode::kOk) {
    WriteFrame(protocol_out, Json{{"event", "init_failed"}, {"error_code", static_cast<int>(init)}},
               nullptr, 0);
    return 2;
  }
  if (!WriteFrame(protocol_out,
                  Json{{"event", "ready"}, {"oellm_version", OELLM_VERSION_STRING}}, nullptr, 0)) {
    return 3;
  }

  int32_t request_id = 0;
  Json header;
  std::string payload;
  while (ReadFrame(STDIN_FILENO, header, payload)) {
    if (header.value("op", "") != "infer") {
      if (!WriteError(protocol_out, "unknown op", -1)) {
        return 3;
      }
      continue;
    }
    auto vla_request = std::make_shared<oellm::OellmVlaRequestData>();
    std::string error;
    bool built = false;
    try {
      built = BuildRequest(header, payload, *vla_request, error);
    } catch (Json::exception const& exception) {
      error = std::string("malformed request header: ") + exception.what();
    }
    if (!built) {
      if (!WriteError(protocol_out, error, -1)) {
        return 3;
      }
      continue;
    }

    oellm::OellmRequest request;
    request.oellm_requests.push_back({});
    request.oellm_requests[0].request_id = request_id++;
    request.oellm_requests[0].data =
        std::vector<std::shared_ptr<oellm::OellmRequestData>>{vla_request};
    oellm::OellmResponse response;
    auto const start = std::chrono::steady_clock::now();
    oellm::OellmErrorCode const status = runtime.Infer(request, response);
    double const wall_ms = std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - start)
                               .count();
    if (status != oellm::OellmErrorCode::kOk || response.response_datas.empty()) {
      if (!WriteError(protocol_out, "runtime inference failed", static_cast<int>(status))) {
        return 3;
      }
      continue;
    }
    auto const vla_response =
        std::dynamic_pointer_cast<oellm::OellmVlaResponseData>(response.response_datas[0]);
    if (vla_response == nullptr || vla_response->action_fp64.empty()) {
      if (!WriteError(protocol_out, "runtime returned no fp64 actions", -1)) {
        return 3;
      }
      continue;
    }

    Json timing{{"runtime_wall", wall_ms}};
    oellm::OellmMetric metric;
    if (runtime.GetOellmInferMetric(metric) == oellm::OellmErrorCode::kOk &&
        !metric.metric_datas.empty()) {
      auto const& data = metric.metric_datas[0];
      timing["total"] = data->e2e;
      auto const vla = std::dynamic_pointer_cast<oellm::VlaMetric>(data);
      if (vla != nullptr) {
        timing["preproc"] = vla->preproc_cost;
        timing["visual"] = vla->visual_cost;
        timing["lm"] = vla->lm_cost;
        timing["action"] = vla->action_cost;
        timing["postproc"] = vla->postproc_cost;
      }
    }
    Json const reply{{"ok", true},
                     {"action_shape", vla_response->action_shape},
                     {"timing_ms", timing}};
    if (!WriteFrame(protocol_out, reply, vla_response->action_fp64.data(),
                    vla_response->action_fp64.size() * sizeof(double))) {
      return 3;
    }
  }
  return 0;
}
