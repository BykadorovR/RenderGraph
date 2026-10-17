module;

#include <volk.h>
#include <tracy/Tracy.hpp>
#include <tracy/TracyVulkan.hpp>
#include <source_location>
#include <optional>
#include <string>
#include <unordered_map>

export module Instrumentation;

import Command;
import Device;

namespace RenderGraph {
class MarkerCPU {
 private:
  // tracy::ScopedZone doesn't have a default constructor.
  std::optional<tracy::ScopedZone> _zone;

 public:
  MarkerCPU(std::string name, std::source_location location);
  MarkerCPU(const MarkerCPU&) = delete;
  MarkerCPU(MarkerCPU&&) = delete;
  MarkerCPU& operator=(const MarkerCPU&) = delete;
  MarkerCPU& operator=(MarkerCPU&&) = delete;
  ~MarkerCPU();
};

class MarkerGPU {
 private:
  // tracy::VkCtxScope doesn't have a default constructor.
  std::optional<tracy::VkCtxScope> _zone;
  VkCommandBuffer _commandBuffer = VK_NULL_HANDLE;
  bool _debugLabelActive = false;

 public:
  MarkerGPU(std::string name, const CommandBuffer& commandBuffer, TracyVkCtx context, std::source_location location);
  MarkerGPU(const MarkerGPU&) = delete;
  MarkerGPU(MarkerGPU&&) = delete;
  MarkerGPU& operator=(const MarkerGPU&) = delete;
  MarkerGPU& operator=(MarkerGPU&&) = delete;
  ~MarkerGPU();
};

export class Tracer final {
 private:
  std::unordered_map<QueueType, TracyVkCtx> _contexts;
  VkDevice _device = VK_NULL_HANDLE;

 public:
  explicit Tracer(const Device& device);
  Tracer(const Tracer&) = delete;
  Tracer(Tracer&&) = delete;
  Tracer& operator=(const Tracer&) = delete;
  Tracer& operator=(Tracer&&) = delete;

  void setObjectName(VkObjectType type, uint64_t handle, std::string name);
  // Guaranteed copy elision (C++17), accessible to importers via auto.
  [[nodiscard]] MarkerCPU traceCPU(std::string name,
                                   std::source_location location = std::source_location::current());
  [[nodiscard]] MarkerGPU traceGPU(std::string name,
                                   QueueType type,
                                   const CommandBuffer& commandBuffer,
                                   std::source_location location = std::source_location::current());
  void markFrame();
  // Call once per frame for every distinct queue context from the frame owner.
  void collect(QueueType type, const CommandBuffer& commandBuffer);
  ~Tracer();
};
}  // namespace RenderGraph
