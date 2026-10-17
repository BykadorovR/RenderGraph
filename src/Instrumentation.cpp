module;
#include <volk.h>
#include <tracy/TracyVulkan.hpp>

#include <cstring>
#include <source_location>
#include <stdexcept>
#include <utility>

module Instrumentation;

import CommandPool;

using namespace RenderGraph;

MarkerCPU::MarkerCPU(std::string name, std::source_location location) {
  _zone.emplace(location.line(), location.file_name(), std::strlen(location.file_name()), location.function_name(),
                std::strlen(location.function_name()), name.data(), name.size(), 0, -1, true);
}

MarkerCPU::~MarkerCPU() = default;

MarkerGPU::MarkerGPU(std::string name,
                     const CommandBuffer& commandBuffer,
                     TracyVkCtx context,
                     std::source_location location) {
  if (!commandBuffer.getActive()) {
    throw std::logic_error("GPU marker requires an active command buffer");
  }

  _commandBuffer = commandBuffer.getCommandBuffer();
  if (vkCmdBeginDebugUtilsLabelEXT != nullptr) {
    VkDebugUtilsLabelEXT label{
        .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT,
        .pLabelName = name.c_str(),
    };
    vkCmdBeginDebugUtilsLabelEXT(_commandBuffer, &label);
    _debugLabelActive = true;
  }

  _zone.emplace(context, location.line(), location.file_name(), std::strlen(location.file_name()),
                location.function_name(), std::strlen(location.function_name()), name.data(), name.size(),
                _commandBuffer, true);
}

MarkerGPU::~MarkerGPU() {
  if (_debugLabelActive) {
    vkCmdEndDebugUtilsLabelEXT(_commandBuffer);
  }
}

Tracer::Tracer(const Device& device) : _device(device.getLogicalDevice()) {
  auto createContext = [&](QueueType type) -> TracyVkCtx {
    if (device.getQueueFamilyProperties(type).timestampValidBits == 0) {
      throw std::runtime_error("Queue family does not support GPU timestamps");
    }

    CommandPool commandPool(type, device);
    CommandBuffer commandBuffer(commandPool, device);
    return TracyVkContext(device.getPhysicalDevice(), _device, device.getQueue(type), commandBuffer.getCommandBuffer());
  };

  TracyVkCtx graphicsContext = createContext(QueueType::GRAPHICS);
  TracyVkContextName(graphicsContext, "Graphics", 8);

  TracyVkCtx computeContext = nullptr;
  try {
    const VkQueue graphicsQueue = device.getQueue(QueueType::GRAPHICS);
    const VkQueue computeQueue = device.getQueue(QueueType::COMPUTE);
    if (graphicsQueue == computeQueue) {
      computeContext = graphicsContext;
    } else {
      computeContext = createContext(QueueType::COMPUTE);
      TracyVkContextName(computeContext, "Compute", 7);
    }
  } catch (...) {
    TracyVkDestroy(graphicsContext);
    throw;
  }

  _contexts.emplace(QueueType::GRAPHICS, graphicsContext);
  _contexts.emplace(QueueType::COMPUTE, computeContext);
}

void Tracer::setObjectName(VkObjectType type, uint64_t handle, std::string name) {
  if (handle == 0 || vkSetDebugUtilsObjectNameEXT == nullptr) {
    return;
  }

  const VkDebugUtilsObjectNameInfoEXT info{
      .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT,
      .objectType = type,
      .objectHandle = handle,
      .pObjectName = name.c_str(),
  };

  if (vkSetDebugUtilsObjectNameEXT(_device, &info) != VK_SUCCESS) {
    throw std::runtime_error("Failed to set Vulkan object name");
  }
}

[[nodiscard]] MarkerCPU Tracer::traceCPU(std::string name, std::source_location location) {
  return MarkerCPU(std::move(name), location);
}

[[nodiscard]] MarkerGPU Tracer::traceGPU(std::string name,
                                         QueueType type,
                                         const CommandBuffer& commandBuffer,
                                         std::source_location location) {
  return MarkerGPU(std::move(name), commandBuffer, _contexts.at(type), location);
}

void Tracer::markFrame() { FrameMark; }

void Tracer::collect(QueueType type, const CommandBuffer& commandBuffer) {
  if (!commandBuffer.getActive()) {
    throw std::logic_error("TracyVkCollect requires a recording command buffer");
  }

  TracyVkCollect(_contexts.at(type), commandBuffer.getCommandBuffer());
}

Tracer::~Tracer() {
  const auto graphicsContext = _contexts.at(QueueType::GRAPHICS);
  const auto computeContext = _contexts.at(QueueType::COMPUTE);

  if (computeContext != graphicsContext) {
    TracyVkDestroy(computeContext);
  }

  TracyVkDestroy(graphicsContext);
}
