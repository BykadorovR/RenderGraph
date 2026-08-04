module Timestamps;

using namespace RenderGraph;

Timestamps::Timestamps(const Device& device, uint32_t maxFramesInFlight)
    : _device(&device),
      _frames(maxFramesInFlight) {
  if (_device->getQueueFamilyProperties(vkb::QueueType::graphics).timestampValidBits == 0)
    throw std::runtime_error("Graphics queue doesn't support timestamps");
  if (_device->getQueueFamilyProperties(vkb::QueueType::compute).timestampValidBits == 0)
    throw std::runtime_error("Compute queue doesn't support timestamps");

  _timestampPeriod = _device->getDevice().physical_device.properties.limits.timestampPeriod;
  const VkQueryPoolCreateInfo createInfo{
      .sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
      .pNext = nullptr,
      .flags = 0,
      .queryType = VK_QUERY_TYPE_TIMESTAMP,
      .queryCount = static_cast<uint32_t>(_queryMaxNumber),
      .pipelineStatistics = 0,
  };
  for (FrameData& frame : _frames) {
    const VkResult status = vkCreateQueryPool(_device->getLogicalDevice(), &createInfo, nullptr, &frame.queryPool);
    if (status != VK_SUCCESS) {
      throw std::runtime_error("Failed to create timestamps query pool");
    }
  }
}

void Timestamps::resetQueryPool() {
  // Reusing this frame slot already requires its previous GPU work to be complete.
  // Timestamp results are optional diagnostics, so an unavailable sample must not block or abort rendering.
  std::scoped_lock lock(_mutexPush, _mutexRequest);

  FrameData& frame = _frames[_currentFrame];
  if (frame.submitted && frame.timestampIndex > 0) {
    std::vector<uint64_t> buffer(static_cast<std::size_t>(frame.timestampIndex));
    const VkResult status = vkGetQueryPoolResults(
        _device->getLogicalDevice(), frame.queryPool, 0, static_cast<uint32_t>(frame.timestampIndex),
        buffer.size() * sizeof(uint64_t), buffer.data(), sizeof(uint64_t),
        VK_QUERY_RESULT_64_BIT);
    if (status != VK_SUCCESS && status != VK_NOT_READY) {
      throw std::runtime_error("vkGetQueryPoolResults failed: " + std::to_string(status));
    }

    if (status == VK_SUCCESS) {
      _timestampResults.clear();
      for (const auto& [name, range] : frame.timestampRanges) {
        _timestampResults[name] = {
            static_cast<double>(buffer[static_cast<std::size_t>(range.x)]) * _timestampPeriod,
            static_cast<double>(buffer[static_cast<std::size_t>(range.y)]) * _timestampPeriod,
        };
      }
    }
  }

  vkResetQueryPool(_device->getLogicalDevice(), frame.queryPool, 0, static_cast<uint32_t>(_queryMaxNumber));

  frame.timestampIndex = 0;
  frame.timestampRanges.clear();
  frame.submitted = false;
}

void Timestamps::pushTimestamp(std::string_view name, const CommandBuffer& commandBuffer) {
  std::scoped_lock lock(_mutexPush);

  FrameData& frame = _frames[_currentFrame];
  if (frame.timestampIndex >= _queryMaxNumber) {
    throw std::runtime_error("More timestamps requested than allocated");
  }

  const int currentIndex = frame.timestampIndex++;
  vkCmdWriteTimestamp2(commandBuffer.getCommandBuffer(), VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, frame.queryPool,
                       static_cast<uint32_t>(currentIndex));
  frame.timestampRanges[std::string(name)] = {
      currentIndex,
      currentIndex,
  };
}

void Timestamps::popTimestamp(std::string_view name, const CommandBuffer& commandBuffer) {
  std::scoped_lock lock(_mutexPush);

  FrameData& frame = _frames[_currentFrame];
  if (frame.timestampIndex >= _queryMaxNumber) {
    throw std::runtime_error("More timestamps requested than allocated");
  }
  const auto rangeIt = frame.timestampRanges.find(std::string(name));
  if (rangeIt == frame.timestampRanges.end()) {
    throw std::runtime_error("Timestamp range was not started: " + std::string(name));
  }

  const int currentIndex = frame.timestampIndex++;
  vkCmdWriteTimestamp2(commandBuffer.getCommandBuffer(), VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, frame.queryPool,
                       static_cast<uint32_t>(currentIndex));
  rangeIt->second.y = currentIndex;
}

void Timestamps::finishFrame() {
  // don't request anything from Vulkan and don't block CPU
  std::scoped_lock lock(_mutexPush);

  FrameData& frame = _frames[_currentFrame];
  frame.submitted = true;
  _currentFrame = (_currentFrame + 1) % static_cast<uint32_t>(_frames.size());
}

std::unordered_map<std::string, glm::dvec2> Timestamps::getTimestamps() {
  std::scoped_lock lock(_mutexRequest);

  return _timestampResults;
}

Timestamps::~Timestamps() {
  if (_device == nullptr) {
    return;
  }

  for (FrameData& frame : _frames) {
    if (frame.queryPool == nullptr) {
      continue;
    }
    vkDestroyQueryPool(_device->getLogicalDevice(), frame.queryPool, nullptr);
    frame.queryPool = nullptr;
  }
}
