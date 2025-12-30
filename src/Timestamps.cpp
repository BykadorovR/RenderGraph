module Timestamps;
using namespace RenderGraph;

Timestamps::Timestamps(const Device& device) : _device(&device) {
  if (_device->getQueueFamilyProperties(vkb::QueueType::graphics).timestampValidBits == 0)
    throw std::runtime_error("Graphics queue doesn't support timestamps");
  if (_device->getQueueFamilyProperties(vkb::QueueType::compute).timestampValidBits == 0)
    throw std::runtime_error("Compute queue doesn't support timestamps");

  _timestampPeriod = _device->getDeviceProperties().limits.timestampPeriod;
  VkQueryPoolCreateInfo createInfo{};
  createInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
  createInfo.flags = 0;
  createInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
  createInfo.queryCount = _queryMaxNumber;

  auto sts = vkCreateQueryPool(_device->getLogicalDevice(), &createInfo, nullptr, &_queryPool);
  if (sts != VK_SUCCESS) throw std::runtime_error("Failed to create timestamps query pool");
}

void Timestamps::resetQueryPool() noexcept {
  vkResetQueryPool(_device->getLogicalDevice(), _queryPool, 0, _queryMaxNumber);
}

void Timestamps::pushTimestamp(std::string_view name, const CommandBuffer& commandBuffer) {
  std::unique_lock<std::mutex> lock(_mutexPush);
  int currentIndex = _timestampIndex;
  vkCmdWriteTimestamp(commandBuffer.getCommandBuffer(), VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, _queryPool, currentIndex);
  _timestampRanges[std::string(name)].x = currentIndex;
  _timestampIndex++;
}

void Timestamps::popTimestamp(std::string_view name, const CommandBuffer& commandBuffer) {
  std::unique_lock<std::mutex> lock(_mutexPush);
  int currentIndex = _timestampIndex;
  vkCmdWriteTimestamp(commandBuffer.getCommandBuffer(), VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, _queryPool, currentIndex);
  _timestampRanges.at(std::string(name)).y = currentIndex;
  _timestampIndex++;
}

void Timestamps::fetchTimestamps() {
  if (_timestampIndex >= _queryMaxNumber) throw std::runtime_error("More timestamps requested than allocated");
  std::unique_lock<std::mutex> lock(_mutexRequest);
  _timestampResults.clear();

  std::vector<uint64_t> buffer(_timestampIndex);
  auto sts = vkGetQueryPoolResults(_device->getLogicalDevice(), _queryPool, 0, _timestampIndex,
                                   buffer.size() * sizeof(uint64_t), buffer.data(), sizeof(uint64_t),
                                   VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
  if (sts == VK_SUCCESS) {
    for (auto [key, value] : _timestampRanges) {
      _timestampResults[key] = {buffer[value.x] * _timestampPeriod, buffer[value.y] * _timestampPeriod};
    }
  }

  _timestampIndex = 0;
  _timestampRanges.clear();
}

std::unordered_map<std::string, glm::dvec2> Timestamps::getTimestamps() {
  std::unique_lock<std::mutex> lock(_mutexRequest);
  return _timestampResults;
}

Timestamps::~Timestamps() { vkDestroyQueryPool(_device->getLogicalDevice(), _queryPool, nullptr); }