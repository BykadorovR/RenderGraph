module;

#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <volk.h>

export module Timestamps;

import Device;
import Command;
import glm;

export namespace RenderGraph {

class Timestamps {
 private:
  struct FrameData {
    VkQueryPool queryPool = nullptr;
    std::unordered_map<std::string, glm::ivec2> timestampRanges;
    int timestampIndex = 0;
    bool submitted = false;
  };

  const Device* _device;
  double _timestampPeriod;
  int _queryMaxNumber = 30;
  std::vector<FrameData> _frames;
  uint32_t _currentFrame = 0;
  std::unordered_map<std::string, glm::dvec2> _timestampResults;
  std::mutex _mutexPush;
  std::mutex _mutexRequest;

 public:
  Timestamps(const Device& device, uint32_t maxFramesInFlight);
  void resetQueryPool();
  void pushTimestamp(std::string_view name, const CommandBuffer& commandBuffer);
  void popTimestamp(std::string_view name, const CommandBuffer& commandBuffer);
  void finishFrame();
  // Return copy, otherwise race condition between calling code
  // and resetQueryPool().
  std::unordered_map<std::string, glm::dvec2> getTimestamps();

  ~Timestamps();
};

}  // namespace RenderGraph
