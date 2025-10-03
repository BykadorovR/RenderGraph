export module Timestamps;
import Device;
import Command;
import glm;
import <volk.h>;
import <map>;
import <string>;
import <mutex>;

export namespace RenderGraph {
class Timestamps {
 private:
  const Device* _device;
  double _timestampPeriod;
  int _queryMaxNumber = 30;
  VkQueryPool _queryPool;
  std::map<std::string, glm::ivec2> _timestampRanges;
  std::map<std::string, glm::dvec2> _timestampResults;
  int _timestampIndex = 0;
  std::mutex _mutexPush, _mutexRequest;

 public:
  Timestamps(const Device& device);
  void resetQueryPool() noexcept;
  void pushTimestamp(std::string_view name, const CommandBuffer& commandBuffer);
  void popTimestamp(std::string_view name, const CommandBuffer& commandBuffer);
  void fetchTimestamps();
  // return copy, otherwise race condition between calling code and fetchTimestamps
  std::map<std::string, glm::dvec2> getTimestamps();
  ~Timestamps();
};
}  // namespace RenderGraph