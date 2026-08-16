module;

#include <VkBootstrap.h>
#include <string>
#include <vector>
#include <volk.h>

export module Device;

import Instance;
import Surface;

export namespace RenderGraph {
enum class QueueType { PRESENT, GRAPHICS, COMPUTE, TRANSFER };

class Device final {
 private:
  vkb::Device _device;
  const Surface* _surface = nullptr;
  const Instance* _instance;
  bool _initialized = false;
  std::vector<VkQueueFamilyProperties> _queueFamilyProperties;
  std::vector<std::string> _optionalExtensions = {"VK_EXT_descriptor_buffer"};

 public:
  explicit Device(const Instance& instance);
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  Device(Device&&) = delete;
  Device& operator=(Device&&) = delete;
  void setSurface(const Surface& surface);
  void initialize();

  void setOptionalExtensions(const std::vector<std::string>& extensions);
  bool isFormatFeatureSupported(VkFormat format, VkImageTiling tiling, VkFormatFeatureFlagBits featureFlagBit) const;
  const VkDevice getLogicalDevice() const noexcept;
  const VkPhysicalDevice getPhysicalDevice() const noexcept;
  const VkQueue getQueue(QueueType type) const;
  int getQueueIndex(QueueType type) const;

  const vkb::Device& getDevice() const noexcept;
  bool isExtensionSupported(std::string name) const;
  std::vector<std::string> getOptionalExtensions() const;
  void getFeatureProperties(auto& property) const noexcept {
    VkPhysicalDeviceProperties2 properties2{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
                                            .pNext = &property};
    vkGetPhysicalDeviceProperties2(getPhysicalDevice(), &properties2);
  }
  const VkQueueFamilyProperties& getQueueFamilyProperties(QueueType type) const;

  ~Device();
};
}  // namespace RenderGraph
