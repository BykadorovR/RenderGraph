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

struct DeviceRequirements {
  VkPhysicalDeviceFeatures features{};
  VkPhysicalDeviceVulkan12Features features12{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
  };
  VkPhysicalDeviceVulkan13Features features13{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
  };
  VkPhysicalDeviceDescriptorBufferFeaturesEXT descriptorBufferFeatures{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_FEATURES_EXT,
      .descriptorBuffer = true,
  };
  VkBaseOutStructure* extensionFeatures = nullptr;
  std::vector<std::string> requiredExtensions;
  std::vector<std::string> optionalExtensions;
};

class Device final {
 private:
  vkb::Device _device;
  const Surface* _surface = nullptr;
  const Instance* _instance;
  bool _initialized = false;
  std::vector<VkQueueFamilyProperties> _queueFamilyProperties;
  std::vector<std::string> _enabledExtensions;

 public:
  explicit Device(const Instance& instance);
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  Device(Device&&) = delete;
  Device& operator=(Device&&) = delete;
  void setSurface(const Surface& surface);
  void initialize(const DeviceRequirements& requirements = {});

  bool isFormatFeatureSupported(VkFormat format, VkImageTiling tiling, VkFormatFeatureFlagBits featureFlagBit) const;
  const VkDevice getLogicalDevice() const noexcept;
  const VkPhysicalDevice getPhysicalDevice() const noexcept;
  const VkQueue getQueue(QueueType type) const;
  int getQueueIndex(QueueType type) const;

  const vkb::Device& getDevice() const noexcept;
  bool isExtensionSupported(std::string name) const;
  bool isExtensionEnabled(std::string name) const;
  void getFeatureProperties(auto& property) const noexcept {
    VkPhysicalDeviceProperties2 properties2{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
                                            .pNext = &property};
    vkGetPhysicalDeviceProperties2(getPhysicalDevice(), &properties2);
  }
  const VkQueueFamilyProperties& getQueueFamilyProperties(QueueType type) const;

  ~Device();
};
}  // namespace RenderGraph
