export module Device;
import Instance;
import Surface;
import <VkBootstrap.h>;
import <volk.h>;

export namespace RenderGraph {
class Device final {
 private:
  vkb::Device _device;
  VkPhysicalDeviceProperties _deviceProperties;
  VkPhysicalDeviceDescriptorBufferPropertiesEXT _descriptorBufferProperties;
  std::vector<VkQueueFamilyProperties> _queueFamilyProperties;

 public:
  Device(const Surface& surface, const Instance& instance);
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  Device(Device&&) = delete;
  Device& operator=(Device&&) = delete;

  bool isFormatFeatureSupported(VkFormat format, VkImageTiling tiling, VkFormatFeatureFlagBits featureFlagBit) const;
  const VkDevice getLogicalDevice() const noexcept;
  const VkPhysicalDevice getPhysicalDevice() const noexcept;
  const VkQueue getQueue(vkb::QueueType type) const;
  int getQueueIndex(vkb::QueueType type) const;

  const vkb::Device& getDevice() const noexcept;
  const VkPhysicalDeviceProperties& getDeviceProperties() const noexcept;
  const VkPhysicalDeviceDescriptorBufferPropertiesEXT& getDescriptorBufferProperties() const noexcept;
  const VkQueueFamilyProperties& getQueueFamilyProperties(vkb::QueueType type) const noexcept;

  ~Device();
};
}  // namespace RenderGraph