export module Device;
import Instance;
import Surface;
import <VkBootstrap.h>;
import <volk.h>;

export namespace RenderGraph {
class Device final {
 private:
  vkb::Device _device;
  const Surface* _surface;
  const Instance* _instance;
  std::vector<VkQueueFamilyProperties> _queueFamilyProperties;
  std::vector<std::string> _optionalExtensions = {"VK_EXT_descriptor_buffer"};

 public:
  Device(const Surface& surface, const Instance& instance);
  Device(const Device&) = delete;
  Device& operator=(const Device&) = delete;
  Device(Device&&) = delete;
  Device& operator=(Device&&) = delete;
  void initialize();

  void setOptionalExtensions(const std::vector<std::string>& extensions) noexcept;
  bool isFormatFeatureSupported(VkFormat format, VkImageTiling tiling, VkFormatFeatureFlagBits featureFlagBit) const;
  const VkDevice getLogicalDevice() const noexcept;
  const VkPhysicalDevice getPhysicalDevice() const noexcept;
  const VkQueue getQueue(vkb::QueueType type) const;
  int getQueueIndex(vkb::QueueType type) const;

  const vkb::Device& getDevice() const noexcept;
  bool isExtensionSupported(std::string name) const;
  std::vector<std::string> getOptionalExtensions() const noexcept;
  void getFeatureProperties(auto& property) const noexcept {
    VkPhysicalDeviceProperties2 properties2{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
                                            .pNext = &property};
    vkGetPhysicalDeviceProperties2(getPhysicalDevice(), &properties2);
  }
  const VkQueueFamilyProperties& getQueueFamilyProperties(vkb::QueueType type) const noexcept;

  ~Device();
};
}  // namespace RenderGraph