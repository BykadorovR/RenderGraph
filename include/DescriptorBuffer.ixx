export module DescriptorBuffer;
import Device;
import Buffer;
import Allocator;
import Command;
import <vector>;
import <map>;
import <volk.h>;
import <memory>;

export namespace RenderGraph {
class DescriptorSetLayout final {
 private:
  const Device* _device;
  VkDescriptorSetLayout _descriptorSetLayout;
  std::vector<VkDescriptorSetLayoutBinding> _info;

 public:
  DescriptorSetLayout(const Device& device) noexcept;
  DescriptorSetLayout(const DescriptorSetLayout&) = delete;
  DescriptorSetLayout& operator=(const DescriptorSetLayout&) = delete;
  DescriptorSetLayout(DescriptorSetLayout&& other);

  DescriptorSetLayout& operator=(DescriptorSetLayout&& other) = delete;

  void createCustom(const std::vector<VkDescriptorSetLayoutBinding>& info);
  const std::vector<VkDescriptorSetLayoutBinding>& getLayoutInfo() const noexcept;
  VkDescriptorSetLayout getDescriptorSetLayout() const noexcept;
  ~DescriptorSetLayout();
};

class DescriptorBuffer final {
 private:
  const Device* _device;
  const MemoryAllocator* _memoryAllocator;

  std::unique_ptr<Buffer> _descriptorBuffer = nullptr;
  VkDeviceAddress _address = 0;
  std::vector<VkDeviceSize> _offsets;
  VkDeviceSize _layoutSize = 0;
  std::vector<uint8_t> _descriptors;
  int _number = 0;
  int _getDescriptorSize(VkDescriptorType descriptorType);
  VkBufferUsageFlags _usage = VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT |
                              VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT |
                              VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  void _add(VkDescriptorGetInfoEXT info, VkDescriptorType descriptorType);
 public:
  DescriptorBuffer(std::initializer_list<const DescriptorSetLayout*> layouts,
                   const MemoryAllocator& memoryAllocator,
                   const Device& device);
  void add(VkDescriptorImageInfo info, VkDescriptorType descriptorType);
  void add(VkDescriptorAddressInfoEXT info, VkDescriptorType descriptorType);
  void initialize(const CommandBuffer& commandBuffer);
  const Buffer* getBuffer();
  VkDescriptorBufferBindingInfoEXT getBufferBindingInfo() const noexcept;
  std::vector<VkDeviceSize> getOffsets() const noexcept;
  VkDeviceSize getLayoutSize() const noexcept;
};
}  // namespace RenderGraph