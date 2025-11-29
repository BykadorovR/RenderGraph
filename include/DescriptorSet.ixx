export module DescriptorSet;
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
  std::vector<uint8_t> _descriptors;
  int _getDescriptorSize(VkDescriptorType descriptorType);
 public:
  DescriptorBuffer(const MemoryAllocator& memoryAllocator, const Device& device);
  void add(VkDescriptorImageInfo info, VkDescriptorType descriptorType);
  void add(VkDescriptorAddressInfoEXT info, VkDescriptorType descriptorType);
  void initialize(const CommandBuffer& commandBuffer);
  const Buffer* getBuffer();
};
}  // namespace RenderGraph