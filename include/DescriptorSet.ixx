export module DescriptorSet;
import Device;
import Buffer;
import Allocator;
import Command;
import DescriptorPool;
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

class DescriptorSet final {
 private:
  DescriptorPool* _descriptorPool;
  const Device* _device;
  VkDescriptorSet _descriptorSet;
  std::vector<VkDescriptorSetLayoutBinding> _layoutInfo;

 public:
  DescriptorSet(const DescriptorSetLayout& layout, DescriptorPool& descriptorPool, const Device& device);
  DescriptorSet(const DescriptorSet&) = delete;
  DescriptorSet& operator=(const DescriptorSet&) = delete;
  DescriptorSet(DescriptorSet&& other) = delete;
  DescriptorSet& operator=(DescriptorSet&& other) = delete;

  void updateCustom(const std::map<int, std::vector<VkDescriptorBufferInfo>>& buffers,
                    const std::map<int, std::vector<VkDescriptorImageInfo>>& images) noexcept;
  VkDescriptorSet getDescriptorSet() const noexcept;
  ~DescriptorSet();
};

class DescriptorBuffer final {
 private:
  const Device* _device;
  const MemoryAllocator* _memoryAllocator;

  std::unique_ptr<Buffer> _descriptorBuffer = nullptr;
  VkDeviceAddress _address = 0;
  std::vector<VkDeviceSize> _offsets;
  VkDeviceSize _layoutSize;
  std::vector<uint8_t> _descriptors;
  int _number = 0;
  int _sizeWithin = 0;
  int _getDescriptorSize(VkDescriptorType descriptorType);
  void _add(VkDescriptorGetInfoEXT info, VkDescriptorType descriptorType);
 public:
  DescriptorBuffer(const DescriptorSetLayout& layout, const MemoryAllocator& memoryAllocator, const Device& device);
  void add(VkDescriptorImageInfo info, VkDescriptorType descriptorType);
  void add(VkDescriptorAddressInfoEXT info, VkDescriptorType descriptorType);
  const Buffer* getBuffer(const CommandBuffer& commandBuffer);
  std::vector<VkDeviceSize> getOffsets() const noexcept;
  VkDeviceSize getLayoutSize() const noexcept;
};
}  // namespace RenderGraph