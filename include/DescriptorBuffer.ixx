export module DescriptorBuffer;
import Device;
import Buffer;
import Allocator;
import Command;
import <vector>;
import <map>;
import <volk.h>;
import <memory>;

// Forward declarations for test classes, not visible outside this module
class DescriptorBufferTest_Create_Test;
class DescriptorBufferTest_BigDescriptorCount_Test;
class DescriptorBufferTest_DifferentBinning_Test;
class DescriptorBufferTest_DifferentSets_Test;
class DescriptorBufferTest_Update_Test;

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
  friend class ::DescriptorBufferTest_Create_Test;
  friend class ::DescriptorBufferTest_BigDescriptorCount_Test;
  friend class ::DescriptorBufferTest_DifferentBinning_Test;
  friend class ::DescriptorBufferTest_DifferentSets_Test;
  friend class ::DescriptorBufferTest_Update_Test;
 private:
  const Device* _device;
  const MemoryAllocator* _memoryAllocator;
  std::vector<const DescriptorSetLayout*> _descriptorLayouts;
  std::unique_ptr<Buffer> _descriptorBuffer = nullptr;
  VkDeviceAddress _address = 0;
  std::vector<std::vector<VkDeviceSize>> _offsets;
  std::vector<VkDeviceSize> _layoutSize;
  std::vector<uint8_t> _descriptors;
  int _binning = 0;
  int _set = 0;
  int _frame = 0;
  int _getDescriptorSize(VkDescriptorType descriptorType);
  VkBufferUsageFlags _usage = VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT |
                              VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT |
                              VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  void _add(VkDescriptorGetInfoEXT info);
 public:
  DescriptorBuffer(const std::vector<const DescriptorSetLayout*>& layouts,
                   const MemoryAllocator& memoryAllocator,
                   const Device& device);
  void add(VkDescriptorImageInfo info);
  void add(VkDescriptorAddressInfoEXT info);
  void initialize(const CommandBuffer& commandBuffer);
  void bind(int frameInFlight, const VkPipelineLayout& pipelineLayout, const CommandBuffer& commandBuffer, VkPipelineBindPoint bindPoint);
};

struct DescriptorPoolSize {
  // number of DESCRIPTORS in descriptor pool
  int uniformBuffer = 100;
  int sampler = 100;
  int computeImage = 100;
  int ssbo = 100;
  // number of DESCRIPTOR SETs in descriptor pool
  int descriptorSets = 1000;
};

class DescriptorPool final {
 private:
  const Device* _device;
  VkDescriptorPool _descriptorPool;
  std::map<VkDescriptorType, int> _descriptorTypes;
  int _descriptorSetsNumber = 0;

 public:
  DescriptorPool(DescriptorPoolSize poolSize, const Device& device);
  DescriptorPool(const DescriptorPool&) = delete;
  DescriptorPool& operator=(const DescriptorPool&) = delete;
  DescriptorPool(DescriptorPool&&) = delete;
  DescriptorPool& operator=(DescriptorPool&&) = delete;

  void notify(const std::vector<VkDescriptorSetLayoutBinding>& layoutInfo, int number) noexcept;
  // needed to calculate real number of descriptor sets and descriptors
  const std::map<VkDescriptorType, int>& getDescriptorsNumber() const noexcept;
  int getDescriptorSetsNumber() const noexcept;
  VkDescriptorPool getDescriptorPool() const noexcept;
  ~DescriptorPool();
};

class DescriptorSet final {
 private:
  DescriptorPool* _descriptorPool;
  const Device* _device;
  VkDescriptorSet _descriptorSet;
  const DescriptorSetLayout* _layout;
  std::vector<VkWriteDescriptorSet> _descriptorWrites;

 public:
  DescriptorSet(const DescriptorSetLayout& layout, DescriptorPool& descriptorPool, const Device& device);
  DescriptorSet(const DescriptorSet&) = delete;
  DescriptorSet& operator=(const DescriptorSet&) = delete;
  DescriptorSet(DescriptorSet&& other) = delete;
  DescriptorSet& operator=(DescriptorSet&& other) = delete;

  void add(std::vector<VkDescriptorBufferInfo> info);
  void add(std::vector<VkDescriptorImageInfo> info);
  void initialize();

  VkDescriptorSet getDescriptorSet() const noexcept;
  ~DescriptorSet();
};

class DescriptorHelper final {
 private:
  DescriptorPool* _descriptorPool;
  std::vector<std::unique_ptr<DescriptorSet>> _descriptorSet;
  std::vector<const DescriptorSetLayout*> _descriptorLayouts;

  std::unique_ptr<DescriptorBuffer> _descriptorBuffer;
 public:
  DescriptorHelper(const std::vector<const DescriptorSetLayout*>& layouts,
                   const MemoryAllocator& memoryAllocator,
                   const Device& device);
  DescriptorHelper(const std::vector<const DescriptorSetLayout*>& layouts,
                   DescriptorPool& descriptorPool,
                   const Device& device);
  void add(std::vector<VkDescriptorImageInfo> images, VkDescriptorType type);
  void add(std::vector<Buffer> buffers, VkDescriptorType type);
  void initialize();

  void bind(int frame, const CommandBuffer& commandBuffer, const VkPipelineLayout& layout);
};
}  // namespace RenderGraph