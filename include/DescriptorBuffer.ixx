export module DescriptorBuffer;
import Device;
import Buffer;
import Allocator;
import Command;
import Texture;
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
class DescriptorSetTest_Create_Test;
class DescriptorSetTest_Update_Test;

export namespace RenderGraph {
class DescriptorSetLayout final {
 private:
  const Device* _device;
  VkDescriptorSetLayout _descriptorSetLayout{};
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

class DescriptorHandler {
 private:
  friend class ::DescriptorSetTest_Create_Test;
  friend class ::DescriptorSetTest_Update_Test;

 protected:
  struct Resource {
    enum class Type { BUFFER, TEXTURE } type;
    std::vector<Buffer*> buffers;
    std::vector<Texture*> textures;
  };
  std::vector<Resource> _resources;

 public:
  void add(std::vector<Texture*> textures);
  void add(std::vector<Buffer*> buffers);
  virtual void initialize(const CommandBuffer& commandBuffer) = 0;
  virtual void bind(VkPipelineBindPoint bindPoint,
                    const VkPipelineLayout& pipelineLayout,
                    const CommandBuffer& commandBuffer) = 0;
  virtual ~DescriptorHandler() = default;
};

class DescriptorBuffer final : public DescriptorHandler {
 private:
  friend class ::DescriptorBufferTest_Create_Test;
  friend class ::DescriptorBufferTest_BigDescriptorCount_Test;
  friend class ::DescriptorBufferTest_DifferentBinning_Test;
  friend class ::DescriptorBufferTest_DifferentSets_Test;
  friend class ::DescriptorBufferTest_Update_Test;

 private:
  const Device* _device;
  const MemoryAllocator* _memoryAllocator;
  std::vector<DescriptorSetLayout*> _descriptorLayouts;
  std::unique_ptr<Buffer> _descriptorBuffer = nullptr;
  VkDeviceAddress _address = 0;
  std::vector<std::vector<VkDeviceSize>> _offsets;
  std::vector<VkDeviceSize> _layoutSize;
  std::vector<uint8_t> _descriptors;
  // binding and iterator inside it
  std::pair<int, int> _binding = {0, 0};
  int _bindingOffset = 0;
  int _set = 0;
  int _frame = 0;
  int _currentBind = 0;
  int _getDescriptorSize(VkDescriptorType descriptorType);
  VkBufferUsageFlags _usage = VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT |
                              VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT |
                              VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
  void _add(VkDescriptorGetInfoEXT info);

 public:
  DescriptorBuffer(const std::vector<DescriptorSetLayout*>& layouts,
                   const MemoryAllocator& memoryAllocator,
                   const Device& device);
  void initialize(const CommandBuffer& commandBuffer) override;
  void bind(VkPipelineBindPoint bindPoint,
            const VkPipelineLayout& pipelineLayout,
            const CommandBuffer& commandBuffer) override;
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

class DescriptorSet final : public DescriptorHandler {
 private:
  friend class ::DescriptorSetTest_Create_Test;
  friend class ::DescriptorSetTest_Update_Test;

 private:
  DescriptorPool* _descriptorPool;
  const Device* _device;
  // frame - set
  std::vector<std::vector<VkDescriptorSet>> _descriptorSet;
  // set
  std::vector<DescriptorSetLayout*> _descriptorLayouts;

  int _bindingNumber = 0;
  int _frame = 0;
  int _number = 0;
  int _currentBind = 0;

  int _calculateDescriptorSetIndex();
  void _allocateDescriptorSetsForNextFrame();

 public:
  DescriptorSet(const std::vector<DescriptorSetLayout*>& layouts, DescriptorPool& descriptorPool, const Device& device);
  DescriptorSet(const DescriptorSet&) = delete;
  DescriptorSet& operator=(const DescriptorSet&) = delete;
  DescriptorSet(DescriptorSet&& other) = delete;
  DescriptorSet& operator=(DescriptorSet&& other) = delete;

  void initialize(const CommandBuffer& commandBuffer) override;
  void bind(VkPipelineBindPoint bindPoint,
            const VkPipelineLayout& pipelineLayout,
            const CommandBuffer& commandBuffer) override;

  ~DescriptorSet();
};
}  // namespace RenderGraph
