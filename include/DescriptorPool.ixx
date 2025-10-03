export module DescriptorPool;
import Device;
import <volk.h>;
import <map>;
import <vector>;

export namespace RenderGraph {
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
}  // namespace RenderGraph