export module DescriptorSet;
import Device;
import Buffer;
import DescriptorPool;
import <vector>;
import <map>;
import <volk.h>;

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
  const DescriptorSetLayout* _layout;

 public:
  DescriptorSet(const DescriptorSetLayout& layout, DescriptorPool& descriptorPool, const Device& device);
  DescriptorSet(const DescriptorSet&) = delete;
  DescriptorSet& operator=(const DescriptorSet&) = delete;
  DescriptorSet(DescriptorSet&& other) = delete;
  DescriptorSet& operator=(DescriptorSet&& other) = delete;

  void updateCustom(const std::map<int, VkDescriptorBufferInfo>& buffers,
                    const std::map<int, VkDescriptorImageInfo>& images) noexcept;
  VkDescriptorSet getDescriptorSet() const noexcept;
  ~DescriptorSet();
};
}  // namespace RenderGraph