module DescriptorPool;
using namespace RenderGraph;

DescriptorPool::DescriptorPool(DescriptorPoolSize poolSize, const Device& device) : _device(&device) {
  std::vector<VkDescriptorPoolSize> poolSizes{
      {.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .descriptorCount = static_cast<uint32_t>(poolSize.uniformBuffer)},
      {.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = static_cast<uint32_t>(poolSize.sampler)},
      {.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = static_cast<uint32_t>(poolSize.computeImage)},
      {.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = static_cast<uint32_t>(poolSize.ssbo)}};

  VkDescriptorPoolCreateInfo poolInfo{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                      .maxSets = static_cast<uint32_t>(poolSize.descriptorSets),
                                      .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
                                      .pPoolSizes = poolSizes.data()};

  if (vkCreateDescriptorPool(device.getLogicalDevice(), &poolInfo, nullptr, &_descriptorPool) != VK_SUCCESS) {
    throw std::runtime_error("failed to create descriptor pool!");
  }
}

void DescriptorPool::notify(const std::vector<VkDescriptorSetLayoutBinding>& layoutInfo, int number) noexcept {  
  for (auto&& info : layoutInfo) {
    _descriptorTypes[info.descriptorType] += number;
  }

  _descriptorSetsNumber += number;
}

const std::map<VkDescriptorType, int>& DescriptorPool::getDescriptorsNumber() const noexcept {
  return _descriptorTypes;
}

int DescriptorPool::getDescriptorSetsNumber() const noexcept { return _descriptorSetsNumber; }

VkDescriptorPool DescriptorPool::getDescriptorPool() const noexcept { return _descriptorPool; }

DescriptorPool::~DescriptorPool() { vkDestroyDescriptorPool(_device->getLogicalDevice(), _descriptorPool, nullptr); }