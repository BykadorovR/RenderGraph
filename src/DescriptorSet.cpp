module DescriptorSet;
import <sstream>;
using namespace RenderGraph;

DescriptorSetLayout::DescriptorSetLayout(const Device& device) noexcept : _device(&device) {}

DescriptorSetLayout::DescriptorSetLayout(DescriptorSetLayout&& other)
    : _device(other._device),
      _descriptorSetLayout(other._descriptorSetLayout),
      _info(std::move(other._info)) {
  other._descriptorSetLayout = nullptr;
}

void DescriptorSetLayout::createCustom(const std::vector<VkDescriptorSetLayoutBinding>& info) {
  _info = info;

  auto layoutInfo = VkDescriptorSetLayoutCreateInfo{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                                    .bindingCount = static_cast<uint32_t>(_info.size()),
                                                    .pBindings = _info.data()};
  if (vkCreateDescriptorSetLayout(_device->getLogicalDevice(), &layoutInfo, nullptr, &_descriptorSetLayout) !=
      VK_SUCCESS) {
    throw std::runtime_error("failed to create descriptor set layout!");
  }
}

const std::vector<VkDescriptorSetLayoutBinding>& DescriptorSetLayout::getLayoutInfo() const noexcept { return _info; }

VkDescriptorSetLayout DescriptorSetLayout::getDescriptorSetLayout() const noexcept { return _descriptorSetLayout; }

DescriptorSetLayout::~DescriptorSetLayout() {
  vkDestroyDescriptorSetLayout(_device->getLogicalDevice(), _descriptorSetLayout, nullptr);
}

DescriptorSet::DescriptorSet(const DescriptorSetLayout& layout, DescriptorPool& descriptorPool, const Device& device)
    : _descriptorPool(&descriptorPool),
      _device(&device) {
  _layout = &layout;

  _descriptorPool->notify(_layout->getLayoutInfo(), 1);

  auto descriptorLayout = _layout->getDescriptorSetLayout();
  VkDescriptorSetAllocateInfo allocInfo{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                        .descriptorPool = _descriptorPool->getDescriptorPool(),
                                        .descriptorSetCount = 1,
                                        .pSetLayouts = &descriptorLayout};
  auto sts = vkAllocateDescriptorSets(device.getLogicalDevice(), &allocInfo, &_descriptorSet);
  if (sts != VK_SUCCESS) {
    std::ostringstream descriptors;
    descriptors << "failed to allocate descriptor sets: allocated sets: " << descriptorPool.getDescriptorSetsNumber()
                << ", descriptors: ";
    for (auto&& [key, value] : descriptorPool.getDescriptorsNumber()) {
      descriptors << key << ":" << value << " ";
    }
    throw std::runtime_error(descriptors.str());
  }
}

void DescriptorSet::updateCustom(const std::map<int, VkDescriptorBufferInfo>& buffers,
                                 const std::map<int, VkDescriptorImageInfo>& images) noexcept {
  std::vector<VkWriteDescriptorSet> descriptorWrites;
  descriptorWrites.reserve(buffers.size() + images.size());
  for (auto&& [key, value] : buffers) {
    VkWriteDescriptorSet descriptorSet = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                          .dstSet = _descriptorSet,
                                          .dstBinding = _layout->getLayoutInfo()[key].binding,
                                          .dstArrayElement = 0,
                                          .descriptorCount = _layout->getLayoutInfo()[key].descriptorCount,
                                          .descriptorType = _layout->getLayoutInfo()[key].descriptorType,
                                          .pBufferInfo = &buffers.at(key)};
    descriptorWrites.push_back(descriptorSet);
  }
  for (auto&& [key, value] : images) {
    VkWriteDescriptorSet descriptorSet = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                          .dstSet = _descriptorSet,
                                          .dstBinding = _layout->getLayoutInfo()[key].binding,
                                          .dstArrayElement = 0,
                                          .descriptorCount = _layout->getLayoutInfo()[key].descriptorCount,
                                          .descriptorType = _layout->getLayoutInfo()[key].descriptorType,
                                          .pImageInfo = &images.at(key)};
    descriptorWrites.push_back(descriptorSet);
  }

  vkUpdateDescriptorSets(_device->getLogicalDevice(), descriptorWrites.size(), descriptorWrites.data(), 0, nullptr);
}

VkDescriptorSet DescriptorSet::getDescriptorSet() const noexcept { return _descriptorSet; }

DescriptorSet::~DescriptorSet() { _descriptorPool->notify(_layout->getLayoutInfo(), -1); }