module DescriptorSet;
import <sstream>;
import <ranges>;
import <algorithm>;
import <numeric>;
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
      .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT,
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
  _layoutInfo = layout.getLayoutInfo();

  _descriptorPool->notify(_layoutInfo, 1);

  auto descriptorLayout = layout.getDescriptorSetLayout();
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

void DescriptorSet::updateCustom(const std::map<int, std::vector<VkDescriptorBufferInfo>>& buffers,
                                 const std::map<int, std::vector<VkDescriptorImageInfo>>& images) noexcept {
  std::vector<VkWriteDescriptorSet> descriptorWrites;
  descriptorWrites.reserve(buffers.size() + images.size());
  for (auto&& [key, value] : buffers) {
    VkWriteDescriptorSet descriptorSet = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                          .dstSet = _descriptorSet,
                                          .dstBinding = _layoutInfo[key].binding,
                                          .dstArrayElement = 0,
                                          .descriptorCount = _layoutInfo[key].descriptorCount,
                                          .descriptorType = _layoutInfo[key].descriptorType,
                                          .pBufferInfo = buffers.at(key).data()};
    descriptorWrites.push_back(descriptorSet);
  }
  for (auto&& [key, value] : images) {
    VkWriteDescriptorSet descriptorSet = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                          .dstSet = _descriptorSet,
                                          .dstBinding = _layoutInfo[key].binding,
                                          .dstArrayElement = 0,
                                          .descriptorCount = _layoutInfo[key].descriptorCount,
                                          .descriptorType = _layoutInfo[key].descriptorType,
                                          .pImageInfo = images.at(key).data()};
    descriptorWrites.push_back(descriptorSet);
  }

  vkUpdateDescriptorSets(_device->getLogicalDevice(), descriptorWrites.size(), descriptorWrites.data(), 0, nullptr);
}

VkDescriptorSet DescriptorSet::getDescriptorSet() const noexcept { return _descriptorSet; }

DescriptorSet::~DescriptorSet() { _descriptorPool->notify(_layoutInfo, -1); }

DescriptorBuffer::DescriptorBuffer(const DescriptorSetLayout& layout, 
                                   const MemoryAllocator& memoryAllocator,
                                   const Device& device) {
  _memoryAllocator = &memoryAllocator;
  _device = &device;
  
  for (int i = 0; i < layout.getLayoutInfo().size(); i++) {
    VkDeviceSize offset;
    vkGetDescriptorSetLayoutBindingOffsetEXT(device.getLogicalDevice(), layout.getDescriptorSetLayout(), i, &offset);
    _offsets.push_back(offset);
  }
}

int DescriptorBuffer::_getDescriptorSize(VkDescriptorType descriptorType) {
  const auto& _bufferProperties = &_device->getDescriptorBufferProperties();
    switch (descriptorType) {
    case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
      return _bufferProperties->combinedImageSamplerDescriptorSize;
    case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
      return _bufferProperties->sampledImageDescriptorSize;
    case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
      return _bufferProperties->storageImageDescriptorSize;    
    case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
      return _bufferProperties->uniformBufferDescriptorSize;
    case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
      return _bufferProperties->storageBufferDescriptorSize;    
    default:
      throw std::runtime_error("Unsupported descriptor type for descriptor buffer");
    }
}

void DescriptorBuffer::add(VkDescriptorImageInfo info, VkDescriptorType descriptorType) {
  if (_descriptorBuffer != nullptr) {
    throw std::runtime_error("Cannot add descriptors after initialization");
  }
  VkDescriptorGetInfoEXT getInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT};
  getInfo.type = descriptorType;
  switch (descriptorType) {
    case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
      getInfo.data.pSampledImage = &info;
      break;
    case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
      getInfo.data.pCombinedImageSampler = &info;
      break;
    case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
      getInfo.data.pStorageImage = &info;
      break;
    default:
      throw std::runtime_error("Unsupported descriptor type for descriptor buffer");
  }

  if (_offsets[_number] > _descriptors.size()) {
    std::vector<uint8_t> allignment(_offsets[_number] - _descriptors.size(), 0);
    std::ranges::copy(allignment, std::back_inserter(_descriptors));
  }

  auto descSize = _getDescriptorSize(descriptorType);
  std::vector<uint8_t> descriptorCPU(descSize);
  vkGetDescriptorEXT(_device->getLogicalDevice(), &getInfo, descSize, descriptorCPU.data());
  std::ranges::copy(descriptorCPU, std::back_inserter(_descriptors));
  _number++;
}

void DescriptorBuffer::add(VkDescriptorAddressInfoEXT info, VkDescriptorType descriptorType) {
  if (_descriptorBuffer != nullptr) {
    throw std::runtime_error("Cannot add descriptors after initialization");
  }
  VkDescriptorGetInfoEXT getInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT};
  getInfo.type = descriptorType;
  switch (descriptorType) {
    case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
      getInfo.data.pUniformBuffer = &info;
      break;
    case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
      getInfo.data.pStorageBuffer = &info;
      break;
    default:
      throw std::runtime_error("Unsupported descriptor type for descriptor buffer");
  }

  if (_offsets[_number] > _descriptors.size()) {
    std::vector<uint8_t> allignment(_offsets[_number] - _descriptors.size(), 0);
    std::ranges::copy(allignment, std::back_inserter(_descriptors));
  }

  auto descSize = _getDescriptorSize(descriptorType);
  std::vector<uint8_t> descriptorCPU(descSize);
  vkGetDescriptorEXT(_device->getLogicalDevice(), &getInfo, descSize, descriptorCPU.data());
  std::ranges::copy(descriptorCPU, std::back_inserter(_descriptors));
  _number++;
}

const Buffer* DescriptorBuffer::getBuffer(const CommandBuffer& commandBuffer) {
  if (_descriptorBuffer == nullptr) {
    // first need to allocate the buffer itself
    int size = _descriptors.size();
    _descriptorBuffer = std::make_unique<Buffer>(
        size,
        VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT, *_memoryAllocator);

    // and bind all descriptors to it
    _descriptorBuffer->setData(std::span(reinterpret_cast<const std::byte*>(_descriptors.data()), _descriptors.size()),
                               commandBuffer);
  }
  return _descriptorBuffer.get();
}

std::vector<VkDeviceSize> DescriptorBuffer::getOffsets() const noexcept { return _offsets; }