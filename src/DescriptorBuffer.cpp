module DescriptorBuffer;
import <sstream>;
import <ranges>;
import <algorithm>;
import <numeric>;
import <iostream>;
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

DescriptorBuffer::DescriptorBuffer(const std::vector<const DescriptorSetLayout*>& layouts,
                                   const MemoryAllocator& memoryAllocator,
                                   const Device& device) {
  _memoryAllocator = &memoryAllocator;
  _device = &device;
  _descriptorLayouts = layouts;
  
  // one buffer for the entire shader, but separate offsets for the sets inside the buffer
  _offsets.resize(layouts.size());
  _layoutSize.resize(layouts.size());
  for (int id = 0; id < layouts.size(); id++) {    
    auto&& layout = layouts[id];
    for (int i = 0; i < layout->getLayoutInfo().size(); i++) {
      for (int j = 0; j < layout->getLayoutInfo()[i].descriptorCount; j++) {        
        VkDeviceSize offset;
        vkGetDescriptorSetLayoutBindingOffsetEXT(device.getLogicalDevice(), layout->getDescriptorSetLayout(), i,
                                                 &offset);
        _offsets[id].push_back(offset + _getDescriptorSize(layout->getLayoutInfo()[i].descriptorType) * j);
      }
    }

    VkDeviceSize layoutSize;
    vkGetDescriptorSetLayoutSizeEXT(device.getLogicalDevice(), layout->getDescriptorSetLayout(), &layoutSize);
    _layoutSize[id] += layoutSize;
  }
}

int DescriptorBuffer::_getDescriptorSize(VkDescriptorType descriptorType) {
  auto bufferProperties = VkPhysicalDeviceDescriptorBufferPropertiesEXT{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_PROPERTIES_EXT};    
  _device->getFeatureProperties(bufferProperties);
    switch (descriptorType) {
    case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
      return bufferProperties.combinedImageSamplerDescriptorSize;
    case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
      return bufferProperties.sampledImageDescriptorSize;
    case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
      return bufferProperties.storageImageDescriptorSize;    
    case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
      return bufferProperties.uniformBufferDescriptorSize;
    case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
      return bufferProperties.storageBufferDescriptorSize;    
    default:
      throw std::runtime_error("Unsupported descriptor type for descriptor buffer");
    }
}

void DescriptorBuffer::_add(VkDescriptorGetInfoEXT info) {  
  auto setSize = std::reduce(_layoutSize.begin(), _layoutSize.end());  
  // allign for the whole set (for frames in flight)
  if (_set == 0) {    
    _descriptors.resize(setSize * (_frame + 1));
  }

  auto descSize = _getDescriptorSize(info.type);
  std::vector<uint8_t> descriptorCPU(descSize);
  vkGetDescriptorEXT(_device->getLogicalDevice(), &info, descSize, descriptorCPU.data());
  std::copy(descriptorCPU.begin(), descriptorCPU.end(), _descriptors.begin() + setSize * _frame + _offsets[_set][_binning]);

  // calculate next frame, set, binning
  _binning++;
  if (_descriptorLayouts[_set]->getLayoutInfo().size() == _binning) {
    _binning = 0;
    _set++;
    if (_descriptorLayouts.size() == _set) {
      _set = 0;
      _frame++;
    }
  }
}

void DescriptorBuffer::add(VkDescriptorImageInfo info) {
  if (_descriptorBuffer != nullptr) {
    throw std::runtime_error("Cannot add descriptors after initialization");
  }
  VkDescriptorGetInfoEXT getInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT};
  getInfo.type = _descriptorLayouts[_set]->getLayoutInfo()[_binning].descriptorType;
  switch (getInfo.type) {
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

  _add(getInfo);
}

void DescriptorBuffer::add(VkDescriptorAddressInfoEXT info) {
  if (_descriptorBuffer != nullptr) {
    throw std::runtime_error("Cannot add descriptors after initialization");
  }
  VkDescriptorGetInfoEXT getInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT};
  getInfo.type = _descriptorLayouts[_set]->getLayoutInfo()[_binning].descriptorType;
  switch (getInfo.type) {
    case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
      getInfo.data.pUniformBuffer = &info;
      break;
    case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
      getInfo.data.pStorageBuffer = &info;
      break;
    default:
      throw std::runtime_error("Unsupported descriptor type for descriptor buffer");
  }

  _add(getInfo);
}

void DescriptorBuffer::initialize(const CommandBuffer& commandBuffer) {
  if (_descriptorBuffer != nullptr) throw std::runtime_error("Descriptor buffer is already initialized");
  // first need to allocate the buffer itself
  int size = _descriptors.size();
  _descriptorBuffer = std::make_unique<Buffer>(
      size, _usage, VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
      *_memoryAllocator);

  // and bind all descriptors to it
  _descriptorBuffer->setData(std::span(reinterpret_cast<const std::byte*>(_descriptors.data()), _descriptors.size()),
                             commandBuffer);
}

void DescriptorBuffer::bind(int frameInFlight,
                            const VkPipelineLayout& pipelineLayout,
                            const CommandBuffer& commandBuffer,
                            VkPipelineBindPoint bindPoint) {
  auto bufferBinding = VkDescriptorBufferBindingInfoEXT{VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT, nullptr,
                                                        _descriptorBuffer->getDeviceAddress(*_device), _usage};
  std::vector<VkDeviceSize> offset(_layoutSize.size());
  for (int i = 0; i < _layoutSize.size(); i++) offset[i] = _layoutSize[i] * frameInFlight;
  std::vector<uint32_t> bufIndex(_layoutSize.size(), 0);
  // we use 1 buffer for the whole shader
  vkCmdBindDescriptorBuffersEXT(commandBuffer.getCommandBuffer(), 1, &bufferBinding);
  // but specify offsets for every set
  vkCmdSetDescriptorBufferOffsetsEXT(commandBuffer.getCommandBuffer(), bindPoint, pipelineLayout,
                                     0, _descriptorLayouts.size(), bufIndex.data(), offset.data());
}

DescriptorPool::DescriptorPool(DescriptorPoolSize poolSize, const Device& device) {
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

DescriptorSet::DescriptorSet(const DescriptorSetLayout& layout, DescriptorPool& descriptorPool, const Device& device) {
  _layout = &layout;

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

void DescriptorSet::add(std::vector<VkDescriptorBufferInfo> info) {
  int key = _descriptorWrites.size();
  VkWriteDescriptorSet descriptorSet = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                        .dstSet = _descriptorSet,
                                        .dstBinding = _layout->getLayoutInfo()[key].binding,
                                        .dstArrayElement = 0,
                                        .descriptorCount = _layout->getLayoutInfo()[key].descriptorCount,
                                        .descriptorType = _layout->getLayoutInfo()[key].descriptorType,
                                        .pBufferInfo = info.data()};
  _descriptorWrites.push_back(descriptorSet);
}

void DescriptorSet::add(std::vector<VkDescriptorImageInfo> info) {
  int key = _descriptorWrites.size();
  VkWriteDescriptorSet descriptorSet = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                        .dstSet = _descriptorSet,
                                        .dstBinding = _layout->getLayoutInfo()[key].binding,
                                        .dstArrayElement = 0,
                                        .descriptorCount = _layout->getLayoutInfo()[key].descriptorCount,
                                        .descriptorType = _layout->getLayoutInfo()[key].descriptorType,
                                        .pImageInfo = info.data()};
  _descriptorWrites.push_back(descriptorSet);
}

void DescriptorSet::initialize() {
  vkUpdateDescriptorSets(_device->getLogicalDevice(), _descriptorWrites.size(), _descriptorWrites.data(), 0, nullptr);
}

VkDescriptorSet DescriptorSet::getDescriptorSet() const noexcept { return _descriptorSet; }

DescriptorSet::~DescriptorSet() { _descriptorPool->notify(_layout->getLayoutInfo(), -1); }

DescriptorHelper::DescriptorHelper(const std::vector<const DescriptorSetLayout*>& layouts,
                                   const MemoryAllocator& memoryAllocator,
                                   const Device& device) {
  _descriptorBuffer = std::make_unique<DescriptorBuffer>(layouts, memoryAllocator, device);
}

DescriptorHelper::DescriptorHelper(const std::vector<const DescriptorSetLayout*>& layouts,
                                   DescriptorPool& descriptorPool,
                                   const Device& device) {
  _descriptorLayouts = layouts;
  _descriptorSet.reserve(_descriptorLayouts.size());
  for (auto&& layout : _descriptorLayouts) {
    _descriptorSet.push_back(std::make_unique<DescriptorSet>(*layout, descriptorPool, device));
  }
}

void DescriptorHelper::add(std::vector<VkDescriptorImageInfo> images, VkDescriptorType type) {  
}

void DescriptorHelper::add(std::vector<Buffer> buffers, VkDescriptorType type) {

}

void DescriptorHelper::initialize() { }

void DescriptorHelper::bind(int frame, const CommandBuffer& commandBuffer, const VkPipelineLayout& layout) {

}
