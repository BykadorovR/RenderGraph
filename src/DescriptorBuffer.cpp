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
                                                    .bindingCount = static_cast<uint32_t>(_info.size()),
                                                    .pBindings = _info.data()};
  auto optionalExtensions = _device->getOptionalExtensions();
  if (_device->isExtensionSupported("VK_EXT_descriptor_buffer") &&
      std::find(optionalExtensions.begin(), optionalExtensions.end(), "VK_EXT_descriptor_buffer") !=
          optionalExtensions.end())
    layoutInfo.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_DESCRIPTOR_BUFFER_BIT_EXT;

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

void DescriptorHandler::add(std::vector<Texture*> textures) {
  Resource r{.type = Resource::Type::TEXTURE, .textures = textures};
  _resources.push_back(std::move(r));
}

void DescriptorHandler::add(std::vector<Buffer*> buffers) {
  Resource r{.type = Resource::Type::BUFFER, .buffers = buffers};
  _resources.push_back(std::move(r));
}

DescriptorBuffer::DescriptorBuffer(const std::vector<DescriptorSetLayout*>& layouts,
                                   const MemoryAllocator& memoryAllocator,
                                   const Device& device) {
  _memoryAllocator = &memoryAllocator;
  _device = &device;
  _descriptorLayouts = layouts;

  auto descriptorBufferProperties = VkPhysicalDeviceDescriptorBufferPropertiesEXT{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_PROPERTIES_EXT};
  device.getFeatureProperties(descriptorBufferProperties);

  const VkDeviceSize alignment = descriptorBufferProperties.descriptorBufferOffsetAlignment;
  const auto alignUp = [](VkDeviceSize value, VkDeviceSize alignment) {
    return (value + alignment - 1) / alignment * alignment;
  };

  // one buffer for the entire shader, but separate offsets for the sets inside the buffer
  _offsets.resize(layouts.size());
  _layoutSize.resize(layouts.size());
  for (int id = 0; id < layouts.size(); id++) {
    auto&& layout = layouts[id];
    for (int i = 0; i < layout->getLayoutInfo().size(); i++) {
      for (int j = 0; j < layout->getLayoutInfo()[i].descriptorCount; j++) {
        VkDeviceSize offset;
        vkGetDescriptorSetLayoutBindingOffsetEXT(device.getLogicalDevice(), layout->getDescriptorSetLayout(),
                                                 layout->getLayoutInfo()[i].binding, &offset);
        _offsets[id].push_back(offset + _getDescriptorSize(layout->getLayoutInfo()[i].descriptorType) * j);
      }
    }

    VkDeviceSize layoutSize;
    vkGetDescriptorSetLayoutSizeEXT(device.getLogicalDevice(), layout->getDescriptorSetLayout(), &layoutSize);
    _layoutSize[id] = alignUp(layoutSize, alignment);
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

  std::vector<VkDeviceSize> offsetSet(_layoutSize.size());
  std::exclusive_scan(_layoutSize.begin(), _layoutSize.end(), offsetSet.begin(), VkDeviceSize{0});
  std::copy(descriptorCPU.begin(), descriptorCPU.end(),
            // offset between frames, between sets, inside set
            _descriptors.begin() + setSize * _frame + offsetSet[_set] + _offsets[_set][_bindingOffset]);

  // calculate next frame, set, binning
  if (_binding.second < _descriptorLayouts[_set]->getLayoutInfo()[_binding.first].descriptorCount) _binding.second++;
  if (_binding.second == _descriptorLayouts[_set]->getLayoutInfo()[_binding.first].descriptorCount) {
    _binding.first++;
    _binding.second = 0;
  }

  _bindingOffset++;

  if (_descriptorLayouts[_set]->getLayoutInfo().size() == _binding.first) {
    _binding.first = 0;
    _bindingOffset = 0;
    _set++;
    if (_descriptorLayouts.size() == _set) {
      _set = 0;
      _frame++;
    }
  }
}

void DescriptorBuffer::initialize(const CommandBuffer& commandBuffer) {
  if (_descriptorBuffer != nullptr) throw std::runtime_error("Descriptor buffer is already initialized");
  for (auto&& resource : _resources) {
    if (resource.type == Resource::Type::BUFFER) {
      for (auto&& buffer : resource.buffers) {
        auto info = VkDescriptorAddressInfoEXT{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT,
                                               .pNext = nullptr,
                                               .address = buffer->getDeviceAddress(*_device),
                                               .range = buffer->getSize(),
                                               .format = VK_FORMAT_UNDEFINED};
        VkDescriptorGetInfoEXT getInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT};
        getInfo.type = _descriptorLayouts[_set]->getLayoutInfo()[_binding.first].descriptorType;
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
    }
    if (resource.type == Resource::Type::TEXTURE) {
      // TODO: change layout and generate mipmaps here
      for (auto&& texture : resource.textures) {
        // change layout if it's not general
        auto&& image = texture->getImageView().getImage();
        if (image.getImageLayout() != VK_IMAGE_LAYOUT_GENERAL) {
          auto dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                               VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
          VkPipelineStageFlags2 dstStageMask = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                                               VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                                               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                               VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
          if (image.getAspectMask() & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) {
            dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            dstStageMask = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                           VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
          }
          image.changeLayout(image.getImageLayout(), VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
                             dstStageMask, dstAccessMask, commandBuffer);
        }
        // generate mip maps if needed
        if (image.getMipMapGenerated() == false && image.getMipMapNumber() > 1) image.generateMipmaps(commandBuffer);

        auto info = VkDescriptorImageInfo{.imageView = texture->getImageView().getImageView(),
                                          .imageLayout = texture->getImageView().getImage().getImageLayout()};
        auto&& sampler = texture->getSampler();
        if (sampler) info.sampler = sampler->getSampler();
        VkDescriptorGetInfoEXT getInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT};
        getInfo.type = _descriptorLayouts[_set]->getLayoutInfo()[_binding.first].descriptorType;
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
    }
  }

  // first need to allocate the buffer itself
  int size = _descriptors.size();
  _descriptorBuffer = std::make_unique<Buffer>(
      size, _usage, VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
      *_memoryAllocator);

  // and bind all descriptors to it
  _descriptorBuffer->setData(std::span(reinterpret_cast<const std::byte*>(_descriptors.data()), _descriptors.size()),
                             commandBuffer);
}

void DescriptorBuffer::bind(VkPipelineBindPoint bindPoint,
                            const VkPipelineLayout& pipelineLayout,
                            const CommandBuffer& commandBuffer) {
  auto bufferBinding = VkDescriptorBufferBindingInfoEXT{VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT, nullptr,
                                                        _descriptorBuffer->getDeviceAddress(*_device), _usage};
  std::vector<VkDeviceSize> offsets(_layoutSize.size());
  auto setSize = std::reduce(_layoutSize.begin(), _layoutSize.end());
  std::exclusive_scan(_layoutSize.begin(), _layoutSize.end(), offsets.begin(), VkDeviceSize{0});
  for (auto&& offset : offsets) offset += setSize * (_currentBind % _frame);
  _currentBind++;

  std::vector<uint32_t> bufIndex(_layoutSize.size(), 0);
  // we use 1 buffer for the whole shader
  vkCmdBindDescriptorBuffersEXT(commandBuffer.getCommandBuffer(), 1, &bufferBinding);
  // but specify offsets for every set
  vkCmdSetDescriptorBufferOffsetsEXT(commandBuffer.getCommandBuffer(), bindPoint, pipelineLayout, 0,
                                     _descriptorLayouts.size(), bufIndex.data(), offsets.data());
}

DescriptorPool::DescriptorPool(DescriptorPoolSize poolSize, const Device& device) {
  _device = &device;

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

void DescriptorPool::notify(const std::vector<VkDescriptorSetLayoutBinding>& layoutInfo, int number) {
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

DescriptorSet::DescriptorSet(const std::vector<DescriptorSetLayout*>& layouts,
                             DescriptorPool& descriptorPool,
                             const Device& device) {
  _descriptorLayouts = layouts;
  _descriptorPool = &descriptorPool;
  _device = &device;

  // pre-allocate for the first frame only
  _allocateDescriptorSetsForNextFrame();

  for (auto&& layout : _descriptorLayouts) _bindingNumber += layout->getLayoutInfo().size();
}

void DescriptorSet::_allocateDescriptorSetsForNextFrame() {
  std::vector<VkDescriptorSet> setFrame;
  for (auto&& layout : _descriptorLayouts) {
    VkDescriptorSet descriptorSet;
    auto descriptorLayout = layout->getDescriptorSetLayout();
    VkDescriptorSetAllocateInfo allocInfo{.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                          .descriptorPool = _descriptorPool->getDescriptorPool(),
                                          .descriptorSetCount = 1,
                                          .pSetLayouts = &descriptorLayout};
    auto sts = vkAllocateDescriptorSets(_device->getLogicalDevice(), &allocInfo, &descriptorSet);
    if (sts != VK_SUCCESS) {
      std::ostringstream descriptors;
      descriptors << "failed to allocate descriptor sets: allocated sets: "
                  << _descriptorPool->getDescriptorSetsNumber() << ", descriptors: ";
      for (auto&& [key, value] : _descriptorPool->getDescriptorsNumber()) {
        descriptors << key << ":" << value << " ";
      }
      throw std::runtime_error(descriptors.str());
    }

    setFrame.push_back(descriptorSet);
  }
  _descriptorSet.push_back(std::move(setFrame));
}

int DescriptorSet::_calculateDescriptorSetIndex() {
  std::size_t acc = 0;
  int i = 0;
  for (const auto& layout : _descriptorLayouts) {
    acc += layout->getLayoutInfo().size();
    if (_number < acc) break;
    ++i;
  }

  return i;
}

void DescriptorSet::initialize(const CommandBuffer& commandBuffer) {
  std::vector<std::vector<VkDescriptorImageInfo>> imageInfoAcc;
  std::vector<std::vector<VkDescriptorBufferInfo>> bufferInfoAcc;
  std::vector<VkWriteDescriptorSet> descriptorWritesAcc;
  for (auto&& resource : _resources) {
    if (_number == _bindingNumber) {
      _frame++;
      _number = 0;
      // alocate descriptors for a new frame
      _allocateDescriptorSetsForNextFrame();
    }
    auto index = _calculateDescriptorSetIndex();
    std::size_t bindingIndex = _number;
    for (int i = 0; i < index; ++i) {
      bindingIndex -= _descriptorLayouts[i]->getLayoutInfo().size();
    }
    if (resource.type == Resource::Type::BUFFER) {
      std::vector<VkDescriptorBufferInfo> bufferInfos(resource.buffers.size());
      for (int i = 0; i < resource.buffers.size(); i++) {
        bufferInfos[i] = VkDescriptorBufferInfo{.buffer = resource.buffers[i]->getBuffer(),
                                                .offset = 0,
                                                .range = resource.buffers[i]->getSize()};
      }
      bufferInfoAcc.push_back(std::move(bufferInfos));
      VkWriteDescriptorSet descriptorSet = {
          .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
          .dstSet = _descriptorSet[_frame][index],
          .dstBinding = _descriptorLayouts[index]->getLayoutInfo()[bindingIndex].binding,
          .dstArrayElement = 0,
          .descriptorCount = _descriptorLayouts[index]->getLayoutInfo()[bindingIndex].descriptorCount,
          .descriptorType = _descriptorLayouts[index]->getLayoutInfo()[bindingIndex].descriptorType,
          .pBufferInfo = bufferInfoAcc.back().data()};
      descriptorWritesAcc.push_back(descriptorSet);
    }
    if (resource.type == Resource::Type::TEXTURE) {
      std::vector<VkDescriptorImageInfo> imageInfos(resource.textures.size());
      for (int i = 0; i < resource.textures.size(); i++) {
        // change layout if it's not general
        auto&& image = resource.textures[i]->getImageView().getImage();
        if (image.getImageLayout() != VK_IMAGE_LAYOUT_GENERAL) {
          auto dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT |
                               VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
          VkPipelineStageFlags2 dstStageMask = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                                               VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                                               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                               VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
          if (image.getAspectMask() & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) {
            dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
                            VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            dstStageMask = VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT |
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT |
                           VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
          }
          image.changeLayout(image.getImageLayout(), VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
                             dstStageMask, dstAccessMask, commandBuffer);
        }
        // generate mip maps if needed
        if (image.getMipMapGenerated() == false && image.getMipMapNumber() > 1) image.generateMipmaps(commandBuffer);

        imageInfos[i] = VkDescriptorImageInfo{
            .imageView = resource.textures[i]->getImageView().getImageView(),
            .imageLayout = resource.textures[i]->getImageView().getImage().getImageLayout()};
        auto&& sampler = resource.textures[i]->getSampler();
        if (sampler) imageInfos[i].sampler = sampler->getSampler();
      }
      imageInfoAcc.push_back(std::move(imageInfos));
      VkWriteDescriptorSet descriptorSet = {
          .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
          .dstSet = _descriptorSet[_frame][index],
          .dstBinding = _descriptorLayouts[index]->getLayoutInfo()[bindingIndex].binding,
          .dstArrayElement = 0,
          .descriptorCount = _descriptorLayouts[index]->getLayoutInfo()[bindingIndex].descriptorCount,
          .descriptorType = _descriptorLayouts[index]->getLayoutInfo()[bindingIndex].descriptorType,
          .pImageInfo = imageInfoAcc.back().data()};
      descriptorWritesAcc.push_back(descriptorSet);
    }

    _number++;
  }
  vkUpdateDescriptorSets(_device->getLogicalDevice(), descriptorWritesAcc.size(), descriptorWritesAcc.data(), 0,
                         nullptr);
}

void DescriptorSet::bind(VkPipelineBindPoint bindPoint,
                         const VkPipelineLayout& pipelineLayout,
                         const CommandBuffer& commandBuffer) {
  auto&& descriptorSet = _descriptorSet[_currentBind % (_frame + 1)];
  vkCmdBindDescriptorSets(commandBuffer.getCommandBuffer(), bindPoint, pipelineLayout, 0, descriptorSet.size(),
                          descriptorSet.data(), 0, nullptr);
  _currentBind++;
}

DescriptorSet::~DescriptorSet() {
  for (auto&& layout : _descriptorLayouts) _descriptorPool->notify(layout->getLayoutInfo(), -1);
}
