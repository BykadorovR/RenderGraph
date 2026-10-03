module;

#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 0

#include <algorithm>
#include <cstddef>
#include <map>
#include <memory>
#include <numeric>
#include <sstream>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>
#include <vk_mem_alloc.h>
#include <volk.h>

module DescriptorBuffer;

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
                                   const Device& device,
                                   std::uint32_t firstSet) {
  _memoryAllocator = &memoryAllocator;
  _device = &device;
  _descriptorLayouts = layouts;
  _firstSet = firstSet;

  device.getFeatureProperties(_descriptorBufferProperties);

  const VkDeviceSize alignment = _descriptorBufferProperties.descriptorBufferOffsetAlignment;
  const auto alignUp = [](VkDeviceSize value, VkDeviceSize alignment) {
    return (value + alignment - 1) / alignment * alignment;
  };

  // one buffer for the entire shader, but separate offsets for the sets inside the buffer
  _offsets.resize(layouts.size());
  _layoutSize.resize(layouts.size());
  for (int id = 0; id < layouts.size(); id++) {
    auto&& layout = layouts[id];
    for (int i = 0; i < layout->getLayoutInfo().size(); i++) {
      const auto& bindingInfo = layout->getLayoutInfo()[i];
      VkDeviceSize offset;
      vkGetDescriptorSetLayoutBindingOffsetEXT(device.getLogicalDevice(), layout->getDescriptorSetLayout(),
                                               bindingInfo.binding, &offset);
      const auto descriptorStride =
          bindingInfo.descriptorType == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER &&
                  !_descriptorBufferProperties.combinedImageSamplerDescriptorSingleArray
              ? _descriptorBufferProperties.sampledImageDescriptorSize
              : _getDescriptorSize(bindingInfo.descriptorType);
      for (int j = 0; j < layout->getLayoutInfo()[i].descriptorCount; j++) {
        _offsets[id].push_back(offset + descriptorStride * j);
      }
    }

    VkDeviceSize layoutSize;
    vkGetDescriptorSetLayoutSizeEXT(device.getLogicalDevice(), layout->getDescriptorSetLayout(), &layoutSize);
    _layoutSize[id] = alignUp(layoutSize, alignment);
  }
}

int DescriptorBuffer::_getDescriptorSize(VkDescriptorType descriptorType) {
  switch (descriptorType) {
    case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
      return _descriptorBufferProperties.combinedImageSamplerDescriptorSize;
    case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
      return _descriptorBufferProperties.sampledImageDescriptorSize;
    case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
      return _descriptorBufferProperties.storageImageDescriptorSize;
    case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
      return _descriptorBufferProperties.uniformBufferDescriptorSize;
    case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
      return _descriptorBufferProperties.storageBufferDescriptorSize;
    default:
      throw std::runtime_error("Unsupported descriptor type for descriptor buffer");
  }
}

void DescriptorBuffer::_writeDescriptor(VkDescriptorGetInfoEXT info,
                                        std::uint32_t frame,
                                        std::uint32_t set,
                                        std::size_t bindingIndex,
                                        std::size_t arrayIndex) {
  const auto& layoutInfo = _descriptorLayouts[set]->getLayoutInfo();
  std::size_t descriptorOffsetIndex = 0;
  for (std::size_t currentBinding = 0; currentBinding < bindingIndex; ++currentBinding)
    descriptorOffsetIndex += layoutInfo[currentBinding].descriptorCount;

  const auto descriptorSize = static_cast<std::size_t>(_getDescriptorSize(info.type));
  std::vector<uint8_t> encodedDescriptor(descriptorSize);
  vkGetDescriptorEXT(_device->getLogicalDevice(), &info, descriptorSize, encodedDescriptor.data());

  const VkDeviceSize frameSize = std::reduce(_layoutSize.begin(), _layoutSize.end(), VkDeviceSize{0});
  std::vector<VkDeviceSize> setOffsets(_layoutSize.size());
  std::exclusive_scan(_layoutSize.begin(), _layoutSize.end(), setOffsets.begin(), VkDeviceSize{0});
  const VkDeviceSize descriptorBase = frameSize * frame + setOffsets[set];

  const auto writeRange = [&](std::size_t sourceOffset, std::size_t size, VkDeviceSize destinationOffset) {
    if (sourceOffset + size > encodedDescriptor.size() || destinationOffset + size > _descriptors.size())
      throw std::out_of_range("Descriptor data range is out of bounds");
    std::copy_n(encodedDescriptor.begin() + sourceOffset, size, _descriptors.begin() + destinationOffset);
    if (_descriptorBuffer == nullptr) return;
    const auto result = vmaCopyMemoryToAllocation(_memoryAllocator->getAllocator(),
                                                  encodedDescriptor.data() + sourceOffset,
                                                  _descriptorBuffer->getAllocation(), destinationOffset, size);
    if (result != VK_SUCCESS)
      throw std::runtime_error("Can't update descriptor buffer data: " + std::to_string(result));
  };

  if (info.type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER &&
      !_descriptorBufferProperties.combinedImageSamplerDescriptorSingleArray) {
    const auto imageSize = static_cast<std::size_t>(_descriptorBufferProperties.sampledImageDescriptorSize);
    const auto samplerSize = static_cast<std::size_t>(_descriptorBufferProperties.samplerDescriptorSize);
    const VkDeviceSize bindingOffset = _offsets[set][descriptorOffsetIndex];
    writeRange(0, imageSize, descriptorBase + bindingOffset + imageSize * arrayIndex);
    writeRange(imageSize, samplerSize,
               descriptorBase + bindingOffset + imageSize * layoutInfo[bindingIndex].descriptorCount +
                   samplerSize * arrayIndex);
    return;
  }

  writeRange(0, descriptorSize, descriptorBase + _offsets[set][descriptorOffsetIndex + arrayIndex]);
}

void DescriptorBuffer::_add(VkDescriptorGetInfoEXT info) {
  auto setSize = std::reduce(_layoutSize.begin(), _layoutSize.end());
  // allign for the whole set (for frames in flight)
  if (_set == 0) {
    _descriptors.resize(setSize * (_frame + 1));
  }

  _writeDescriptor(info, _frame, _set, _binding.first, _binding.second);

  // calculate next frame, set, binning
  if (_binding.second < _descriptorLayouts[_set]->getLayoutInfo()[_binding.first].descriptorCount) _binding.second++;
  if (_binding.second == _descriptorLayouts[_set]->getLayoutInfo()[_binding.first].descriptorCount) {
    _binding.first++;
    _binding.second = 0;
  }

  if (_descriptorLayouts[_set]->getLayoutInfo().size() == _binding.first) {
    _binding.first = 0;
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
          auto dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT |
                               VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
          VkPipelineStageFlags2 dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |
                                               VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                                               VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                                               VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
          if (image.getAspectMask() & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) {
            dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT |
                            VK_ACCESS_2_SHADER_WRITE_BIT;
            dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                           VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                           VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
          }
          image.changeLayout(image.getImageLayout(), VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0,
                             dstStageMask, dstAccessMask, commandBuffer);
        }
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

  // Descriptor buffers are explicitly created as persistently mapped host-visible buffers.
  const auto descriptorData =
      std::span(reinterpret_cast<const std::byte*>(_descriptors.data()), _descriptors.size());
  const auto result = vmaCopyMemoryToAllocation(_memoryAllocator->getAllocator(), descriptorData.data(),
                                                _descriptorBuffer->getAllocation(), 0, descriptorData.size());
  if (result != VK_SUCCESS) {
    throw std::runtime_error("Can't upload descriptor buffer data: " + std::to_string(result));
  }

  const VkBufferMemoryBarrier2 descriptorBarrier{
      .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
      .srcStageMask = VK_PIPELINE_STAGE_2_HOST_BIT,
      .srcAccessMask = VK_ACCESS_2_HOST_WRITE_BIT,
      .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
      .dstAccessMask = VK_ACCESS_2_DESCRIPTOR_BUFFER_READ_BIT_EXT,
      .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
      .buffer = _descriptorBuffer->getBuffer(),
      .offset = 0,
      .size = descriptorData.size(),
  };
  const VkDependencyInfo dependencyInfo{
      .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
      .bufferMemoryBarrierCount = 1,
      .pBufferMemoryBarriers = &descriptorBarrier,
  };
  vkCmdPipelineBarrier2(commandBuffer.getCommandBuffer(), &dependencyInfo);
}

void DescriptorBuffer::update(std::uint32_t binding,
                              std::vector<Texture*> textures,
                              const CommandBuffer& commandBuffer,
                              std::uint32_t frame,
                              std::uint32_t set) {
  if (_descriptorBuffer == nullptr) throw std::runtime_error("Descriptor buffer is not initialized");
  if (frame >= static_cast<std::uint32_t>(_frame)) throw std::out_of_range("Descriptor frame is out of range");
  if (set >= _descriptorLayouts.size()) throw std::out_of_range("Descriptor set is out of range");

  const auto& layoutInfo = _descriptorLayouts[set]->getLayoutInfo();
  const auto bindingInfo = std::find_if(layoutInfo.begin(), layoutInfo.end(),
                                        [binding](const auto& info) { return info.binding == binding; });
  if (bindingInfo == layoutInfo.end()) throw std::out_of_range("Descriptor binding is out of range");
  if (textures.size() != bindingInfo->descriptorCount)
    throw std::invalid_argument("Texture count does not match descriptor count");
  if (bindingInfo->descriptorType != VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE &&
      bindingInfo->descriptorType != VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER &&
      bindingInfo->descriptorType != VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
    throw std::invalid_argument("Descriptor binding is not a texture");

  const auto bindingIndex = static_cast<std::size_t>(std::distance(layoutInfo.begin(), bindingInfo));

  for (std::size_t index = 0; index < textures.size(); ++index) {
    auto* texture = textures[index];
    if (texture == nullptr) throw std::invalid_argument("Texture is null");

    auto& image = texture->getImageView().getImage();
    if (image.getImageLayout() != VK_IMAGE_LAYOUT_GENERAL) {
      auto dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT |
                           VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
      VkPipelineStageFlags2 dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |
                                           VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                                           VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                                           VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
      if (image.getAspectMask() & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) {
        dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT |
                        VK_ACCESS_2_SHADER_WRITE_BIT;
        dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                       VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
      }
      image.changeLayout(image.getImageLayout(), VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0,
                         dstStageMask, dstAccessMask, commandBuffer);
    }

    VkDescriptorImageInfo imageInfo{
        .imageView = texture->getImageView().getImageView(),
        .imageLayout = texture->getImageView().getImage().getImageLayout(),
    };
    if (const auto& sampler = texture->getSampler()) imageInfo.sampler = sampler->getSampler();

    VkDescriptorGetInfoEXT descriptorInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT};
    descriptorInfo.type = bindingInfo->descriptorType;
    switch (descriptorInfo.type) {
      case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
        descriptorInfo.data.pSampledImage = &imageInfo;
        break;
      case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
        descriptorInfo.data.pCombinedImageSampler = &imageInfo;
        break;
      case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
        descriptorInfo.data.pStorageImage = &imageInfo;
        break;
      default:
        throw std::invalid_argument("Descriptor binding is not a texture");
    }

    _writeDescriptor(descriptorInfo, frame, set, bindingIndex, index);
  }

  std::size_t bindingsPerFrame = 0;
  for (const auto* layout : _descriptorLayouts) bindingsPerFrame += layout->getLayoutInfo().size();
  std::size_t resourceIndex = frame * bindingsPerFrame + bindingIndex;
  for (std::uint32_t currentSet = 0; currentSet < set; ++currentSet)
    resourceIndex += _descriptorLayouts[currentSet]->getLayoutInfo().size();
  if (resourceIndex < _resources.size())
    _resources[resourceIndex] = Resource{.type = Resource::Type::TEXTURE, .textures = std::move(textures)};

}

void DescriptorBuffer::update(std::uint32_t binding,
                              std::vector<Buffer*> buffers,
                              std::uint32_t frame,
                              std::uint32_t set) {
  if (_descriptorBuffer == nullptr) throw std::runtime_error("Descriptor buffer is not initialized");
  if (frame >= static_cast<std::uint32_t>(_frame)) throw std::out_of_range("Descriptor frame is out of range");
  if (set >= _descriptorLayouts.size()) throw std::out_of_range("Descriptor set is out of range");

  const auto& layoutInfo = _descriptorLayouts[set]->getLayoutInfo();
  const auto bindingInfo = std::find_if(layoutInfo.begin(), layoutInfo.end(),
                                        [binding](const auto& info) { return info.binding == binding; });
  if (bindingInfo == layoutInfo.end()) throw std::out_of_range("Descriptor binding is out of range");
  if (buffers.size() != bindingInfo->descriptorCount)
    throw std::invalid_argument("Buffer count does not match descriptor count");
  if (bindingInfo->descriptorType != VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER &&
      bindingInfo->descriptorType != VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
    throw std::invalid_argument("Descriptor binding is not a buffer");

  const auto bindingIndex = static_cast<std::size_t>(std::distance(layoutInfo.begin(), bindingInfo));

  for (std::size_t index = 0; index < buffers.size(); ++index) {
    auto* buffer = buffers[index];
    if (buffer == nullptr) throw std::invalid_argument("Buffer is null");

    const VkDescriptorAddressInfoEXT addressInfo{
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_ADDRESS_INFO_EXT,
        .address = buffer->getDeviceAddress(*_device),
        .range = buffer->getSize(),
        .format = VK_FORMAT_UNDEFINED,
    };
    VkDescriptorGetInfoEXT descriptorInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_GET_INFO_EXT};
    descriptorInfo.type = bindingInfo->descriptorType;
    switch (descriptorInfo.type) {
      case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
        descriptorInfo.data.pUniformBuffer = &addressInfo;
        break;
      case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
        descriptorInfo.data.pStorageBuffer = &addressInfo;
        break;
      default:
        throw std::invalid_argument("Descriptor binding is not a buffer");
    }

    _writeDescriptor(descriptorInfo, frame, set, bindingIndex, index);
  }

  std::size_t bindingsPerFrame = 0;
  for (const auto* layout : _descriptorLayouts) bindingsPerFrame += layout->getLayoutInfo().size();
  std::size_t resourceIndex = frame * bindingsPerFrame + bindingIndex;
  for (std::uint32_t currentSet = 0; currentSet < set; ++currentSet)
    resourceIndex += _descriptorLayouts[currentSet]->getLayoutInfo().size();
  if (resourceIndex < _resources.size())
    _resources[resourceIndex] = Resource{.type = Resource::Type::BUFFER, .buffers = std::move(buffers)};

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
  vkCmdSetDescriptorBufferOffsetsEXT(commandBuffer.getCommandBuffer(), bindPoint, pipelineLayout, _firstSet,
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
                                      .flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
                                      .maxSets = static_cast<uint32_t>(poolSize.descriptorSets),
                                      .poolSizeCount = static_cast<uint32_t>(poolSizes.size()),
                                      .pPoolSizes = poolSizes.data()};

  if (vkCreateDescriptorPool(device.getLogicalDevice(), &poolInfo, nullptr, &_descriptorPool) != VK_SUCCESS) {
    throw std::runtime_error("failed to create descriptor pool!");
  }
}

void DescriptorPool::notify(const std::vector<VkDescriptorSetLayoutBinding>& layoutInfo, int number) {
  for (auto&& info : layoutInfo) {
    _descriptorTypes[info.descriptorType] += info.descriptorCount * number;
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
                             const Device& device,
                             std::uint32_t firstSet) {
  _descriptorLayouts = layouts;
  _descriptorPool = &descriptorPool;
  _device = &device;
  _firstSet = firstSet;

  // pre-allocate for the first frame only
  _allocateDescriptorSetsForNextFrame();

  for (auto&& layout : _descriptorLayouts) _bindingNumber += layout->getLayoutInfo().size();
}

void DescriptorSet::_allocateDescriptorSetsForNextFrame() {
  std::vector<VkDescriptorSetLayout> layouts;
  layouts.reserve(_descriptorLayouts.size());
  for (const auto* layout : _descriptorLayouts) layouts.push_back(layout->getDescriptorSetLayout());

  std::vector<VkDescriptorSet> setFrame(layouts.size());
  const VkDescriptorSetAllocateInfo allocInfo{
      .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
      .descriptorPool = _descriptorPool->getDescriptorPool(),
      .descriptorSetCount = static_cast<std::uint32_t>(layouts.size()),
      .pSetLayouts = layouts.data(),
  };
  const auto status = vkAllocateDescriptorSets(_device->getLogicalDevice(), &allocInfo, setFrame.data());
  if (status != VK_SUCCESS) {
    std::ostringstream descriptors;
    descriptors << "failed to allocate descriptor sets: allocated sets: "
                << _descriptorPool->getDescriptorSetsNumber() << ", descriptors: ";
    for (auto&& [key, value] : _descriptorPool->getDescriptorsNumber()) {
      descriptors << key << ":" << value << " ";
    }
    throw std::runtime_error(descriptors.str());
  }

  for (const auto* layout : _descriptorLayouts) _descriptorPool->notify(layout->getLayoutInfo(), 1);
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
          auto dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT |
                               VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
          VkPipelineStageFlags2 dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |
                                               VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                                               VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                                               VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
          if (image.getAspectMask() & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) {
            dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT |
                            VK_ACCESS_2_SHADER_WRITE_BIT;
            dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                           VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                           VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
          }
          image.changeLayout(image.getImageLayout(), VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0,
                             dstStageMask, dstAccessMask, commandBuffer);
        }
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

void DescriptorSet::update(std::uint32_t binding,
                           std::vector<Texture*> textures,
                           const CommandBuffer& commandBuffer,
                           std::uint32_t frame,
                           std::uint32_t set) {
  if (frame >= _descriptorSet.size()) throw std::out_of_range("Descriptor frame is out of range");
  if (set >= _descriptorLayouts.size()) throw std::out_of_range("Descriptor set is out of range");

  const auto& layoutInfo = _descriptorLayouts[set]->getLayoutInfo();
  const auto bindingInfo = std::find_if(layoutInfo.begin(), layoutInfo.end(),
                                        [binding](const auto& info) { return info.binding == binding; });
  if (bindingInfo == layoutInfo.end()) throw std::out_of_range("Descriptor binding is out of range");
  if (textures.size() != bindingInfo->descriptorCount)
    throw std::invalid_argument("Texture count does not match descriptor count");
  if (bindingInfo->descriptorType != VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE &&
      bindingInfo->descriptorType != VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER &&
      bindingInfo->descriptorType != VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
    throw std::invalid_argument("Descriptor binding is not a texture");

  std::vector<VkDescriptorImageInfo> imageInfos;
  imageInfos.reserve(textures.size());
  for (auto* texture : textures) {
    if (texture == nullptr) throw std::invalid_argument("Texture is null");
    auto& image = texture->getImageView().getImage();
    if (image.getImageLayout() != VK_IMAGE_LAYOUT_GENERAL) {
      auto dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT |
                           VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
      VkPipelineStageFlags2 dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |
                                           VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                                           VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT |
                                           VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
      if (image.getAspectMask() & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) {
        dstAccessMask = VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT |
                        VK_ACCESS_2_SHADER_WRITE_BIT;
        dstStageMask = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                       VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                       VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT;
      }
      image.changeLayout(image.getImageLayout(), VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0,
                         dstStageMask, dstAccessMask, commandBuffer);
    }

    VkDescriptorImageInfo imageInfo{
        .imageView = texture->getImageView().getImageView(),
        .imageLayout = texture->getImageView().getImage().getImageLayout(),
    };
    if (const auto& sampler = texture->getSampler()) imageInfo.sampler = sampler->getSampler();
    imageInfos.push_back(imageInfo);
  }

  const VkWriteDescriptorSet descriptorWrite{
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = _descriptorSet[frame][set],
      .dstBinding = binding,
      .dstArrayElement = 0,
      .descriptorCount = static_cast<std::uint32_t>(imageInfos.size()),
      .descriptorType = bindingInfo->descriptorType,
      .pImageInfo = imageInfos.data(),
  };
  vkUpdateDescriptorSets(_device->getLogicalDevice(), 1, &descriptorWrite, 0, nullptr);

  const auto bindingIndex = static_cast<std::size_t>(std::distance(layoutInfo.begin(), bindingInfo));
  std::size_t resourceIndex = frame * _bindingNumber + bindingIndex;
  for (std::uint32_t currentSet = 0; currentSet < set; ++currentSet)
    resourceIndex += _descriptorLayouts[currentSet]->getLayoutInfo().size();
  if (resourceIndex < _resources.size())
    _resources[resourceIndex] = Resource{.type = Resource::Type::TEXTURE, .textures = std::move(textures)};
}

void DescriptorSet::update(std::uint32_t binding,
                           std::vector<Buffer*> buffers,
                           std::uint32_t frame,
                           std::uint32_t set) {
  if (frame >= _descriptorSet.size()) throw std::out_of_range("Descriptor frame is out of range");
  if (set >= _descriptorLayouts.size()) throw std::out_of_range("Descriptor set is out of range");

  const auto& layoutInfo = _descriptorLayouts[set]->getLayoutInfo();
  const auto bindingInfo = std::find_if(layoutInfo.begin(), layoutInfo.end(),
                                        [binding](const auto& info) { return info.binding == binding; });
  if (bindingInfo == layoutInfo.end()) throw std::out_of_range("Descriptor binding is out of range");
  if (buffers.size() != bindingInfo->descriptorCount)
    throw std::invalid_argument("Buffer count does not match descriptor count");
  if (bindingInfo->descriptorType != VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER &&
      bindingInfo->descriptorType != VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
    throw std::invalid_argument("Descriptor binding is not a buffer");

  std::vector<VkDescriptorBufferInfo> bufferInfos;
  bufferInfos.reserve(buffers.size());
  for (auto* buffer : buffers) {
    if (buffer == nullptr) throw std::invalid_argument("Buffer is null");
    bufferInfos.push_back({.buffer = buffer->getBuffer(), .offset = 0, .range = buffer->getSize()});
  }

  const VkWriteDescriptorSet descriptorWrite{
      .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
      .dstSet = _descriptorSet[frame][set],
      .dstBinding = binding,
      .dstArrayElement = 0,
      .descriptorCount = static_cast<std::uint32_t>(bufferInfos.size()),
      .descriptorType = bindingInfo->descriptorType,
      .pBufferInfo = bufferInfos.data(),
  };
  vkUpdateDescriptorSets(_device->getLogicalDevice(), 1, &descriptorWrite, 0, nullptr);

  const auto bindingIndex = static_cast<std::size_t>(std::distance(layoutInfo.begin(), bindingInfo));
  std::size_t resourceIndex = frame * _bindingNumber + bindingIndex;
  for (std::uint32_t currentSet = 0; currentSet < set; ++currentSet)
    resourceIndex += _descriptorLayouts[currentSet]->getLayoutInfo().size();
  if (resourceIndex < _resources.size())
    _resources[resourceIndex] = Resource{.type = Resource::Type::BUFFER, .buffers = std::move(buffers)};
}

void DescriptorSet::bind(VkPipelineBindPoint bindPoint,
                         const VkPipelineLayout& pipelineLayout,
                         const CommandBuffer& commandBuffer) {
  auto&& descriptorSet = _descriptorSet[_currentBind % (_frame + 1)];
  vkCmdBindDescriptorSets(commandBuffer.getCommandBuffer(), bindPoint, pipelineLayout, _firstSet, descriptorSet.size(),
                          descriptorSet.data(), 0, nullptr);
  _currentBind++;
}

DescriptorSet::~DescriptorSet() {
  for (const auto& setFrame : _descriptorSet) {
    vkFreeDescriptorSets(_device->getLogicalDevice(), _descriptorPool->getDescriptorPool(),
                         static_cast<std::uint32_t>(setFrame.size()), setFrame.data());
    for (const auto* layout : _descriptorLayouts) _descriptorPool->notify(layout->getLayoutInfo(), -1);
  }
}
