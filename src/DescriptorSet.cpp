module DescriptorSet;
import <sstream>;
import <ranges>;
import <algorithm>;
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

DescriptorBuffer::DescriptorBuffer(const MemoryAllocator& memoryAllocator, const Device& device) { _memoryAllocator = &memoryAllocator; _device = &device; }

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
  auto descSize = _getDescriptorSize(descriptorType);
  std::vector<uint8_t> descriptorCPU(descSize);
  vkGetDescriptorEXT(_device->getLogicalDevice(), &getInfo, descSize, descriptorCPU.data());
  std::ranges::copy(descriptorCPU, std::back_inserter(_descriptors));  
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
  auto descSize = _getDescriptorSize(descriptorType);
  std::vector<uint8_t> descriptorCPU(descSize);
  vkGetDescriptorEXT(_device->getLogicalDevice(), &getInfo, descSize, descriptorCPU.data());
  std::ranges::copy(descriptorCPU, std::back_inserter(_descriptors));
}

const Buffer* DescriptorBuffer::getBuffer(const CommandBuffer& commandBuffer) {
  if (_descriptorBuffer == nullptr) {
    // first need to allocate the buffer itself
    int size = _descriptors.size();
    _descriptorBuffer = std::make_unique<Buffer>(
        size,
        VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT, *_memoryAllocator);

    // and bind all descriptors to it
    _descriptorBuffer->setData(std::span(reinterpret_cast<const std::byte*>(_descriptors.data()), _descriptors.size()),
                               commandBuffer);
  }
  return _descriptorBuffer.get();
}

#if 0
auto bufferBinding = VkDescriptorBufferBindingInfoEXT{
    VK_STRUCTURE_TYPE_DESCRIPTOR_BUFFER_BINDING_INFO_EXT, nullptr, _address,
    VK_BUFFER_USAGE_RESOURCE_DESCRIPTOR_BUFFER_BIT_EXT | VK_BUFFER_USAGE_SAMPLER_DESCRIPTOR_BUFFER_BIT_EXT |
        VK_BUFFER_USAGE_PUSH_DESCRIPTORS_DESCRIPTOR_BUFFER_BIT_EXT};

VkDeviceSize offset = 0;
uint32_t bufIndex = 0;
vkCmdBindDescriptorBuffersEXT(commandBuffer.getCommandBuffer(), 1, &bufferBinding);
vkCmdSetDescriptorBufferOffsetsEXT(commandBuffer.getCommandBuffer(), stage, pipelineLayout, 0, 1, &bufIndex, &offset);
#endif