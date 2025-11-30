module Device;
using namespace RenderGraph;

Device::Device(const Surface& surface, const Instance& instance) {
  VkPhysicalDeviceFeatures deviceFeatures{
      .geometryShader = true,
      .tessellationShader = true,
      .fillModeNonSolid = true,
      .samplerAnisotropy = true,
  };

  VkPhysicalDeviceDynamicRenderingFeatures dynamicRenderingFeature{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES,
      .dynamicRendering = true};
  VkPhysicalDeviceTimelineSemaphoreFeatures timelineFeatures = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES,
      .timelineSemaphore = true,
  };
  // for timestamps reset
  VkPhysicalDeviceHostQueryResetFeatures resetFeatures = {
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_HOST_QUERY_RESET_FEATURES,
      .hostQueryReset = true};
  VkPhysicalDeviceDescriptorBufferFeaturesEXT descriptorBufferFeatures{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_FEATURES_EXT,
      .descriptorBuffer = true};
  VkPhysicalDeviceBufferDeviceAddressFeatures bufferDeviceAddressFeatures{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES,
      .bufferDeviceAddress = true
  };

  vkb::PhysicalDeviceSelector deviceSelector(instance.getInstance());
  deviceSelector.set_required_features(deviceFeatures);
  deviceSelector.allow_any_gpu_device_type(false);
  // not part of Vulkan 1.3 core
  deviceSelector.add_required_extension("VK_EXT_descriptor_buffer");
  // VK_KHR_SWAPCHAIN_EXTENSION_NAME is added by default
  auto deviceSelectorResult = deviceSelector.set_surface(surface.getSurface()).select();
  if (!deviceSelectorResult) {
    throw std::runtime_error(deviceSelectorResult.error().message());
  }
  auto devicePhysical = deviceSelectorResult.value();

  vkb::DeviceBuilder builder{devicePhysical};
  builder.add_pNext(&descriptorBufferFeatures);
  builder.add_pNext(&dynamicRenderingFeature);
  builder.add_pNext(&timelineFeatures);
  builder.add_pNext(&resetFeatures);
  builder.add_pNext(&bufferDeviceAddressFeatures);
  auto builderResult = builder.build();
  if (!builderResult) {
    throw std::runtime_error(builderResult.error().message());
  }
  _device = builderResult.value();
  volkLoadDevice(_device.device);

  // request properties
  _descriptorBufferProperties = VkPhysicalDeviceDescriptorBufferPropertiesEXT{
      VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_PROPERTIES_EXT};
  VkPhysicalDeviceProperties2 properties2{.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
                                          .pNext = &_descriptorBufferProperties};
  vkGetPhysicalDeviceProperties2(getPhysicalDevice(), &properties2);
  _deviceProperties = properties2.properties;

  uint32_t queueFamilyCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(getPhysicalDevice(), &queueFamilyCount, nullptr);
  _queueFamilyProperties.resize(queueFamilyCount);
  vkGetPhysicalDeviceQueueFamilyProperties(getPhysicalDevice(), &queueFamilyCount, _queueFamilyProperties.data());
}

const VkQueueFamilyProperties& Device::getQueueFamilyProperties(vkb::QueueType type) const noexcept {
  return _queueFamilyProperties[getQueueIndex(type)];
}

const VkPhysicalDeviceProperties& Device::getDeviceProperties() const noexcept { return _deviceProperties; }

const VkPhysicalDeviceDescriptorBufferPropertiesEXT& Device::getDescriptorBufferProperties() const noexcept {
  return _descriptorBufferProperties;
}

bool Device::isFormatFeatureSupported(VkFormat format,
                                      VkImageTiling tiling,
                                      VkFormatFeatureFlagBits featureFlagBit) const {
  VkFormatProperties props;
  vkGetPhysicalDeviceFormatProperties(getPhysicalDevice(), format, &props);

  if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & featureFlagBit) == featureFlagBit) {
    return true;
  } else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & featureFlagBit) == featureFlagBit) {
    return true;
  }

  return false;
}

const VkDevice Device::getLogicalDevice() const noexcept { return _device.device; }

const VkPhysicalDevice Device::getPhysicalDevice() const noexcept { return _device.physical_device.physical_device; }

const vkb::Device& Device::getDevice() const noexcept { return _device; }

const VkQueue Device::getQueue(vkb::QueueType type) const {
  auto queueResult = _device.get_dedicated_queue(type);
  if (!queueResult) {
    queueResult = _device.get_queue(type);
    // use default queue that should support everything
    if (!queueResult) {
      queueResult = _device.get_queue(vkb::QueueType::present);
    }
  }

  return queueResult.value();
}

int Device::getQueueIndex(vkb::QueueType type) const {
  auto queueResult = _device.get_dedicated_queue_index(type);
  if (!queueResult) {
    queueResult = _device.get_queue_index(type);
    // use default queue that should support everything
    if (!queueResult) {
      queueResult = _device.get_queue_index(vkb::QueueType::present);
    }
  }

  return queueResult.value();
}

Device::~Device() { vkb::destroy_device(_device); }