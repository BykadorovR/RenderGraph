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
  vkb::PhysicalDeviceSelector deviceSelector(instance.getInstance());
  deviceSelector.set_required_features(deviceFeatures);
  deviceSelector.allow_any_gpu_device_type(false);
  // VK_KHR_SWAPCHAIN_EXTENSION_NAME is added by default
  auto deviceSelectorResult = deviceSelector.set_surface(surface.getSurface()).select();
  if (!deviceSelectorResult) {
    throw std::runtime_error(deviceSelectorResult.error().message());
  }
  auto devicePhysical = deviceSelectorResult.value();

  vkb::DeviceBuilder builder{devicePhysical};
  builder.add_pNext(&dynamicRenderingFeature);
  builder.add_pNext(&timelineFeatures);
  builder.add_pNext(&resetFeatures);
  auto builderResult = builder.build();
  if (!builderResult) {
    throw std::runtime_error(builderResult.error().message());
  }
  _device = builderResult.value();
  volkLoadDevice(_device.device);

  // request properties
  vkGetPhysicalDeviceProperties(getPhysicalDevice(), &_deviceProperties);

  uint32_t queueFamilyCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(getPhysicalDevice(), &queueFamilyCount, nullptr);
  _queueFamilyProperties.resize(queueFamilyCount);
  vkGetPhysicalDeviceQueueFamilyProperties(getPhysicalDevice(), &queueFamilyCount, _queueFamilyProperties.data());
}

const VkQueueFamilyProperties& Device::getQueueFamilyProperties(vkb::QueueType type) const noexcept {
  return _queueFamilyProperties[getQueueIndex(type)];
}

const VkPhysicalDeviceProperties& Device::getDeviceProperties() const noexcept { return _deviceProperties; }

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