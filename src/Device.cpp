module Device;
import <ranges>;
import <algorithm>;

using namespace RenderGraph;

Device::Device(const Surface& surface, const Instance& instance) {
  _instance = &instance;
  _surface = &surface;
}

void Device::initialize() {
  // Vulkan 1.0 features
  VkPhysicalDeviceFeatures deviceFeatures{
      .geometryShader = true,
      .tessellationShader = true,
      .fillModeNonSolid = true,
      .samplerAnisotropy = true,
  };

  // Vulkan 1.0+ features
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
      .bufferDeviceAddress = true};
  VkPhysicalDeviceDescriptorIndexingFeatures descriptorIndexingFeatures{
      .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES,
      .descriptorBindingSampledImageUpdateAfterBind = true,
      .descriptorBindingPartiallyBound = true,
      .descriptorBindingVariableDescriptorCount = true};

  vkb::PhysicalDeviceSelector deviceSelector(_instance->getInstance());
  deviceSelector.set_required_features(deviceFeatures);
  deviceSelector.allow_any_gpu_device_type(false);
  // not part of Vulkan 1.3 core
  if (std::find(_desiredExtensions.begin(), _desiredExtensions.end(), "VK_EXT_descriptor_buffer") !=
      _desiredExtensions.end())
    deviceSelector.add_desired_extension("VK_EXT_descriptor_buffer");
  // VK_KHR_SWAPCHAIN_EXTENSION_NAME is added by default
  deviceSelector.set_surface(_surface->getSurface());
  auto deviceSelectorResult = deviceSelector.select();
  if (!deviceSelectorResult) {
    throw std::runtime_error(deviceSelectorResult.error().message());
  }
  auto devicePhysical = deviceSelectorResult.value();

  vkb::DeviceBuilder builder{devicePhysical};
  if (devicePhysical.is_extension_present("VK_EXT_descriptor_buffer") &&
      std::find(_desiredExtensions.begin(), _desiredExtensions.end(), "VK_EXT_descriptor_buffer") !=
          _desiredExtensions.end())
    builder.add_pNext(&descriptorBufferFeatures);
  if (devicePhysical.is_extension_present("VK_KHR_dynamic_rendering") &&
      std::find(_desiredExtensions.begin(), _desiredExtensions.end(), "VK_KHR_dynamic_rendering") !=
          _desiredExtensions.end())
    builder.add_pNext(&dynamicRenderingFeature);
  // rest should be available
  builder.add_pNext(&timelineFeatures);
  builder.add_pNext(&resetFeatures);
  builder.add_pNext(&bufferDeviceAddressFeatures);
  builder.add_pNext(&descriptorIndexingFeatures);
  auto builderResult = builder.build();
  if (!builderResult) {
    throw std::runtime_error(builderResult.error().message());
  }
  _device = builderResult.value();
  volkLoadDevice(_device.device);

  uint32_t queueFamilyCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(getPhysicalDevice(), &queueFamilyCount, nullptr);
  _queueFamilyProperties.resize(queueFamilyCount);
  vkGetPhysicalDeviceQueueFamilyProperties(getPhysicalDevice(), &queueFamilyCount, _queueFamilyProperties.data());
}

void Device::setDesiredExtensions(const std::vector<std::string>& extensions) noexcept {
  _desiredExtensions = extensions;
}

const VkQueueFamilyProperties& Device::getQueueFamilyProperties(vkb::QueueType type) const noexcept {
  return _queueFamilyProperties[getQueueIndex(type)];
}

bool Device::isExtensionSupported(std::string name) const {
  return _device.physical_device.is_extension_present(name.c_str());
}

std::vector<std::string> Device::getDesiredExtensions() const noexcept { return _desiredExtensions; }

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